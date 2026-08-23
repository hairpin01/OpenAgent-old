from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest
import cubkit

from OpenAgentLib.Plugin.PluginsEngine import (
    _OpenAgentPluginSkillMixin,
    PluginSourceAdmissionStatus,
    parse_v2_plugin_metadata,
)
from OpenAgentLib.PluginCapabilities import CapabilityResponse
from OpenAgentLib.PluginDiscovery import inspect_v2_plugin_source
from OpenAgentLib.InstalledPluginRegistry import InstalledPluginRegistry
from OpenAgentLib.IsolatedPluginInvoker import IsolatedPluginInvoker
from OpenAgentLib.PluginHost import PluginHost
from OpenAgentLib.ToolKernel import ToolContext
from OpenAgentLib.ToolModelBoundary import ModelToolErrorCode, ModelTurnKind
from OpenAgentLib.V2Bootstrap import build_v2_tool_runtime

SRC = Path(__file__).resolve().parents[1] / "Src"
sys.path.insert(0, str(SRC))
src_package = ModuleType("Src")
src_package.__path__ = [str(SRC)]
sys.modules.setdefault("Src", src_package)
original_load_strings = cubkit.load_strings
cubkit.load_strings = lambda: (lambda key, **_kwargs: key)
try:
    OpenAgent = importlib.import_module("Src.OpenAgentMain").OpenAgent
finally:
    cubkit.load_strings = original_load_strings


ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "repo-MCUB-fork" / "OpenAgent" / "plugins"


class _Response:
    def __init__(self, status: int, body: bytes | object) -> None:
        self.status = status
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def read(self) -> bytes:
        return self._body if isinstance(self._body, bytes) else b""

    async def text(self) -> str:
        return self._body.decode() if isinstance(self._body, bytes) else ""

    async def json(self):
        return self._body


class _Session:
    def __init__(self, responses: dict[str, _Response]) -> None:
        self.responses = responses

    def get(self, url: str, **_kwargs):
        return self.responses.get(url, _Response(404, b""))


class _Harness(_OpenAgentPluginSkillMixin):
    def __init__(
        self,
        tmp_path: Path,
        responses: dict[str, _Response],
        *,
        capability_handler=None,
    ) -> None:
        self.kernel = SimpleNamespace(WORK_DIR=str(tmp_path))
        self.log = SimpleNamespace(
            info=lambda *_args: None, warning=lambda *_args: None
        )
        self._http_client = SimpleNamespace(session=self._session)
        self._plugins_cache = [{"stale": True}]
        self._tool_map_cache = object()
        self._tool_registry_cache = object()
        self._installed_plugin_registry = InstalledPluginRegistry()
        self._v2_plugin_invoker = IsolatedPluginInvoker(
            PluginHost(),
            {},
            plugin_root=tmp_path,
            openagent_source=SRC,
            capability_handler=capability_handler or (lambda *_args: {"ok": False}),
        )

        self._responses = responses

    @property
    def _v2_plugin_sources(self) -> dict[str, object]:
        return self._v2_plugin_invoker._sources

    @property
    def _plugin_files(self) -> dict[str, Path]:
        return {
            Path(record.source.path).stem: Path(record.source.path)
            for record in self._installed_plugin_registry.snapshot()
        }

    async def _session(self):
        return _Session(self._responses)


def _raw(filename: str) -> str:
    return f"https://raw.githubusercontent.com/hairpin01/repo-MCUB-fork/main/OpenAgent/plugins/{filename}"


def _load_installed(harness: _Harness) -> None:
    asyncio.run(harness._load_installed_plugins(set()))


def test_catalog_v2_parser_covers_direct_factory_and_mixed_sources() -> None:
    direct = parse_v2_plugin_metadata(
        (CANONICAL / "ast_grep.py").read_text(), "ast_grep.py"
    )
    factory = parse_v2_plugin_metadata((CANONICAL / "chat.py").read_text(), "chat.py")
    mixed = parse_v2_plugin_metadata((CANONICAL / "file.py").read_text(), "file.py")

    assert direct is not None
    assert direct["plugin_id"] == "openagent.ast_grep"
    assert direct["name"] == "Ast Grep"
    assert direct["version"] == "2.0.0"
    assert direct["api_version"] == "2"
    assert direct["author"] == "hairpin01"
    assert "ast_grep.search" in direct["tools"]
    assert direct["description"] != "?"
    assert factory is not None
    assert factory["plugin_id"] == "openagent.chat"
    assert factory["tools"]
    assert mixed is not None
    assert {"file.send", "file.download_media", "file.read_text"} <= set(mixed["tools"])


def test_catalog_filters_helpers_and_lists_fourteen_plugins(tmp_path: Path) -> None:
    catalog_url = "https://api.github.com/repos/hairpin01/repo-MCUB-fork/contents/OpenAgent/plugins"
    entries = [
        {
            "name": path.name,
            "type": "file",
            "download_url": f"https://example/{path.name}",
        }
        for path in CANONICAL.glob("*.py")
    ]
    responses = {catalog_url: _Response(200, entries)}
    responses.update(
        {
            f"https://example/{path.name}": _Response(200, path.read_bytes())
            for path in CANONICAL.glob("*.py")
        }
    )
    harness = _Harness(tmp_path, responses)

    plugins = asyncio.run(harness._fetch_repo_plugins())

    assert len(plugins) == 14
    assert {plugin["plugin_name"] for plugin in plugins}.isdisjoint(
        {"__init__", "_telegram_v2", "_resource_v2"}
    )
    assert all(plugin["author"] == "hairpin01" for plugin in plugins)
    assert all(plugin["description"] != "?" for plugin in plugins)


def test_repo_install_downloads_dependencies_and_package_init(tmp_path: Path) -> None:
    responses = {
        _raw(name): _Response(200, (CANONICAL / name).read_bytes())
        for name in ("chat.py", "_telegram_v2.py", "__init__.py")
    }
    harness = _Harness(tmp_path, responses)

    assert asyncio.run(harness._install_plugin_from_repo("chat")) == "chat"

    installed = tmp_path / "openagent_plugins"
    assert (installed / "chat.py").read_bytes() == (CANONICAL / "chat.py").read_bytes()
    assert (installed / "_telegram_v2.py").exists()
    assert (installed / "__init__.py").exists()
    assert harness._plugins_cache is None
    assert harness._v2_plugin_sources["chat"].path == (installed / "chat.py").resolve()
    assert harness._plugin_files["chat"] == (installed / "chat.py").resolve()


def test_repo_install_rolls_back_on_dependency_http_failure(tmp_path: Path) -> None:
    plugins = tmp_path / "openagent_plugins"
    plugins.mkdir()
    target = plugins / "chat.py"
    helper = plugins / "_telegram_v2.py"
    target.write_bytes(b"old chat")
    helper.write_bytes(b"shared helper")
    responses = {
        _raw("chat.py"): _Response(200, (CANONICAL / "chat.py").read_bytes()),
        _raw("_telegram_v2.py"): _Response(500, b""),
        _raw("__init__.py"): _Response(200, (CANONICAL / "__init__.py").read_bytes()),
    }

    with pytest.raises(ValueError):
        asyncio.run(_Harness(tmp_path, responses)._install_plugin_from_repo("chat"))

    assert target.read_bytes() == b"old chat"
    assert helper.read_bytes() == b"shared helper"
    assert not (plugins / "__init__.py").exists()


def test_repo_install_restores_update_after_admission_failure(tmp_path: Path) -> None:
    plugins = tmp_path / "openagent_plugins"
    plugins.mkdir()
    target = plugins / "ast_grep.py"
    target.write_bytes(b"old direct plugin")
    target.chmod(0o640)
    responses = {
        _raw(name): _Response(200, (CANONICAL / name).read_bytes())
        for name in ("ast_grep.py", "_resource_v2.py", "__init__.py")
    }
    harness = _Harness(tmp_path, responses)

    original = __import__(
        "OpenAgentLib.Plugin.PluginsEngine", fromlist=["inspect_v2_plugin_source"]
    )
    saved = original.inspect_v2_plugin_source
    original.inspect_v2_plugin_source = lambda _path: (_ for _ in ()).throw(
        ValueError("rejected")
    )
    try:
        with pytest.raises(ValueError, match="rejected"):
            asyncio.run(harness._install_plugin_from_repo("ast_grep"))
    finally:
        original.inspect_v2_plugin_source = saved

    assert target.read_bytes() == b"old direct plugin"
    assert target.stat().st_mode & 0o777 == 0o640
    assert not (plugins / "_resource_v2.py").exists()
    assert not (plugins / "__init__.py").exists()
    assert harness._v2_plugin_sources == {}
    assert harness._plugin_files == {}


def test_terminal_admission_classifies_missing_source(tmp_path: Path) -> None:
    harness = _Harness(tmp_path, {})

    _load_installed(harness)

    assert harness._sibling_plugin_admissions == {
        "terminal": PluginSourceAdmissionStatus.MISSING
    }


@pytest.mark.parametrize(
    ("tool_id", "arguments"),
    (
        ("terminal.run", {"argv": ["fixture-command"], "cwd": "."}),
        ("terminal.list_files", {"path": "."}),
    ),
)
def test_unadmitted_terminal_tools_return_classified_safe_failures(
    tmp_path: Path, tool_id: str, arguments: dict[str, object]
) -> None:
    harness = _Harness(tmp_path, {})
    _load_installed(harness)
    runtime = build_v2_tool_runtime(
        SimpleNamespace(),
        host_invoker=harness._v2_plugin_invoker,
        installed_records=harness._installed_plugin_registry.snapshot(),
        include_sibling_fallback=False,
    )
    output = runtime.boundary(ToolContext("terminal-admission-fixture")).parse(
        json.dumps({"tool": tool_id, "args": arguments})
    )

    assert output.kind is ModelTurnKind.INVALID
    assert not output.calls
    assert output.errors[0].code is ModelToolErrorCode.UNKNOWN_TOOL


def test_terminal_admission_classifies_invalid_source(tmp_path: Path) -> None:
    plugins = tmp_path / "openagent_plugins"
    plugins.mkdir()
    (plugins / "terminal.py").write_text("VALUE = 1\n", encoding="utf-8")
    harness = _Harness(tmp_path, {})

    _load_installed(harness)

    assert harness._sibling_plugin_admissions == {
        "terminal": PluginSourceAdmissionStatus.INVALID
    }
    assert "terminal" not in harness._v2_plugin_sources


def test_terminal_repo_source_is_admitted_and_handlers_execute(
    tmp_path: Path,
) -> None:
    responses = {
        _raw(name): _Response(200, (CANONICAL / name).read_bytes())
        for name in ("terminal.py", "_resource_v2.py", "__init__.py")
    }
    asyncio.run(_Harness(tmp_path, responses)._install_plugin_from_repo("terminal"))

    def capability_handler(_call, _policy, request):
        data = (
            {
                "exit_code": 0,
                "stdout": "fixture output",
                "stderr": "",
                "truncated": False,
            }
            if request.operation == "run"
            else {"entries": ["alpha.txt", "beta"]}
        )
        return CapabilityResponse(
            host_request_id=request.host_request_id,
            call_id=request.call_id,
            capability_request_id=request.capability_request_id,
            ok=True,
            data=data,
        )

    harness = _Harness(tmp_path, {}, capability_handler=capability_handler)
    _load_installed(harness)
    record = harness._installed_plugin_registry.snapshot()[0]
    terminal_run = next(
        tool for tool in record.manifest.tools if tool.canonical_id == "terminal.run"
    )
    assert terminal_run.input_schema["required"] == ("argv", "cwd")
    assert terminal_run.input_schema["additionalProperties"] is False
    assert terminal_run.input_schema["properties"]["argv"]["type"] == "array"
    assert terminal_run.output_schema["required"] == (
        "ok",
        "exit_code",
        "stdout",
        "stderr",
        "truncated",
    )
    assert "status" not in terminal_run.input_schema
    assert "status" not in terminal_run.output_schema
    runtime = build_v2_tool_runtime(
        SimpleNamespace(),
        host_invoker=harness._v2_plugin_invoker,
        installed_records=harness._installed_plugin_registry.snapshot(),
        include_sibling_fallback=False,
    )

    async def invoke(tool_id: str, arguments: dict[str, object]):
        call = runtime.registry.create_call(
            call_id=f"fixture-{tool_id.rsplit('.', 1)[-1]}",
            requested_name=tool_id,
            arguments=arguments,
            context=ToolContext("terminal-admission-fixture"),
        )
        return await harness._v2_plugin_invoker.invoke(
            call, SimpleNamespace(), retryable=False
        )

    run_outcome = asyncio.run(
        invoke("terminal.run", {"argv": ["fixture-command"], "cwd": "."})
    )
    list_outcome = asyncio.run(invoke("terminal.list_files", {"path": "."}))

    assert harness._sibling_plugin_admissions == {
        "terminal": PluginSourceAdmissionStatus.ADMITTED
    }
    assert run_outcome.response.error is None, run_outcome.response
    assert list_outcome.response.error is None, list_outcome.response
    assert run_outcome.response.result == {
        "ok": True,
        "exit_code": 0,
        "stdout": "fixture output",
        "stderr": "",
        "truncated": False,
    }
    assert list_outcome.response.result["ok"] is True
    assert tuple(list_outcome.response.result["entries"]) == ("alpha.txt", "beta")


def test_repo_install_registers_live_invoker_source(tmp_path: Path) -> None:
    class _RecordingHost:
        def __init__(self) -> None:
            self.requests = []

        async def call(self, request, **_kwargs):
            self.requests.append(request)
            return SimpleNamespace()

    def call_for(module: str) -> SimpleNamespace:
        return SimpleNamespace(
            spec=SimpleNamespace(source_module=module),
            call_id=f"catalog-{module}",
            canonical_id=f"{module}.test",
            arguments={},
            context=None,
        )

    responses = {
        _raw(name): _Response(200, (CANONICAL / name).read_bytes())
        for name in (
            "chat.py",
            "ast_grep.py",
            "_telegram_v2.py",
            "_resource_v2.py",
            "__init__.py",
        )
    }
    harness = _Harness(tmp_path, responses)
    host = _RecordingHost()
    harness._v2_plugin_invoker = IsolatedPluginInvoker(
        host,
        {},
        plugin_root=tmp_path,
        openagent_source=ROOT / "Src",
        capability_handler=lambda *_args: {"ok": False},
    )

    assert asyncio.run(harness._install_plugin_from_repo("chat")) == "chat"
    assert asyncio.run(harness._install_plugin_from_repo("ast_grep")) == "ast_grep"

    for module in ("chat", "ast_grep"):
        source = harness._v2_plugin_sources[module]
        assert harness._plugin_files[module] == source.path
        assert harness._v2_plugin_invoker._sources[module] == source
        asyncio.run(
            harness._v2_plugin_invoker.invoke(
                call_for(module), SimpleNamespace(), retryable=False
            )
        )
    assert [request.payload["module"] for request in host.requests] == [
        "openagent_plugins.chat",
        "openagent_plugins.ast_grep",
    ]


def test_registration_failure_restores_previous_source_file_and_invoker(
    tmp_path: Path,
) -> None:
    plugins = tmp_path / "openagent_plugins"
    plugins.mkdir()
    target = plugins / "chat.py"
    target.write_bytes((CANONICAL / "chat.py").read_bytes())
    old_source = inspect_v2_plugin_source(target)

    class _FailingInvoker:
        def __init__(self) -> None:
            self._sources = {"chat": old_source}

        def register_source(self, _module, _source) -> None:
            raise RuntimeError("live source update rejected")

        def unregister_source(self, module) -> None:
            self._sources.pop(module, None)

    harness = _Harness(
        tmp_path,
        {
            _raw(name): _Response(200, (CANONICAL / name).read_bytes())
            for name in ("chat.py", "_telegram_v2.py", "__init__.py")
        },
    )
    harness._v2_plugin_invoker = _FailingInvoker()

    with pytest.raises(ValueError, match="validated source registration"):
        asyncio.run(harness._install_plugin_from_repo("chat"))

    assert target.read_bytes() == (CANONICAL / "chat.py").read_bytes()
    assert harness._v2_plugin_invoker._sources == {"chat": old_source}
    assert harness._v2_plugin_invoker._sources == {"chat": old_source}


def test_catalog_install_refetches_and_keeps_current_page() -> None:
    class _Call:
        def __init__(self) -> None:
            self.answers = []

        async def answer(self, text=None, *, alert=False) -> None:
            self.answers.append((text, alert))

    class _CatalogHarness:
        strings = staticmethod(lambda key, **_kwargs: key)

        def __init__(self) -> None:
            self._plugins_cache = None
            self.rendered_pages = []
            self.looked_up_stems = []

        async def _install_plugin_from_repo(self, _name: str) -> str:
            return "chat"

        async def _fetch_repo_plugins(self):
            self._plugins_cache = [{"plugin_name": str(index)} for index in range(6)]
            return self._plugins_cache

        def _find_installed_plugin_presentation(self, *, source_stem: str):
            self.looked_up_stems.append(source_stem)
            return SimpleNamespace(display_name="Chat")

        async def _oaplugin_catalog(self, _call, page: int) -> None:
            self.rendered_pages.append(page)

    harness = _CatalogHarness()
    call = _Call()
    asyncio.run(OpenAgent._oaplugin_install(harness, call, "chat", 4))

    assert harness.rendered_pages == [4]
    assert harness.looked_up_stems == ["chat"]
    assert call.answers[-1] == ("plugin_installed_alert", True)


def test_catalog_install_error_shows_alert_without_navigation() -> None:
    class _Call:
        def __init__(self) -> None:
            self.answers = []

        async def answer(self, text=None, *, alert=False) -> None:
            self.answers.append((text, alert))

    class _CatalogHarness:
        strings = staticmethod(lambda key, **_kwargs: key)

        async def _install_plugin_from_repo(self, _name: str) -> str:
            raise ValueError("rejected")

        async def _fetch_repo_plugins(self):
            raise AssertionError("catalog must not be fetched after install error")

        async def _oaplugin_catalog(self, _call, _page: int) -> None:
            raise AssertionError("catalog must not be rendered after install error")

    call = _Call()
    asyncio.run(OpenAgent._oaplugin_install(_CatalogHarness(), call, "chat", 4))

    assert call.answers[-1] == ("generic_error", True)
