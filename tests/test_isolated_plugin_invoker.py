from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from conftest import ROOT

from OpenAgentLib.IsolatedPluginInvoker import IsolatedPluginInvoker
from OpenAgentLib.InstalledPluginRegistry import (
    InstalledPluginManifest,
    InstalledPluginRecord,
    InstalledPluginRegistry,
    InstalledPluginSource,
    InstalledPluginTool,
)
from OpenAgentLib.PluginDiscovery import inspect_v2_plugin_source
from OpenAgentLib.PluginHost import PluginHost, PluginHostRequest
from OpenAgentLib import PluginHostWorker, PluginSDK
from OpenAgentLib.ToolKernel import ToolArgumentError, ToolContext, ToolResultStatus
from OpenAgentLib.ToolPolicy import (
    ConfirmationState,
    ToolConfirmationGrant,
    ToolPolicyRequest,
)
from OpenAgentLib.V2Bootstrap import build_v2_tool_runtime


class _RuntimeApp:
    """No native service is exercised by this isolated-plugin routing test."""

    _v2_source_event = None


class _RecordingHost:
    def __init__(self) -> None:
        self.requests = []

    async def call(self, request, **_kwargs):
        self.requests.append(request)
        return SimpleNamespace()


class _BlockingHost:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def call(self, _request, **_kwargs):
        self.started.set()
        await self.release.wait()
        return SimpleNamespace()


def _call_for(module: str) -> SimpleNamespace:
    return SimpleNamespace(
        spec=SimpleNamespace(source_module=module),
        call_id="isolated-payload",
        canonical_id="chat.send",
        arguments={},
        context=None,
    )


def _record_for(source) -> InstalledPluginRecord:
    return InstalledPluginRecord(
        plugin_id="openagent.chat",
        source=InstalledPluginSource(source.path, source.digest),
        manifest=InstalledPluginManifest(
            manifest_version="2",
            api_version="2",
            version="1",
            entrypoint="openagent_plugins.chat.HANDLERS",
            capabilities=frozenset({"network"}),
            tools=(
                InstalledPluginTool(
                    canonical_id="chat.send", capabilities=frozenset({"network"})
                ),
            ),
        ),
    )


def test_eval_plugin_executes_only_through_isolated_host() -> None:
    plugin_root = ROOT.parent / "repo-MCUB-fork" / "OpenAgent"
    source = inspect_v2_plugin_source(plugin_root / "plugins" / "eval.py")
    invoker = IsolatedPluginInvoker(
        PluginHost(),
        {"eval": source},
        plugin_root=plugin_root,
        openagent_source=ROOT / "Src",
        capability_handler=lambda _call, _policy, request: {"ok": False},
    )
    runtime = build_v2_tool_runtime(_RuntimeApp(), host_invoker=invoker)
    call = runtime.registry.create_call(
        call_id="plugin-call",
        requested_name="eval.python",
        arguments={"code": "1 + 1"},
        context=ToolContext("isolated"),
    )

    result, _trace = asyncio.run(
        runtime.executor.execute(
            call,
            ToolPolicyRequest(
                enabled_tool_ids=frozenset({call.canonical_id}),
                granted_capabilities=call.spec.capabilities,
                confirmation=ConfirmationState.APPROVED,
                confirmation_grant=ToolConfirmationGrant.for_call("test", call),
            ),
        )
    )

    assert result.status is ToolResultStatus.SUCCESS
    assert result.output["ok"] is True


def test_register_source_requires_matching_module_under_plugin_root(
    tmp_path: Path,
) -> None:
    plugin_root = tmp_path / "root"
    plugins = plugin_root / "plugins"
    plugins.mkdir(parents=True)
    (plugins / "__init__.py").write_text("")
    source_path = plugins / "sample.py"
    source_path.write_text(
        "from OpenAgentLib.PluginSDK import PluginManifest\n"
        "MANIFEST = PluginManifest()\n"
    )
    source = inspect_v2_plugin_source(source_path)
    invoker = IsolatedPluginInvoker(
        PluginHost(),
        {},
        plugin_root=plugin_root,
        openagent_source=ROOT / "Src",
        capability_handler=lambda *_args: {"ok": False},
    )

    invoker.register_source("sample", source)
    assert invoker._sources["sample"] == source
    assert invoker.unregister_source("sample") == source
    with pytest.raises(ValueError, match="must match"):
        invoker.register_source("other", source)

    outside = tmp_path / "outside.py"
    outside.write_text(source_path.read_text())
    with pytest.raises(ValueError, match="outside"):
        invoker.register_source("outside", inspect_v2_plugin_source(outside))

    direct = plugin_root / "direct.py"
    direct.write_text(source_path.read_text())
    with pytest.raises(ValueError, match="importable package"):
        invoker.register_source("direct", inspect_v2_plugin_source(direct))

    unsafe = plugins / "bad-name.py"
    unsafe.write_text(source_path.read_text())
    with pytest.raises(ValueError, match="unsafe"):
        invoker.register_source("bad-name", inspect_v2_plugin_source(unsafe))

    helper = plugins / "_helper.py"
    helper.write_text(source_path.read_text())
    with pytest.raises(ValueError, match="helper"):
        invoker.register_source("_helper", inspect_v2_plugin_source(helper))


def test_record_admission_allows_only_same_digest_relocated_source(
    tmp_path: Path,
) -> None:
    source_text = (
        "from OpenAgentLib.PluginSDK import PluginManifest\n"
        "MANIFEST = PluginManifest()\n"
    )
    cache_root = tmp_path / "cubkit-cache"
    cache_package = cache_root / "openagent_plugins"
    cache_package.mkdir(parents=True)
    (cache_package / "__init__.py").write_text("")
    cached_path = cache_package / "terminal.py"
    cached_path.write_text(source_text)
    cached_source = inspect_v2_plugin_source(cached_path)
    record = InstalledPluginRecord(
        plugin_id="openagent.terminal",
        source=InstalledPluginSource(cached_source.path, cached_source.digest),
        manifest=InstalledPluginManifest(
            manifest_version="2",
            api_version="2",
            version="1",
            entrypoint="openagent_plugins.terminal.HANDLERS",
            capabilities=frozenset({"process"}),
            tools=(
                InstalledPluginTool(
                    canonical_id="terminal.run", capabilities=frozenset({"process"})
                ),
            ),
        ),
    )
    installed_root = tmp_path / "installed"
    installed_package = installed_root / "openagent_plugins"
    installed_package.mkdir(parents=True)
    (installed_package / "__init__.py").write_text("")
    relocated_path = installed_package / "terminal.py"
    relocated_path.write_text(source_text)
    invoker = IsolatedPluginInvoker(
        PluginHost(),
        {},
        plugin_root=installed_root,
        openagent_source=ROOT / "Src",
        capability_handler=lambda *_args: {"ok": False},
    )

    assert invoker.validate_source(
        "terminal", inspect_v2_plugin_source(relocated_path), record=record
    ) == ("terminal", "openagent_plugins.terminal")

    relocated_path.write_text(source_text + "# changed\n")
    with pytest.raises(ValueError, match="does not admit"):
        invoker.validate_source(
            "terminal", inspect_v2_plugin_source(relocated_path), record=record
        )

    untrusted_package = tmp_path / "untrusted" / "openagent_plugins"
    untrusted_package.mkdir(parents=True)
    (untrusted_package / "__init__.py").write_text("")
    untrusted_path = untrusted_package / "terminal.py"
    untrusted_path.write_text(source_text)
    with pytest.raises(ValueError, match="outside"):
        invoker.validate_source(
            "terminal", inspect_v2_plugin_source(untrusted_path), record=record
        )


def test_invoker_derives_module_path_from_admitted_source(tmp_path: Path) -> None:
    plugin_root = tmp_path / "root"
    live_package = plugin_root / "openagent_plugins"
    live_package.mkdir(parents=True)
    (live_package / "__init__.py").write_text("")
    source_path = live_package / "chat.py"
    source_path.write_text(
        "from OpenAgentLib.PluginSDK import PluginManifest\n"
        "MANIFEST = PluginManifest()\n"
    )
    host = _RecordingHost()
    invoker = IsolatedPluginInvoker(
        host,
        {"chat": inspect_v2_plugin_source(source_path)},
        plugin_root=plugin_root,
        openagent_source=ROOT / "Src",
        capability_handler=lambda *_args: {"ok": False},
    )

    asyncio.run(invoker.invoke(_call_for("chat"), SimpleNamespace(), retryable=False))

    assert host.requests[0].payload["module"] == "openagent_plugins.chat"
    assert host.requests[0].payload["entrypoint"] == "openagent_plugins.chat.HANDLERS"


def test_invoker_worker_relay_preserves_terminal_arguments_and_revalidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin_root = tmp_path / "pluginroot"
    package = plugin_root / "openagent_plugins"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    source_path = package / "terminal.py"
    source_path.write_text(
        "from OpenAgentLib.PluginSDK import PluginManifest, PluginToolDeclaration\n"
        "_SCHEMA = {\n"
        "    'type': 'object',\n"
        "    'properties': {\n"
        "        'argv': {'type': 'array', 'items': {'type': 'string'}},\n"
        "        'cwd': {'type': 'string'},\n"
        "    },\n"
        "    'required': ['argv', 'cwd'],\n"
        "    'additionalProperties': False,\n"
        "}\n"
        "MANIFEST = PluginManifest(\n"
        "    plugin_id='openagent.terminal',\n"
        "    version='2.0.0',\n"
        "    api_version='2',\n"
        "    entrypoint='openagent_plugins.terminal.HANDLERS',\n"
        "    tools=(PluginToolDeclaration(\n"
        "        canonical_id='terminal.run',\n"
        "        input_schema=_SCHEMA,\n"
        "        output_schema={'type': 'object', 'additionalProperties': True},\n"
        "        capabilities=frozenset({'process'}),\n"
        "    ),),\n"
        "    capabilities=frozenset({'process'}),\n"
        ")\n"
        "PLUGIN_MANIFEST = MANIFEST\n"
        "HANDLED = []\n"
        "def _run(call, _capability):\n"
        "    HANDLED.append(dict(call.arguments))\n"
        "    return {'argv': list(call.arguments['argv']), 'cwd': call.arguments['cwd']}\n"
        "HANDLERS = {'terminal.run': _run}\n"
        "TOOL_HANDLERS = HANDLERS\n"
    )
    source = inspect_v2_plugin_source(source_path)
    real_path = Path
    monkeypatch.setattr(
        PluginHostWorker,
        "Path",
        lambda value: plugin_root if value == "/mnt/pluginroot" else real_path(value),
    )
    monkeypatch.syspath_prepend(str(plugin_root))
    monkeypatch.delitem(sys.modules, "openagent_plugins.terminal", raising=False)
    monkeypatch.delitem(sys.modules, "openagent_plugins", raising=False)

    class _WorkerRelayHost:
        def __init__(self) -> None:
            self.requests: list[PluginHostRequest] = []
            self.results: list[object] = []

        async def call(self, request: PluginHostRequest, **_kwargs: object) -> object:
            self.requests.append(request)
            frame = json.loads(request.to_json_line(max_frame_bytes=16_384))
            self.results.append(PluginHostWorker._plugin_call(frame, 16_384))
            return SimpleNamespace()

    capability_contexts = []

    class _RecordingCapabilityClient:
        def __init__(self, context, _transport) -> None:
            capability_contexts.append(context)

    monkeypatch.setattr(PluginSDK, "CapabilityClient", _RecordingCapabilityClient)
    host = _WorkerRelayHost()
    invoker = IsolatedPluginInvoker(
        host,
        {"terminal": source},
        plugin_root=plugin_root,
        openagent_source=ROOT / "Src",
        capability_handler=lambda *_args: {"ok": False},
    )
    valid_arguments = {"argv": ["git", "status"], "cwd": "."}
    call = SimpleNamespace(
        spec=SimpleNamespace(source_module="terminal"),
        call_id="terminal-worker",
        canonical_id="terminal.run",
        arguments=valid_arguments,
        context=ToolContext("terminal-worker", actor_id="test-user"),
    )

    asyncio.run(invoker.invoke(call, SimpleNamespace(), retryable=False))

    assert host.requests[0].payload["arguments"] == {
        "argv": ("git", "status"),
        "cwd": ".",
    }
    assert host.results == [{"argv": ["git", "status"], "cwd": "."}]
    assert capability_contexts[0].actor_scope == "actor:test-user"
    module = importlib.import_module("openagent_plugins.terminal")
    assert module.HANDLED == [{"argv": ("git", "status"), "cwd": "."}]

    for invalid_arguments in (
        {"command": "git status"},
        {"cmd": "git status"},
        {"argv": ["git", "status"]},
    ):
        call.arguments = invalid_arguments
        with pytest.raises(ToolArgumentError):
            asyncio.run(invoker.invoke(call, SimpleNamespace(), retryable=False))

    assert module.HANDLED == [{"argv": ("git", "status"), "cwd": "."}]


def test_invocation_owns_and_releases_exact_registry_task(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "root"
        package = root / "openagent_plugins"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        source_path = package / "chat.py"
        source_path.write_text(
            "from OpenAgentLib.PluginSDK import PluginManifest\n"
            "MANIFEST = PluginManifest()\n"
        )
        source = inspect_v2_plugin_source(source_path)
        registry = InstalledPluginRegistry()
        installed = registry.install_active(_record_for(source))
        host = _BlockingHost()
        invoker = IsolatedPluginInvoker(
            host,
            {"chat": source},
            plugin_root=root,
            openagent_source=ROOT / "Src",
            capability_handler=lambda *_args: {"ok": False},
            registry=registry,
        )
        invoker.register_source("chat", source, record=installed)
        task = asyncio.create_task(
            invoker.invoke(_call_for("chat"), SimpleNamespace(), retryable=False)
        )
        await host.started.wait()
        assert registry.get(installed.plugin_id).task_ids == {"isolated-payload"}
        await invoker.quiesce(installed.plugin_id, installed.generation)
        assert task.cancelled()
        assert registry.get(installed.plugin_id).task_ids == frozenset()

    asyncio.run(scenario())


def test_invoker_rejects_non_active_and_stale_records(tmp_path: Path) -> None:
    root = tmp_path / "root"
    package = root / "openagent_plugins"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    source_path = package / "chat.py"
    source_path.write_text(
        "from OpenAgentLib.PluginSDK import PluginManifest\n"
        "MANIFEST = PluginManifest()\n"
    )
    source = inspect_v2_plugin_source(source_path)
    registry = InstalledPluginRegistry()
    initial = registry.install_active(_record_for(source))
    unloading = registry.begin_unload(
        initial.plugin_id, expected_generation=initial.generation
    )
    disabled = registry.complete_unload(
        unloading.plugin_id, expected_generation=unloading.generation
    )
    replacement = registry.replace(
        _record_for(source), expected_generation=disabled.generation
    )
    host = _RecordingHost()
    invoker = IsolatedPluginInvoker(
        host,
        {},
        plugin_root=root,
        openagent_source=ROOT / "Src",
        capability_handler=lambda *_args: {"ok": False},
        registry=registry,
    )
    invoker.register_source("chat", source, record=replacement)

    with pytest.raises(RuntimeError, match="plugin is not active") as non_active:
        asyncio.run(
            invoker.invoke(_call_for("chat"), SimpleNamespace(), retryable=False)
        )
    assert non_active.value.tool_failure_stage == "admission"

    active = registry.activate(
        replacement.plugin_id, expected_generation=replacement.generation
    )
    with pytest.raises(RuntimeError, match="plugin is not active") as stale:
        asyncio.run(
            invoker.invoke(_call_for("chat"), SimpleNamespace(), retryable=False)
        )
    assert stale.value.tool_failure_stage == "admission"

    invoker.register_source("chat", source, record=active)
    asyncio.run(invoker.invoke(_call_for("chat"), SimpleNamespace(), retryable=False))
    assert len(host.requests) == 1


def test_invoker_keeps_canonical_plugins_module_path() -> None:
    plugin_root = ROOT.parent / "repo-MCUB-fork" / "OpenAgent"
    host = _RecordingHost()
    invoker = IsolatedPluginInvoker(
        host,
        {"eval": inspect_v2_plugin_source(plugin_root / "plugins" / "eval.py")},
        plugin_root=plugin_root,
        openagent_source=ROOT / "Src",
        capability_handler=lambda *_args: {"ok": False},
    )

    asyncio.run(invoker.invoke(_call_for("eval"), SimpleNamespace(), retryable=False))

    assert host.requests[0].payload["module"] == "plugins.eval"
