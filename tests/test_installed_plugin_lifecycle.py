from __future__ import annotations

import asyncio
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from OpenAgentLib.InstalledPluginRegistry import (
    InstalledPluginRegistry,
    InstalledPluginStatus,
)
from OpenAgentLib.IsolatedPluginInvoker import IsolatedPluginInvoker
from OpenAgentLib.Plugin.PluginsEngine import _OpenAgentPluginSkillMixin
from OpenAgentLib.PluginDiscovery import (
    inspect_installed_v2_plugin_source,
    rebuild_installed_plugin_record,
)
from OpenAgentLib.V2Bootstrap import build_v2_tool_runtime

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "repo-MCUB-fork" / "OpenAgent" / "plugins"


class _Log:
    def info(self, *_args: object) -> None:
        pass

    def warning(self, *_args: object) -> None:
        pass


class _RecordingHost:
    def __init__(self) -> None:
        self.requests: list[object] = []

    async def call(self, request: object, **_kwargs: object) -> object:
        self.requests.append(request)
        return SimpleNamespace()


class _Invoker:
    def __init__(self, fail_modules: set[str] = set()) -> None:
        self.fail_modules = fail_modules
        self.sources: dict[str, object] = {}

    def register_source(self, module: str, source: object, **_kwargs: object) -> None:
        if module in self.fail_modules:
            raise RuntimeError(f"cannot bind {module}")
        self.sources[module] = source

    def unregister_source(self, module: str) -> object | None:
        return self.sources.pop(module, None)

    def snapshot_source(self, module: str) -> object | None:
        source = self.sources.get(module)
        return (source, None) if source is not None else None

    def restore_source(self, module: str, snapshot: object) -> None:
        source, _record = snapshot
        self.sources[module] = source


class _ActivationFailRegistry(InstalledPluginRegistry):
    def activate(self, plugin_id: object, *, expected_generation: int):
        raise RuntimeError(f"cannot activate {plugin_id}")


class _Harness(_OpenAgentPluginSkillMixin):
    def __init__(
        self,
        tmp_path: Path,
        *,
        fail_modules: set[str] = set(),
        registry: InstalledPluginRegistry | None = None,
    ) -> None:
        self.kernel = SimpleNamespace(WORK_DIR=str(tmp_path))
        self.log = _Log()
        self.startup_disabled_ids: set[str] = set()
        self._installed_plugin_registry = registry or InstalledPluginRegistry()
        self._v2_plugin_invoker = _Invoker(fail_modules)


class _IsolatedHarness(_Harness):
    def __init__(self, tmp_path: Path, registry: InstalledPluginRegistry) -> None:
        super().__init__(tmp_path, registry=registry)
        self.host = _RecordingHost()
        self._v2_plugin_invoker = IsolatedPluginInvoker(
            self.host,
            {},
            plugin_root=tmp_path,
            openagent_source=ROOT / "OpenAgent-old" / "Src",
            capability_handler=lambda *_args: {"ok": False},
            registry=registry,
        )


def _install_canonical_plugins(tmp_path: Path) -> Path:
    installed = tmp_path / "openagent_plugins"
    installed.mkdir()
    for source in CANONICAL.glob("*.py"):
        shutil.copy(source, installed / source.name)
    return installed


def _load(harness: _Harness) -> None:
    asyncio.run(harness._load_installed_plugins(harness.startup_disabled_ids))


def _runtime_tool_ids(harness: _Harness) -> set[str]:
    runtime = build_v2_tool_runtime(
        SimpleNamespace(),
        installed_records=harness._installed_plugin_registry.snapshot(
            status=InstalledPluginStatus.ACTIVE
        ),
        include_sibling_fallback=False,
    )
    return {spec.canonical_id for spec in runtime.registry.specs()}


async def _assert_active_bindings_invoke(
    harness: _IsolatedHarness, call_prefix: str
) -> None:
    records = harness._installed_plugin_registry.snapshot(
        status=InstalledPluginStatus.ACTIVE
    )
    invoker = harness._v2_plugin_invoker
    modules = {
        str(record.manifest.metadata["source_module"]): record for record in records
    }
    assert set(invoker._records) == set(modules)
    for module, current in modules.items():
        bound = invoker._records[module]
        assert bound is current
        assert bound.generation == current.generation
        assert bound.status is InstalledPluginStatus.ACTIVE
        assert bound.enabled
        await invoker.invoke(
            SimpleNamespace(
                spec=SimpleNamespace(source_module=module),
                call_id=f"{call_prefix}-{module}",
                canonical_id=current.manifest.tools[0].canonical_id,
                arguments={},
                context=None,
            ),
            SimpleNamespace(),
            retryable=False,
        )


def test_startup_binds_enabled_sources_to_current_active_records_across_restart(
    tmp_path: Path,
) -> None:
    _install_canonical_plugins(tmp_path)
    registry = InstalledPluginRegistry()
    first = _IsolatedHarness(tmp_path, registry)
    _load(first)

    runtime_tool_ids = _runtime_tool_ids(first)
    assert {"terminal.run", "terminal.inspect"} <= runtime_tool_ids
    asyncio.run(_assert_active_bindings_invoke(first, "initial"))

    restarted = _IsolatedHarness(tmp_path, registry)
    assert restarted._v2_plugin_invoker._records == {}
    _load(restarted)
    asyncio.run(_assert_active_bindings_invoke(restarted, "restart"))


def test_restart_rebuilds_all_installed_sources_without_importing_them(
    tmp_path: Path,
) -> None:
    installed = _install_canonical_plugins(tmp_path)
    chat = installed / "chat.py"
    chat.write_text("raise RuntimeError('must not execute')\n" + chat.read_text())
    first = _Harness(tmp_path)
    _load(first)

    records = first._installed_plugin_registry.snapshot()
    assert len(records) == 14
    assert all(record.status is InstalledPluginStatus.ACTIVE for record in records)
    assert set(first._v2_plugin_invoker.sources) == {
        path.stem
        for path in CANONICAL.glob("*.py")
        if not path.name.startswith("_") and path.name != "__init__.py"
    }
    assert first._installed_plugin_diagnostics == {}

    second = _Harness(tmp_path)
    _load(second)
    assert [
        (
            record.plugin_id,
            record.source,
            record.manifest,
            record.enabled,
            record.generation,
            record.status,
        )
        for record in second._installed_plugin_registry.snapshot()
    ] == [
        (
            record.plugin_id,
            record.source,
            record.manifest,
            record.enabled,
            record.generation,
            record.status,
        )
        for record in records
    ]


def test_reconstruction_normalizes_disabled_stems_and_keeps_records_visible(
    tmp_path: Path,
) -> None:
    _install_canonical_plugins(tmp_path)
    harness = _Harness(tmp_path)
    harness.startup_disabled_ids = {"chat", "openagent.file"}
    _load(harness)

    chat = harness._installed_plugin_registry.get("openagent.chat")
    file = harness._installed_plugin_registry.get("openagent.file")
    assert chat.status is InstalledPluginStatus.DISABLED
    assert file.status is InstalledPluginStatus.DISABLED
    assert not chat.enabled and not file.enabled
    assert "chat" not in harness._v2_plugin_invoker.sources
    assert (
        '"openagent.chat"'
        in (tmp_path / "openagent_plugins" / "disabled_plugins.json").read_text()
    )


def test_helpers_and_malformed_files_are_not_executable_and_are_diagnosed(
    tmp_path: Path,
) -> None:
    installed = _install_canonical_plugins(tmp_path)
    (installed / "_helper.py").write_text("raise RuntimeError('ignored')\n")
    (installed / "bad.py").write_text("this is not valid python (\n")
    harness = _Harness(tmp_path)
    _load(harness)

    assert "_helper" not in harness._v2_plugin_invoker.sources
    assert "bad" not in harness._v2_plugin_invoker.sources
    diagnostics = harness._installed_plugin_diagnostics
    assert str((installed / "bad.py").resolve()) in diagnostics
    with pytest.raises(TypeError):
        diagnostics["other"] = "mutate"  # type: ignore[index]


def test_invoker_binding_failure_is_visible_but_never_active(tmp_path: Path) -> None:
    installed = tmp_path / "openagent_plugins"
    installed.mkdir()
    shutil.copy(CANONICAL / "chat.py", installed / "chat.py")
    harness = _Harness(tmp_path, fail_modules={"chat"})
    _load(harness)

    record = harness._installed_plugin_registry.get("openagent.chat")
    assert record.status is InstalledPluginStatus.FAILED
    assert "chat" not in harness._v2_plugin_invoker.sources
    assert (
        str((installed / "chat.py").resolve()) in harness._installed_plugin_diagnostics
    )


def test_activation_failure_cleans_registered_source_and_runtime_tools(
    tmp_path: Path,
) -> None:
    installed = tmp_path / "openagent_plugins"
    installed.mkdir()
    shutil.copy(CANONICAL / "chat.py", installed / "chat.py")
    harness = _Harness(tmp_path, registry=_ActivationFailRegistry())
    _load(harness)

    record = harness._installed_plugin_registry.get("openagent.chat")
    assert record.status is InstalledPluginStatus.FAILED
    assert record.enabled
    assert "chat" not in harness._v2_plugin_invoker.sources
    assert "chat" not in harness._v2_plugin_invoker.sources
    assert "chat.get_history" not in _runtime_tool_ids(harness)


def test_disabled_record_is_catalog_visible_but_absent_from_runtime(
    tmp_path: Path,
) -> None:
    installed = tmp_path / "openagent_plugins"
    installed.mkdir()
    shutil.copy(CANONICAL / "chat.py", installed / "chat.py")
    harness = _Harness(tmp_path)
    harness.startup_disabled_ids = {"chat"}
    _load(harness)

    record = harness._installed_plugin_registry.get("openagent.chat")
    assert record.status is InstalledPluginStatus.DISABLED
    assert record in harness._installed_plugin_registry.catalog_snapshot()
    assert "chat.get_history" not in _runtime_tool_ids(harness)


def test_repeated_loader_call_is_idempotent_for_unchanged_active_source(
    tmp_path: Path,
) -> None:
    installed = tmp_path / "openagent_plugins"
    installed.mkdir()
    shutil.copy(CANONICAL / "chat.py", installed / "chat.py")
    harness = _Harness(tmp_path)
    _load(harness)
    first = harness._installed_plugin_registry.get("openagent.chat")
    _load(harness)

    second = harness._installed_plugin_registry.get("openagent.chat")
    assert second == first
    assert second.status is InstalledPluginStatus.ACTIVE
    assert set(harness._v2_plugin_invoker.sources) == {"chat"}


def test_duplicate_declarations_fail_closed_without_load_order_winner(
    tmp_path: Path,
) -> None:
    installed = tmp_path / "openagent_plugins"
    installed.mkdir()
    shutil.copy(CANONICAL / "chat.py", installed / "chat.py")
    shutil.copy(CANONICAL / "chat.py", installed / "chat_copy.py")
    harness = _Harness(tmp_path)
    _load(harness)

    assert harness._installed_plugin_registry.snapshot() == ()
    assert set(harness._installed_plugin_diagnostics) == {
        str((installed / "chat.py").resolve()),
        str((installed / "chat_copy.py").resolve()),
    }


def test_changed_source_digest_replaces_static_registry_generation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "chat.py"
    shutil.copy(CANONICAL / "chat.py", source)
    registry = InstalledPluginRegistry()
    first = rebuild_installed_plugin_record(
        registry, inspect_installed_v2_plugin_source(source)
    )
    active = registry.activate(first.plugin_id, expected_generation=first.generation)
    source.write_text(source.read_text() + "\n# changed digest\n")

    replacement = rebuild_installed_plugin_record(
        registry, inspect_installed_v2_plugin_source(source)
    )
    assert replacement.generation > active.generation
    assert replacement.source.digest != active.source.digest
    assert replacement.status is InstalledPluginStatus.INSTALLED
