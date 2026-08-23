from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from OpenAgentLib.InstalledPluginRegistry import (
    InstalledPluginRegistry,
    InstalledPluginStatus,
)
from OpenAgentLib.Plugin.PluginsEngine import _OpenAgentPluginSkillMixin

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "repo-MCUB-fork" / "OpenAgent" / "plugins"


class FakeCommandEvent:
    """Minimal command event retained to exercise command-facing state ownership."""

    def __init__(self, actor_id: int = 1) -> None:
        self.sender_id = actor_id


class _Log:
    def info(self, *_args: object) -> None:
        pass

    def warning(self, *_args: object) -> None:
        pass


class _Invoker:
    def __init__(self) -> None:
        self.sources: dict[str, object] = {}
        self.records: dict[str, object] = {}

    def validate_source(
        self, module: str, _source: object, **_kwargs: object
    ) -> tuple[str, str]:
        return module, f"openagent_plugins.{module}"

    def register_validated_source(
        self, validated: tuple[str, str], source: object, **kwargs: object
    ) -> None:
        self.sources[validated[0]] = source
        self.records[validated[0]] = kwargs.get("record")

    def register_source(self, module: str, source: object, **kwargs: object) -> None:
        self.register_validated_source((module, module), source, **kwargs)

    def unregister_source(self, module: str) -> object | None:
        self.records.pop(module, None)
        return self.sources.pop(module, None)

    def snapshot_source(self, module: str) -> tuple[object, object] | None:
        source = self.sources.get(module)
        return (source, self.records.get(module)) if source is not None else None

    def restore_source(self, module: str, snapshot: tuple[object, object]) -> None:
        self.sources[module], self.records[module] = snapshot

    async def quiesce(self, _plugin_id: str, _generation: int) -> None:
        return None


class _Harness(_OpenAgentPluginSkillMixin):
    def __init__(self, tmp_path: Path) -> None:
        self.kernel = SimpleNamespace(WORK_DIR=str(tmp_path))
        self.log = _Log()
        self._installed_plugin_registry = InstalledPluginRegistry()
        self._v2_plugin_invoker = _Invoker()
        self._plugins_cache: list[dict[str, object]] | None = None
        self._tool_map_cache = None
        self._tool_registry_cache = None


def _bundle() -> dict[str, bytes]:
    return {
        name: (CANONICAL / name).read_bytes()
        for name in ("chat.py", "_telegram_v2.py", "__init__.py")
    }


def test_installed_plugin_e2e_registry_generation_lifecycle(tmp_path: Path) -> None:
    event = FakeCommandEvent()
    first = _Harness(tmp_path)
    installed = asyncio.run(first._install_v2_plugin_files(_bundle(), "chat.py"))
    assert event.sender_id == 1
    assert installed.status is InstalledPluginStatus.ACTIVE
    assert (
        first._installed_plugin_registry.catalog_snapshot()[0].plugin_id
        == "openagent.chat"
    )
    assert first._v2_plugin_invoker.records["chat"] is installed

    disabled = asyncio.run(
        first._set_installed_plugin_enabled(
            installed.plugin_id, expected_generation=installed.generation, enabled=False
        )
    )
    assert disabled.status is InstalledPluginStatus.DISABLED
    assert "chat" not in first._v2_plugin_invoker.sources

    restarted = _Harness(tmp_path)
    asyncio.run(restarted._load_installed_plugins(restarted._load_disabled_plugins()))
    restored = restarted._installed_plugin_registry.get("openagent.chat")
    assert restored.status is InstalledPluginStatus.DISABLED
    assert restored.source.digest == disabled.source.digest

    active = asyncio.run(
        restarted._set_installed_plugin_enabled(
            restored.plugin_id, expected_generation=restored.generation, enabled=True
        )
    )
    assert active.status is InstalledPluginStatus.ACTIVE
    assert restarted._v2_plugin_invoker.records["chat"] is active

    restarted._unregister_plugin(active.plugin_id)
    assert restarted._installed_plugin_registry.find(active.plugin_id) is None
    assert "chat" not in restarted._v2_plugin_invoker.sources
