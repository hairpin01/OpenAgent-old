from __future__ import annotations

from pathlib import Path

import pytest

from conftest import ROOT
from OpenAgentLib.PluginDiscovery import inspect_v2_plugin_source
from OpenAgentLib.PluginSDK import LegacyPluginMigrationError


def test_legacy_plugin_is_rejected_without_module_execution(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    plugin = tmp_path / "legacy.py"
    plugin.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('side effect')\n"
        "class LegacyPlugin:\n"
        "    tool_registry = ('legacy.tool',)\n",
        encoding="utf-8",
    )

    with pytest.raises(LegacyPluginMigrationError, match="removed legacy execution"):
        inspect_v2_plugin_source(plugin)

    assert not marker.exists()


def test_static_v2_marker_is_admitted_without_module_execution(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    plugin = tmp_path / "v2.py"
    plugin.write_text(
        "from OpenAgentLib.PluginSDK import PluginManifest\n"
        f"Path({str(marker)!r}).write_text('side effect')\n"
        "MANIFEST = PluginManifest('demo.plugin', '1.0.0', '2', 'demo.main', (), frozenset())\n",
        encoding="utf-8",
    )

    source = inspect_v2_plugin_source(plugin)

    assert source.path == plugin.resolve()
    assert len(source.digest) == 64
    assert not marker.exists()


def test_sibling_v2_plugins_are_inspected_without_importing_them() -> None:
    plugins = ROOT.parent / "repo-MCUB-fork" / "OpenAgent" / "plugins"
    expected = {
        "ast_grep",
        "chat",
        "contacts",
        "creation",
        "dialog",
        "eval",
        "file",
        "mcub",
        "message",
        "moderation",
        "profile",
        "task",
        "terminal",
        "web",
    }

    discovered = {
        path.stem: inspect_v2_plugin_source(path)
        for path in sorted(plugins.glob("*.py"))
        if not path.name.startswith("_") and path.name != "__init__.py"
    }

    assert set(discovered) == expected
    assert all(len(source.digest) == 64 for source in discovered.values())


def test_normal_runtime_has_no_legacy_dispatch_or_plugin_exec() -> None:
    runtime_sources = (
        ROOT / "Src" / "OpenAgentLib" / "Plugin" / "PluginsEngine.py",
        ROOT / "Src" / "OpenAgentLib" / "ResponseAgent.py",
        ROOT / "Src" / "OpenAgentLib" / "Lifecycle.py",
        ROOT / "Src" / "OpenAgentLib" / "ToolDispatch.py",
        ROOT / "Src" / "OpenAgentLib" / "SystemPlugins" / "base.py",
    )
    forbidden = (
        "_get_tool_map(",
        "_dispatch_tool(",
        "create_subprocess_shell",
        "spec_from_file_location",
        "exec_module",
    )

    for path in runtime_sources:
        source = path.read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden), path
