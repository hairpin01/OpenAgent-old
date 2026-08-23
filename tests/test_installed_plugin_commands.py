from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import ModuleType

import cubkit
import pytest

from command_testkit import FakeCommandEvent
from OpenAgentLib.InstalledPluginRegistry import (
    InstalledPluginManifest,
    InstalledPluginRecord,
    InstalledPluginRegistry,
    InstalledPluginSource,
    InstalledPluginStatus,
    InstalledPluginTool,
)
from OpenAgentLib.InstalledPluginActions import InstalledPluginActionStore
from OpenAgentLib.Plugin.PluginsEngine import _OpenAgentPluginSkillMixin

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


class _Button:
    @staticmethod
    def inline(text, _callback, *, args=(), style=None):
        return {"kind": "inline", "text": text, "args": tuple(args), "style": style}

    @staticmethod
    def url(text, url):
        return {"kind": "url", "text": text, "url": url}


class _Harness(_OpenAgentPluginSkillMixin):
    Button = _Button
    _oaplugin_noop = OpenAgent._oaplugin_noop
    _oaplugin_main = OpenAgent._oaplugin_main
    _oaplugin_catalog = OpenAgent._oaplugin_catalog
    _oaplugin_install = OpenAgent._oaplugin_install
    _oaplugin_manager = OpenAgent._oaplugin_manager
    _oaplugin_uninstall = OpenAgent._oaplugin_uninstall

    def __init__(self, registry: InstalledPluginRegistry) -> None:
        self._installed_plugin_registry = registry
        self._plugins_cache = None

    @staticmethod
    def strings(key: str, **kwargs) -> str:
        values = {
            "plugins_enabled_title": "Installed plugins:\n",
            "plugins_none_installed": "None\n",
            "plugin_id_label": "ID",
            "plugin_author_label": "Author",
            "plugin_tools_label": "Tools",
            "plugin_version_label": "Version",
            "plugin_permissions_label": "Permissions",
            "plugin_requirements_label": "Requirements",
            "plugin_actions_title": "Actions",
            "plugin_delete_btn": "Delete",
            "plugin_installed_btn": "Installed",
            "plugin_install_btn": "Install",
            "plugin_code_btn": "Code",
            "plugin_manager_no_installed": "No installed plugins",
            "back_btn": "Back",
        }
        if key == "plugins_total":
            return f"Total: {kwargs['count']}"
        if key == "plugin_more_tools":
            return f" (+{kwargs['count']})"
        if key == "plugin_deleted_alert":
            return f"Deleted {kwargs['name']}"
        if key == "plugin_installed_alert":
            return f"Installed {kwargs['name']}"
        if key == "generic_error":
            return f"Error: {kwargs['error']}"
        return values.get(key, key)


def _candidate(
    tmp_path: Path,
    plugin_id: str,
    source_name: str,
    tool_id: str,
    *,
    display_name: str = "Example",
    metadata: object | None = None,
    input_schema: object | None = None,
    enabled: bool = True,
) -> InstalledPluginRecord:
    source = tmp_path / source_name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# plugin\n", encoding="utf-8")
    return InstalledPluginRecord(
        plugin_id=plugin_id,
        source=InstalledPluginSource(source.resolve(), "a" * 64),
        manifest=InstalledPluginManifest(
            manifest_version="2",
            api_version="2",
            version="2.4.1",
            entrypoint="example_plugin.HANDLERS",
            display_name=display_name,
            metadata=(
                metadata
                if metadata is not None
                else {
                    "author": "Ada",
                    "description": "Useful plugin",
                    "requirements": ["demo>=1"],
                }
            ),
            capabilities=frozenset({"network"}),
            tools=(
                InstalledPluginTool(
                    canonical_id=tool_id,
                    aliases=(f"{tool_id}.alias",),
                    capabilities=frozenset({"network"}),
                    input_schema=(
                        input_schema if input_schema is not None else {"type": "object"}
                    ),
                    description="Runs the tool",
                ),
            ),
        ),
        enabled=enabled,
        status=(
            InstalledPluginStatus.ADMITTED
            if enabled
            else InstalledPluginStatus.DISABLED
        ),
    )


def _three_state_registry(tmp_path: Path) -> InstalledPluginRegistry:
    registry = InstalledPluginRegistry()
    registry.install_active(
        _candidate(tmp_path, "demo.active", "active.py", "active.run")
    )
    registry.install(
        _candidate(
            tmp_path,
            "demo.disabled",
            "disabled.py",
            "disabled.run",
            enabled=False,
        )
    )
    installed = registry.install(
        _candidate(tmp_path, "demo.failed", "failed.py", "failed.run")
    )
    registry.fail(
        installed.plugin_id,
        expected_generation=installed.generation,
        failure_code="startup_failed",
        failure_reason="token=super-secret connection rejected",
    )
    return registry


def test_overview_empty_and_three_states_use_registry_only(tmp_path: Path) -> None:
    empty = _Harness(InstalledPluginRegistry())
    assert "None" in OpenAgent._format_oaplugin_overview(empty)
    assert "Total: 0" in OpenAgent._format_oaplugin_overview(empty)

    harness = _Harness(_three_state_registry(tmp_path))
    assert not hasattr(harness, "_plugins")
    text = OpenAgent._format_oaplugin_overview(harness)

    assert "Total: 3" in text
    assert all(state in text for state in ("active", "disabled", "failed"))
    assert "demo.active" in text and "active.run" in text
    assert "Generation:" in text and "Enabled: <code>no</code>" in text
    assert "super-secret" not in text and "token=[redacted]" in text


def test_presentation_defaults_and_html_escape_malicious_metadata(
    tmp_path: Path,
) -> None:
    registry = InstalledPluginRegistry()
    registry.install_active(
        _candidate(
            tmp_path,
            "demo.escape",
            "escape.py",
            "escape.run",
            display_name="<b>owned</b>",
            metadata={"author": "<script>x</script>", "description": "<i>bad</i>"},
        )
    )
    harness = _Harness(registry)
    harness._installed_plugin_actions = InstalledPluginActionStore()

    text = OpenAgent._format_oaplugin_overview(harness)
    docs = harness._format_plugin_docs("demo.escape")

    assert "&lt;b&gt;owned&lt;/b&gt;" in text
    assert "&lt;script&gt;x&lt;/script&gt;" in text
    assert "<script>" not in text and "<i>bad</i>" not in text
    assert "escape.run" in docs and "state: active" in docs


def test_plugin_docs_expose_exact_installed_tool_schema(tmp_path: Path) -> None:
    schema = {
        "type": "object",
        "properties": {
            "argv": {"type": "array", "items": {"type": "string"}},
            "cwd": {"type": "string"},
        },
        "required": ["argv", "cwd"],
        "additionalProperties": False,
    }
    registry = InstalledPluginRegistry()
    registry.install_active(
        _candidate(
            tmp_path,
            "openagent.terminal",
            "terminal.py",
            "terminal.run",
            input_schema=schema,
        )
    )

    docs = _Harness(registry)._format_plugin_docs("openagent.terminal")

    assert '"required":["argv","cwd"]' in docs
    assert '"argv":{"items":{"type":"string"},"type":"array"}' in docs
    assert '"cwd":{"type":"string"}' in docs


@pytest.mark.asyncio
async def test_catalog_marker_matches_canonical_or_unique_stem_and_rejects_ambiguous(
    tmp_path: Path,
) -> None:
    registry = InstalledPluginRegistry()
    registry.install_active(_candidate(tmp_path, "demo.one", "shared.py", "one.run"))
    registry.install_active(
        _candidate(tmp_path / "other", "demo.two", "shared.py", "two.run")
    )
    harness = _Harness(registry)

    assert harness._find_installed_plugin_presentation(plugin_id="demo.one") is not None
    assert harness._find_installed_plugin_presentation(source_stem="shared") is None

    harness._plugins_cache = [
        {
            "plugin_id": "demo.one",
            "plugin_name": "remote-name",
            "file_name": "remote-name.py",
            "name": "Remote",
            "version": "1",
            "author": "Repo",
            "description": "Remote catalog entry",
            "tools": [],
            "permissions": [],
            "requirements": [],
        }
    ]
    call = FakeCommandEvent()
    await OpenAgent._oaplugin_catalog(harness, call, 0)
    buttons = call.calls[-1].kwargs["buttons"]
    assert buttons[0][0]["text"] == "Installed"

    harness._plugins_cache[0]["plugin_id"] = "remote.uninstalled"
    harness._plugins_cache[0]["plugin_name"] = "shared"
    other_call = FakeCommandEvent()
    await OpenAgent._oaplugin_catalog(harness, other_call, 0)
    other_buttons = other_call.calls[-1].kwargs["buttons"]
    assert other_buttons[0][0]["text"] == "Install"


@pytest.mark.asyncio
async def test_manager_renders_record_and_clamps_out_of_range_page(
    tmp_path: Path,
) -> None:
    harness = _Harness(_three_state_registry(tmp_path))
    call = FakeCommandEvent()

    await OpenAgent._oaplugin_manager(harness, call, 1)
    text = call.calls[-1].args[0]
    assert "demo.disabled" in text
    assert "State: <code>disabled</code>" in text
    assert "Enabled: <code>no</code>" in text
    assert call.calls[-1].kwargs["buttons"][0][0]["args"][0] != "demo.disabled"

    stale = FakeCommandEvent()
    await OpenAgent._oaplugin_manager(harness, stale, 99)
    assert [item.operation for item in stale.calls] == ["edit"]
    assert "demo.failed" in stale.calls[0].args[0]


@pytest.mark.asyncio
async def test_delete_uses_canonical_registry_id_and_clamps_page(
    tmp_path: Path,
) -> None:
    registry = InstalledPluginRegistry()
    registry.install_active(
        _candidate(tmp_path, "demo.delete", "delete.py", "delete.run")
    )
    registry.install_active(_candidate(tmp_path, "demo.keep", "keep.py", "keep.run"))
    harness = _Harness(registry)
    harness._installed_plugin_actions = InstalledPluginActionStore()
    call = FakeCommandEvent()

    action = harness._installed_plugin_actions.issue(
        registry,
        registry.get("demo.delete"),
        actor_id=200,
        kind="delete",
        payload={"page": 9},
    )
    await OpenAgent._oaplugin_uninstall(harness, call, action.token)

    assert registry.find("demo.delete") is None
    assert any(item.operation == "edit" for item in call.calls)
    assert "demo.keep" in next(
        item.args[0] for item in call.calls if item.operation == "edit"
    )


@pytest.mark.asyncio
async def test_delete_error_reports_without_navigation(tmp_path: Path) -> None:
    class _FailingHarness(_Harness):
        def _unregister_plugin(self, _name: str) -> None:
            raise RuntimeError("cannot remove")

    registry = InstalledPluginRegistry()
    registry.install_active(
        _candidate(tmp_path, "demo.faildelete", "faildelete.py", "faildelete.run")
    )
    harness = _FailingHarness(registry)
    harness._installed_plugin_actions = InstalledPluginActionStore()
    call = FakeCommandEvent()

    action = harness._installed_plugin_actions.issue(
        registry,
        registry.get("demo.faildelete"),
        actor_id=200,
        kind="delete",
        payload={"page": 0},
    )
    await OpenAgent._oaplugin_uninstall(harness, call, action.token)

    assert [item.operation for item in call.calls] == ["answer"]
    assert "cannot remove" in call.calls[0].args[0]


def test_registry_count_excludes_helper_and_remote_catalog_entries(
    tmp_path: Path,
) -> None:
    registry = InstalledPluginRegistry()
    registry.install_active(_candidate(tmp_path, "demo.real", "real.py", "real.run"))
    (tmp_path / "_helper.py").write_text("# helper\n", encoding="utf-8")
    harness = _Harness(registry)
    harness._plugins_cache = [{"plugin_id": "remote.only", "plugin_name": "remote"}]

    assert len(harness._registry_catalog_snapshot()) == 1
    assert "Total: 1" in OpenAgent._format_oaplugin_overview(harness)
    assert harness._find_installed_plugin_presentation(plugin_id="remote.only") is None
