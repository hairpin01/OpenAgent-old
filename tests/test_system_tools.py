from __future__ import annotations

from pathlib import Path

from conftest import ROOT, load_source_module

base = load_source_module(
    "openagent_system_tools_base_test",
    "Src/OpenAgentLib/SystemPlugins/base.py",
)


def test_system_tool_parallel_safe_defaults_to_false() -> None:
    tool = base.SystemTool(
        tool_class="test",
        name="read",
        docs={"desc": "Read test data"},
    )
    assert not tool.parallel_safe


def test_registry_loads_explicit_parallel_metadata() -> None:
    root = ROOT / "Src/OpenAgentLib/SystemPlugins"
    registry = base.SystemToolRegistry(Path(root))
    tools = registry.load()
    registry.validate()

    assert tools["utility.list_tools"].parallel_safe
    assert tools["utility.search_tool"].parallel_safe
    assert not tools["todo.current"].parallel_safe
    assert not tools["code.choose_filename"].parallel_safe
    assert not tools["todo.clear"].parallel_safe
