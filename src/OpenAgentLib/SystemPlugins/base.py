# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable
import importlib.util
import inspect
import sys
import uuid


SystemToolHandler = str | Callable[..., Any]


@dataclass(frozen=True)
class SystemTool:
    """Descriptor for one built-in OpenAgent tool.

    A system tool is declared in exactly one Python file under
    ``OpenAgentLib/SystemPlugins/<group>/<tool>.py``. The file exports either a
    ``SYSTEM_TOOL`` instance or a ``create_tool()`` factory returning one.
    """

    group: str
    name: str
    handler: SystemToolHandler
    docs: dict[str, str] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    dangerous: bool = False

    @property
    def tool_class(self) -> str:
        return self.group.strip().lower()

    @property
    def full_name(self) -> str:
        group = self.tool_class
        name = self.name.strip().lower()
        if not group or not name:
            raise ValueError("SystemTool group and name are required")
        return f"{group}.{name}"

    @property
    def all_names(self) -> tuple[str, ...]:
        names = [self.full_name]
        for alias in self.aliases:
            clean = str(alias or "").strip().lower()
            if clean and clean not in names:
                names.append(clean)
        return tuple(names)


def discover_system_tools(root: Path | None = None) -> dict[str, SystemTool]:
    """Load all SystemPlugins recursively without hardcoded plugin names."""

    root = root or Path(__file__).resolve().parent
    tools: dict[str, SystemTool] = {}
    if not root.exists():
        return tools

    for file_path in sorted(root.rglob("*.py")):
        if file_path.name in {"__init__.py", "base.py"} or file_path.name.startswith("_"):
            continue
        tool = _load_tool_file(file_path, root)
        for name in tool.all_names:
            if name in tools:
                raise ValueError(f"Duplicate system tool name: {name}")
            tools[name] = tool
    return tools


def _load_tool_file(file_path: Path, root: Path) -> SystemTool:
    module = _load_module(file_path, root)
    factory = getattr(module, "create_tool", None)
    if callable(factory):
        raw_tool = factory()
    else:
        raw_tool = getattr(module, "SYSTEM_TOOL", None)

    if isinstance(raw_tool, SystemTool):
        return raw_tool
    if isinstance(raw_tool, dict):
        data = dict(raw_tool)
        data.setdefault("group", data.pop("tool_class", data.pop("class", "")))
        return SystemTool(**data)
    raise ValueError(f"{file_path} must export SYSTEM_TOOL or create_tool()")


def _load_module(file_path: Path, root: Path) -> ModuleType:
    rel = file_path.resolve().relative_to(root.resolve())
    stem = ".".join(rel.with_suffix("").parts)
    module_name = f"openagent_system_tool_{stem}_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import system tool file: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def accepts_agent(handler: SystemToolHandler) -> bool:
    if isinstance(handler, str):
        return False
    try:
        return "agent" in inspect.signature(handler).parameters
    except (TypeError, ValueError):
        return False
