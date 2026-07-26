# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Callable
import importlib.util
import inspect
import sys


SystemToolHandler = str | Callable[..., Any]


@dataclass(frozen=True)
class SystemTool:
    """Strict descriptor for one built-in OpenAgent tool.

    Recommended file format::

        from OpenAgentLib.SystemPlugins import SystemTool

        SYSTEM_TOOL = SystemTool(
            tool_class="file",
            name="edit",
            handler="handle",
            docs={"desc": "Edit a file", "args": "path", "body": "content"},
        )

        async def handle(agent, attrs_raw, body, **kwargs):
            ...

    The descriptor is validated at startup. String handlers are resolved lazily
    from the same file on first dispatch, so importing the registry does not bind
    every handler eagerly.
    """

    tool_class: str
    name: str
    handler: SystemToolHandler = "handle"
    docs: dict[str, str] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    dangerous: bool = False
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    api_version: str = "1"
    source_path: Path | None = field(default=None, compare=False, repr=False)
    module_name: str | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_class", self._clean_identifier(self.tool_class, "tool_class"))
        object.__setattr__(self, "name", self._clean_identifier(self.name, "name"))
        object.__setattr__(self, "api_version", str(self.api_version or "1"))
        if not isinstance(self.docs, dict) or not str(self.docs.get("desc") or "").strip():
            raise ValueError(f"{self.full_name}: docs.desc is required")
        if isinstance(self.handler, str):
            clean_handler = self.handler.strip()
            if not clean_handler:
                raise ValueError(f"{self.full_name}: handler is required")
            object.__setattr__(self, "handler", clean_handler)
        elif not callable(self.handler):
            raise ValueError(f"{self.full_name}: handler must be a method name or callable")
        object.__setattr__(
            self,
            "aliases",
            tuple(str(alias).strip().lower() for alias in self.aliases if str(alias or "").strip()),
        )

    @staticmethod
    def _clean_identifier(value: str, field_name: str) -> str:
        clean = str(value or "").strip().lower()
        if not clean:
            raise ValueError(f"SystemTool {field_name} is required")
        return clean

    @property
    def full_name(self) -> str:
        return f"{self.tool_class}.{self.name}"

    @property
    def all_names(self) -> tuple[str, ...]:
        names = [self.full_name]
        for alias in self.aliases:
            if alias not in names:
                names.append(alias)
        return tuple(names)

    def with_source(self, source_path: Path, module_name: str) -> SystemTool:
        return replace(self, source_path=source_path, module_name=module_name)

    def validate_handler_reference(self) -> None:
        if callable(self.handler):
            return
        if self.source_path is None or self.module_name is None:
            raise ValueError(f"{self.full_name}: lazy handler has no source module")
        module = _module_from_cache_or_file(self.module_name, self.source_path)
        handler = getattr(module, self.handler, None)
        if not callable(handler):
            raise ValueError(f"{self.full_name}: handler {self.handler!r} is not callable")

    def resolve_handler(self) -> Callable[..., Any]:
        if callable(self.handler):
            return self.handler
        self.validate_handler_reference()
        assert self.module_name is not None and self.source_path is not None
        module = _module_from_cache_or_file(self.module_name, self.source_path)
        return getattr(module, self.handler)


class SystemToolRegistry:
    """Registry for bundled one-file system tools."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parent
        self.tools: dict[str, SystemTool] = {}

    def load(self) -> dict[str, SystemTool]:
        self.tools = discover_system_tools(self.root)
        return self.tools

    def validate(self) -> None:
        for tool_name, tool in self.tools.items():
            if tool_name not in tool.all_names:
                raise ValueError(f"Invalid registry entry {tool_name!r} for {tool.full_name}")
            tool.validate_handler_reference()

    def tool_map(self) -> dict[str, SystemTool]:
        return dict(self.tools)

    def docs(self) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for tool_name, tool in self.tools.items():
            entry = dict(tool.docs)
            entry.setdefault("api_version", tool.api_version)
            if tool.dangerous:
                entry.setdefault("dangerous", "true")
            if tool_name != tool.full_name:
                entry.setdefault("alias_of", tool.full_name)
            result[tool_name] = entry
        return result


class UserPluginRegistry:
    """Small adapter that keeps user/plugin tool registration separate."""

    def __init__(self, plugins: dict[str, Any]) -> None:
        self.plugins = plugins

    def tool_map(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for plugin in self.plugins.values():
            for tool_name, handler in getattr(plugin, "tool_map", {}).items():
                if tool_name and handler:
                    result[str(tool_name).strip().lower()] = str(handler).strip()
        return result

    def tool_names(self) -> set[str]:
        names: set[str] = set()
        for plugin in self.plugins.values():
            names.update(str(item).strip().lower() for item in getattr(plugin, "tool_registry", ()) if str(item or "").strip())
            names.update(str(item).strip().lower() for item in getattr(plugin, "tool_map", {}).keys() if str(item or "").strip())
        return names


def discover_system_tools(root: Path | None = None) -> dict[str, SystemTool]:
    """Load all SystemPlugins recursively without hardcoded plugin names."""

    registry = SystemToolRegistry(root) if not isinstance(root, SystemToolRegistry) else root
    root_path = registry.root
    tools: dict[str, SystemTool] = {}
    if not root_path.exists():
        return tools

    for file_path in sorted(root_path.rglob("*.py")):
        if file_path.name in {"__init__.py", "base.py"} or file_path.name.startswith("_"):
            continue
        tool = _load_tool_file(file_path, root_path)
        for name in tool.all_names:
            if name in tools:
                raise ValueError(f"Duplicate system tool name: {name}")
            tools[name] = tool
    return tools


def _load_tool_file(file_path: Path, root: Path) -> SystemTool:
    module_name = _module_name(file_path, root)
    module = _module_from_cache_or_file(module_name, file_path)
    factory = getattr(module, "create_tool", None)
    raw_tool = factory() if callable(factory) else getattr(module, "SYSTEM_TOOL", None)

    if isinstance(raw_tool, SystemTool):
        tool = raw_tool
    elif isinstance(raw_tool, dict):
        data = dict(raw_tool)
        if "group" in data and "tool_class" not in data:
            data["tool_class"] = data.pop("group")
        if "class" in data and "tool_class" not in data:
            data["tool_class"] = data.pop("class")
        tool = SystemTool(**data)
    else:
        raise ValueError(f"{file_path} must export SYSTEM_TOOL or create_tool()")
    return tool.with_source(file_path, module_name)


def _module_name(file_path: Path, root: Path) -> str:
    rel = file_path.resolve().relative_to(root.resolve())
    stem = ".".join(rel.with_suffix("").parts)
    return f"OpenAgentLib.SystemPlugins._loaded.{stem.replace('-', '_')}"


def _module_from_cache_or_file(module_name: str, file_path: Path) -> ModuleType:
    cached = sys.modules.get(module_name)
    if isinstance(cached, ModuleType):
        return cached
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import system tool file: {file_path}")
    module = importlib.util.module_from_spec(spec)
    _ensure_tool_api_module()
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _ensure_tool_api_module() -> None:
    """Expose a tiny import target for standalone SystemPlugin files.

    Importing ``OpenAgentLib.SystemPlugins`` can pull in the whole OpenAgent
    package in test/minimal environments. Tool files import this lightweight API
    instead, while still using the strict SystemTool class.
    """

    api_name = "openagent_system_tool_api"
    module = sys.modules.get(api_name)
    if not isinstance(module, ModuleType):
        module = ModuleType(api_name)
        sys.modules[api_name] = module
    module.SystemTool = SystemTool


def accepts_agent(handler: SystemToolHandler) -> bool:
    try:
        target = handler if callable(handler) else None
        return bool(target and "agent" in inspect.signature(target).parameters)
    except (TypeError, ValueError):
        return False
