from __future__ import annotations

import asyncio
import importlib
import json
import logging
import pkgutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from Settings import DEFAULT_TOOL_TIMEOUT, MAX_TOOL_OUTPUT_CHARS

ToolHandler = Callable[["ToolContext", dict[str, Any]], Awaitable[str]]


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    timeout: float | None = None  # overrides DEFAULT_TOOL_TIMEOUT for this tool

    def __post_init__(self) -> None:
        if (
            not self.name
            or self.name != self.name.lower()
            or not self.name.isascii()
            or not self.name.replace("_", "").isalnum()
            or self.name[0].isdigit()
        ):
            raise ValueError(f"invalid tool name: {self.name!r}")
        if not self.description.strip():
            raise ValueError(f"{self.name}: description is required")
        if not isinstance(self.parameters, dict):
            raise ValueError(f"{self.name}: parameters must be a dict")
        if self.parameters.get("type") != "object":
            raise ValueError(f"{self.name}: parameters must be an object schema")
        if not callable(self.handler):
            raise ValueError(f"{self.name}: handler must be callable")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError(f"{self.name}: timeout must be positive")


@dataclass(slots=True)
class ToolContext:
    logger: logging.Logger
    http: Any = None
    event: Any = None
    chat_id: int | str | None = None
    tools: list[Tool] | None = None
    state: dict[str, Any] = field(default_factory=dict)  # active request state

    def add_module(
        self,
        format_name: str,
        module: str,
        fn: Callable[..., str | None] | None = None,
        *,
        after: str | None = None,
    ) -> None:
        """Add a text module to ``format_name`` for the current request only.

        ``fn`` is optional: without it the module is taken from TEXT_MODULES.
        The module is placed after ``after``, or at the end of the format.
        """
        entries = self.state.setdefault("ui", {}).setdefault(format_name, [])
        if any(item[1] == module for item in entries):
            return
        entries.append((after, module, fn))


def load_tools(logger: logging.Logger | None = None) -> list[Tool]:
    """Import Builtin/Custom modules and collect their ``TOOL`` objects."""
    log = logger or logging.getLogger(__name__)
    tools: list[Tool] = []
    seen: set[str] = set()

    from . import Builtin, Custom

    for package in (Builtin, Custom):
        modules = sorted(
            pkgutil.iter_modules(package.__path__), key=lambda item: item.name
        )
        for info in modules:
            if info.name.startswith("_"):
                continue

            module_name = f"{package.__name__}.{info.name}"
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:
                log.warning("tool module %s skipped: %s", module_name, exc)
                continue

            tool = getattr(module, "TOOL", None)
            if not isinstance(tool, Tool):
                log.warning("tool module %s has no TOOL", module_name)
                continue
            if tool.name in seen:
                log.warning("duplicate tool %s in %s skipped", tool.name, module_name)
                continue

            seen.add(tool.name)
            tools.append(tool)

    return tools


def tool_schemas(tools: list[Tool]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def _clip(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text
    return text[:MAX_TOOL_OUTPUT_CHARS].rstrip() + "…"


async def call_tool(
    tools: list[Tool],
    name: str,
    arguments: dict[str, Any],
    context: ToolContext,
    *,
    timeout: float = DEFAULT_TOOL_TIMEOUT,
) -> str:
    tool = next((item for item in tools if item.name == name), None)
    if tool is None:
        return json.dumps(
            {"ok": False, "error": f"unknown tool: {name}"}, ensure_ascii=False
        )

    limit = tool.timeout if tool.timeout is not None else timeout
    try:
        result = await asyncio.wait_for(
            tool.handler(context, arguments), timeout=limit
        )
    except asyncio.TimeoutError:
        return json.dumps(
            {"ok": False, "error": f"tool timed out after {limit}s"},
            ensure_ascii=False,
        )
    except Exception as exc:
        context.logger.warning("tool %s failed: %s", name, exc)
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)

    return _clip(result if isinstance(result, str) else str(result))
