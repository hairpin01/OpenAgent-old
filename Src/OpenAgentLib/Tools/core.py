from __future__ import annotations

import asyncio
import copy
import importlib
import json
import logging
import pkgutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from Settings import DEFAULT_TOOL_TIMEOUT, MAX_TOOL_OUTPUT_CHARS

ToolHandler = Callable[["ToolContext", dict[str, Any]], Awaitable[str]]


class AgentPlugin:
    """Base class for plugins and tools in OpenAgent.

    A plugin can define a tool that LLMs can call (by setting name, description,
    parameters and implement execute) and/or provide lifecycle hooks for:
    - Pre-execution interception / short-circuit (on_before_call)
    - Post-execution interception and Self-Managed Context (on_after_call)
    - Final response transformation (on_final_response)
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    timeout: float | None = None

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        handler: ToolHandler | None = None,
        timeout: float | None = None,
    ) -> None:
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if parameters is not None:
            self.parameters = parameters
        if timeout is not None:
            self.timeout = timeout
        if handler is not None:
            self._handler = handler
        else:
            self._handler = None

        if self.name:
            if (
                self.name != self.name.lower()
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
            if self.timeout is not None and self.timeout <= 0:
                raise ValueError(f"{self.name}: timeout must be positive")

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> str:
        """Execute the tool action."""
        if self._handler is not None:
            return await self._handler(ctx, args)
        raise NotImplementedError(f"Plugin {self.name or type(self).__name__} does not implement execute()")

    async def on_before_call(
        self, name: str, args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any] | str | None:
        """Hook called before any tool execution.

        Can return:
        - None: continue normal execution.
        - dict: replace / mutate arguments passed to the tool.
        - str: short-circuit execution and return this result immediately.
        """
        return None

    async def on_after_call(
        self,
        name: str,
        args: dict[str, Any],
        result: str,
        ctx: ToolContext,
        messages: list[dict[str, Any]],
    ) -> str:
        """Hook called after any tool execution.

        Allows Self-Managed Context: inspect/modify `messages` (e.g. prune old history,
        compress tool outputs) and/or modify `result`.
        """
        return result

    async def on_final_response(
        self,
        content: str,
        ctx: ToolContext,
        messages: list[dict[str, Any]],
    ) -> str:
        """Hook called on the final assistant response before showing to user."""
        return content


# Backward compatibility alias
Tool = AgentPlugin


@dataclass(slots=True)
class ToolContext:
    logger: logging.Logger
    http: Any = None
    event: Any = None
    chat_id: int | str | None = None
    client: Any = None  # userbot client from the module (not the inline bot)
    tools: list[AgentPlugin] | None = None
    plugins: list[AgentPlugin] | None = None
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


def load_tools(logger: logging.Logger | None = None) -> list[AgentPlugin]:
    """Import Builtin/Custom modules and collect their ``PLUGIN`` or ``TOOL`` objects."""
    log = logger or logging.getLogger(__name__)
    plugins: list[AgentPlugin] = []
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
                log.warning("plugin module %s skipped: %s", module_name, exc)
                continue

            plugin = getattr(module, "PLUGIN", None) or getattr(module, "TOOL", None)
            if not isinstance(plugin, AgentPlugin):
                log.warning("plugin module %s has no PLUGIN or TOOL", module_name)
                continue
            if plugin.name and plugin.name in seen:
                log.warning("duplicate plugin %s in %s skipped", plugin.name, module_name)
                continue

            if plugin.name:
                seen.add(plugin.name)
            plugins.append(plugin)

    return plugins


def tool_schemas(tools: list[AgentPlugin]) -> list[dict[str, Any]]:
    schemas = []
    for tool in tools:
        if not tool.name:
            continue
        params = (
            copy.deepcopy(tool.parameters)
            if tool.parameters
            else {"type": "object", "properties": {}}
        )
        if params.get("type") == "object":
            props = params.setdefault("properties", {})
            if "thinking" not in props and tool.name != "thinking":
                props["thinking"] = {
                    "type": "string",
                    "description": "Short explanation of why you are calling this tool (one short sentence, shown to the user).",
                }
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": params,
                },
            }
        )
    return schemas


def _clip(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text
    return text[:MAX_TOOL_OUTPUT_CHARS].rstrip() + "…"


async def call_tool(
    tools: list[AgentPlugin],
    name: str,
    arguments: dict[str, Any],
    context: ToolContext,
    *,
    timeout: float = DEFAULT_TOOL_TIMEOUT,
    messages: list[dict[str, Any]] | None = None,
) -> str:
    """Execute a tool with on_before_call and on_after_call hooks."""
    cur_messages = messages if messages is not None else []

    # 1. on_before_call hook across all active plugins
    for plugin in tools:
        try:
            hook_res = await plugin.on_before_call(name, arguments, context)
            if isinstance(hook_res, str):
                # Short-circuit execution - run on_after_call as well
                res_str = _clip(hook_res)
                for p in tools:
                    try:
                        res_str = await p.on_after_call(
                            name, arguments, res_str, context, cur_messages
                        )
                    except Exception as exc:
                        context.logger.warning(
                            "on_after_call hook failed in %s: %s", p.name, exc
                        )
                return _clip(res_str)
            if isinstance(hook_res, dict):
                arguments = hook_res
        except Exception as exc:
            context.logger.warning(
                "on_before_call hook failed in %s: %s", plugin.name, exc
            )

    tool = next((item for item in tools if item.name == name), None)
    if tool is None:
        return json.dumps(
            {"ok": False, "error": f"unknown tool: {name}"}, ensure_ascii=False
        )

    # 2. Main execution
    limit = tool.timeout if tool.timeout is not None else timeout
    try:
        raw_result = await asyncio.wait_for(
            tool.execute(context, arguments), timeout=limit
        )
    except asyncio.TimeoutError:
        raw_result = json.dumps(
            {"ok": False, "error": f"tool timed out after {limit}s"},
            ensure_ascii=False,
        )
    except Exception as exc:
        context.logger.warning("tool %s failed: %s", name, exc)
        raw_result = json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)

    result_str = raw_result if isinstance(raw_result, str) else str(raw_result)

    # 3. on_after_call hook across all active plugins (Self-Managed Context)
    cur_messages = messages if messages is not None else []
    for plugin in tools:
        try:
            result_str = await plugin.on_after_call(
                name, arguments, result_str, context, cur_messages
            )
        except Exception as exc:
            context.logger.warning("on_after_call hook failed in %s: %s", plugin.name, exc)

    return _clip(result_str)


async def run_final_response_hooks(
    plugins: list[AgentPlugin],
    content: str,
    context: ToolContext,
    messages: list[dict[str, Any]],
) -> str:
    """Pass assistant's final response through on_final_response hooks."""
    for plugin in plugins:
        try:
            content = await plugin.on_final_response(content, context, messages)
        except Exception as exc:
            context.logger.warning(
                "on_final_response hook failed in %s: %s", plugin.name, exc
            )
    return content
