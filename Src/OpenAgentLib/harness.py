"""The agent loop: provider calls + tool execution.

``run`` is provider- and UI-agnostic: it walks the chat <-> tool steps of
one request and emits ``tool_start`` / ``tool_done`` events on
``on_event``. The UI (e.g. the telegram module) redraws on those events
and refreshes the elapsed-time header on its own while waiting.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from Settings import (
    CONTEXT_TOOL_RECORD_LIMIT,
    DEBUG,
    DEFAULT_TOOL_TIMEOUT,
    MAX_TOOL_ITERATIONS,
    PROGRESS_UPDATE_INTERVAL,
)
from .Tools import Tool, ToolContext, call_tool, tool_schemas


class AgentError(Exception):
    """Provider or configuration error with a ready user-facing text.

    ``detail`` carries the raw provider error text for the expanded
    error block in the message.
    """

    def __init__(self, text: str, detail: str | None = None):
        super().__init__(text)
        self.detail = detail


@dataclass
class RunInput:
    """Everything ``run`` needs for one request."""

    messages: list[dict[str, Any]]  # system + history + prompt, built by the caller
    tools: list[Tool]
    tool_context: ToolContext
    state: dict[str, Any]  # UI state (tool_bar, thinking notes, ...)
    provider: Any  # .chat(messages, tools) -> (message dict, usage|None)
    logger: Any
    strings: Callable[[str], str]
    on_event: Callable[[str, dict[str, Any]], Awaitable[None]]  # tool_start / tool_done
    on_usage: Callable[[dict[str, Any]], Awaitable[None]]  # per-step token usage
    tool_timeout: float = DEFAULT_TOOL_TIMEOUT


async def _await_ticked(
    inp: RunInput, task: asyncio.Task, event: str
) -> Any:
    """Wait for ``task``, emitting ``event`` every PROGRESS_UPDATE_INTERVAL."""
    while True:
        try:
            done, _ = await asyncio.wait({task}, timeout=PROGRESS_UPDATE_INTERVAL)
        except asyncio.CancelledError:
            task.cancel()
            raise
        if done:
            return task.result()
        await inp.on_event(event, {})


async def run(inp: RunInput) -> str:
    """Run the chat <-> tool loop and return the final answer text."""
    messages = inp.messages
    start_len = len(messages)  # first index of this exchange's new messages
    schemas = tool_schemas(inp.tools)
    state = inp.state

    for step in range(MAX_TOOL_ITERATIONS):
        # The UI keeps refreshing on its own (outer wait loop) while the
        # provider answer is in flight
        message, usage = await inp.provider.chat(messages, schemas)
        if usage is not None:
            await inp.on_usage(usage)
        calls = message.get("tool_calls") or []
        if not calls:
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise AgentError(inp.strings("err_bad_reply"))
            # Compact record of this exchange's tool calls for the context
            record: list[dict[str, str]] = []
            for item in messages[start_len:]:
                if not isinstance(item, dict) or item.get("role") != "tool":
                    continue
                record.append({
                    "name": str(item.get("tool_name") or "?"),
                    "output": str(item.get("content") or ""),
                })
            state["exchange_tools"] = record[:CONTEXT_TOOL_RECORD_LIMIT]
            return content.strip()

        messages.append({
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": calls,
        })

        for index, call in enumerate(calls):
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            raw_arguments = function.get("arguments") or "{}"
            state["tool_calls"] = int(state.get("tool_calls") or 0) + 1

            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except (TypeError, ValueError):
                    arguments = None
            else:
                arguments = raw_arguments

            if not isinstance(arguments, dict):
                result = json.dumps(
                    {"ok": False, "error": "tool arguments must be a JSON object"},
                    ensure_ascii=False,
                )
                state.setdefault("tool_bar", []).append(
                    {
                        "name": name,
                        "args": str(raw_arguments),
                        "status": "err",
                        "result": result,
                        "elapsed": 0.0,
                    }
                )
            else:
                state["tool"] = name
                entry = {
                    "name": name,
                    "args": json.dumps(arguments, ensure_ascii=False),
                    "status": "run",
                    "start": time.perf_counter(),
                }
                state.setdefault("tool_bar", []).append(entry)
                await inp.on_event("tool_start", {"name": name})
                if DEBUG:
                    inp.logger.debug("tool call %s: %s", name, arguments)
                tool_task = asyncio.create_task(
                    call_tool(
                        inp.tools,
                        name,
                        arguments,
                        inp.tool_context,
                        timeout=inp.tool_timeout,
                    )
                )
                result = await _await_ticked(inp, tool_task, "tool_tick")
                entry["result"] = result
                entry["elapsed"] = time.perf_counter() - entry["start"]
                entry["status"] = "ok" if _tool_ok(result) else "err"
                state["tool"] = ""
                # A fast tool may have changed the UI (e.g. added a thinking
                # note) — push it now, since _await_ticked returns as soon as
                # the task is done without a final event.
                await inp.on_event(
                    "tool_done", {"name": name, "status": entry["status"]}
                )

            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id") or f"call_{step}_{index}",
                "tool_name": name,
                "content": result,
            })

    raise AgentError(inp.strings("err_tool_limit"))


def _tool_ok(result: str) -> bool:
    """A tool failed when it returned JSON with ``"ok": false``."""
    try:
        data = json.loads(result)
    except (TypeError, ValueError):
        return True
    return not (isinstance(data, dict) and data.get("ok") is False)
