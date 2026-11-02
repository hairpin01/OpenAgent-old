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
    provider: Any  # .chat(messages, tools, on_delta) -> (message dict, usage|None)
    logger: Any
    strings: Callable[[str], str]
    on_event: Callable[[str, dict[str, Any]], Awaitable[None]]  # tool_start / tool_done
    on_usage: Callable[[dict[str, Any]], Awaitable[None]]  # per-step token usage
    tool_timeout: float = DEFAULT_TOOL_TIMEOUT
    # Live answer text (everything received so far); None disables streaming
    on_stream: Callable[[str], Awaitable[None]] | None = None


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


class _StreamGate:
    """Decide which part of a streamed reply is shown live.

    A reply that starts with "{" is a tool call written as JSON and is never
    shown; any other reply is passed on as it arrives (the whole text so far,
    on every piece).
    """

    def __init__(self, emit: Callable[[str], Awaitable[None]]) -> None:
        self._emit = emit
        self._text = ""
        self._live: bool | None = None  # None: not decided yet

    async def feed(self, delta: str) -> None:
        self._text += delta
        if self._live is None:
            head = self._text.lstrip()
            if not head:
                return
            self._live = not head.startswith("{")
        if self._live:
            await self._emit(self._text)

    async def reset(self) -> None:
        """Drop the live text (the step ended in tool calls or was redone)."""
        if self._live:
            await self._emit("")


def _json_tool_calls(content: Any, tools: list[Tool]) -> list[dict[str, Any]]:
    """Tool calls a model wrote as a JSON object in the reply text.

    Understands ``{"name": ..., "arguments": {...}}`` (also ``parameters``,
    or nested under ``function``) and ``{"tool_calls": [...]}``. Only names of
    known tools count, so a JSON answer the user asked for stays an answer.
    """
    if not isinstance(content, str) or not content.lstrip().startswith("{"):
        return []
    try:
        data = json.loads(content)
    except ValueError:
        return []

    known = {tool.name for tool in tools}
    items = data.get("tool_calls") if isinstance(data, dict) else None
    if not isinstance(items, list):
        items = [data]

    calls: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            function = item
        name = function.get("name")
        if not isinstance(name, str) or name not in known:
            continue
        arguments = function.get("arguments", function.get("parameters", {}))
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        calls.append(
            {
                "id": f"call_json_{len(calls)}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return calls


def _drain_supplements(messages: list[dict[str, Any]], state: dict[str, Any]) -> None:
    """Move queued user supplements into the dialogue as user messages."""
    pending = state.get("supplements")
    if not pending:
        return
    state["supplements"] = []
    for text in pending:
        messages.append({"role": "user", "content": text})
    # Keep the supplements visible in the prompt block
    state["prompt"] = "\n".join([str(state.get("prompt") or ""), *pending]).strip()


async def run(inp: RunInput) -> str:
    """Run the chat <-> tool loop and return the final answer text."""
    messages = inp.messages
    start_len = len(messages)  # first index of this exchange's new messages
    schemas = tool_schemas(inp.tools)
    state = inp.state

    for step in range(MAX_TOOL_ITERATIONS):
        _drain_supplements(messages, state)
        # The UI keeps refreshing on its own (outer wait loop) while the
        # provider answer is in flight
        gate = _StreamGate(inp.on_stream) if inp.on_stream else None
        message, usage = await inp.provider.chat(
            messages, schemas, on_delta=gate.feed if gate else None
        )
        if usage is not None:
            await inp.on_usage(usage)
        calls = message.get("tool_calls") or []
        if not calls:
            calls = _json_tool_calls(message.get("content"), inp.tools)
            if calls:
                message = {**message, "content": "", "tool_calls": calls}
        if not calls and state.get("supplements"):
            # Extra input arrived while the final answer was in flight:
            # keep that answer in the dialogue and answer again with it
            messages.append(
                {"role": "assistant", "content": message.get("content") or ""}
            )
            if gate:
                await gate.reset()
            continue
        if calls and gate:
            await gate.reset()
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
                thinking_text = arguments.pop("thinking", None)
                # The "thinking" tool writes its own notes in its handler;
                # counting the injected pseudo-argument too would add the same
                # note twice.
                if name != "thinking" and thinking_text and isinstance(thinking_text, str):
                    clean_thinking = " ".join(thinking_text.split())
                    if clean_thinking:
                        lines = state.setdefault("thinking_lines", [])
                        lines.append(clean_thinking)
                        del lines[:-3]

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
