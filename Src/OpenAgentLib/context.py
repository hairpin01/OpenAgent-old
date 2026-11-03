"""Chat context: DB-backed history store and provider message building.

``ContextStore`` keeps the (user, assistant) pairs of one conversation in
the cubkit DB; ``build_messages`` turns a config + history + prompt into
the message list the provider receives. Neither part knows about the
telegram UI or the harness loop.
"""

from __future__ import annotations

import ast
import json
from typing import Any

from Settings import (
    CONTEXT_TOOL_RECORD_CHARS,
    MAX_CONTEXT_MESSAGES,
    RICH_HTML_RULES,
    TOOL_RULES,
)
from .Tools import Tool


class ContextStore:
    """One conversation's history, stored under ``ctx_{chat_id}``."""

    def __init__(self, db: Any, prefix: str, log: Any) -> None:
        self._db = db
        self._prefix = prefix
        self._log = log

    @staticmethod
    def limit_messages() -> int:
        """Messages kept in context; 0 in config means unlimited (hard cap)."""
        return MAX_CONTEXT_MESSAGES * 2

    def _key(self, chat_id: int | str) -> str:
        return f"ctx_{chat_id}"

    async def load(self, chat_id: int | str) -> list[dict[str, Any]]:
        try:
            raw = await self._db.db_get(self._prefix, self._key(chat_id))
        except Exception as exc:
            self._log.error("OpenAgent context load failed: %s", exc)
            return []

        if raw is None:
            return []

        if isinstance(raw, str):
            s = raw.strip()
            if not s:
                return []
            try:
                raw = json.loads(s)
            except ValueError:
                try:
                    raw = ast.literal_eval(s)
                except (ValueError, SyntaxError):
                    self._log.error("OpenAgent ctx unparsable: %r", raw[:200])
                    return []

        if not isinstance(raw, list):
            return []

        limit = self.limit_messages()
        out: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content:
                entry: dict[str, Any] = {"role": role, "content": content}
                if isinstance(item.get("tools"), list):
                    entry["tools"] = item["tools"]
                out.append(entry)
        out = out[-limit:]
        if len(out) % 2:
            out = out[1:]  # keep user/assistant pairs aligned
        return out

    async def save(self, chat_id: int | str, history: list[dict[str, str]]) -> None:
        try:
            await self._db.db_set(self._prefix, self._key(chat_id), history)
        except Exception as exc:
            self._log.error("OpenAgent context save failed: %s", exc)

    async def clear(self, chat_id: int | str) -> int:
        history = await self.load(chat_id)
        removed = len(history)
        if removed:
            await self.save(chat_id, [])
        return removed

    async def append(
        self,
        chat_id: int | str,
        prompt: str,
        answer: str,
        tools_record: list[dict[str, str]] | None = None,
    ) -> int:
        history = await self.load(chat_id)
        answer_entry: dict[str, Any] = {"role": "assistant", "content": answer}
        if tools_record:
            answer_entry["tools"] = tools_record
        history.append({"role": "user", "content": prompt})
        history.append(answer_entry)
        limit = self.limit_messages()
        if len(history) > limit:
            history = history[-limit:]
            if len(history) % 2:
                history = history[1:]  # keep user/assistant pairs aligned
        await self.save(chat_id, history)
        return len(history) // 2


def history_to_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert stored context entries to provider messages.

    Tool records saved alongside an answer are inlined into it as a
    compact transcript, so the model keeps memory of which tools ran
    and what they returned.
    """
    out: list[dict[str, Any]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role == "user" and isinstance(content, str) and content:
            out.append({"role": "user", "content": content})
        elif role == "assistant" and isinstance(content, str):
            message: dict[str, Any] = {"role": "assistant", "content": content}
            calls = item.get("tool_calls")
            if isinstance(calls, list) and calls:
                message["tool_calls"] = calls
            tools = item.get("tools")
            if isinstance(tools, list) and tools:
                lines = [
                    f"[{i}] {str(rec.get('name') or '?')}: "
                    f"{str(rec.get('output') or '')[:CONTEXT_TOOL_RECORD_CHARS]}"
                    for i, rec in enumerate(tools, 1)
                    if isinstance(rec, dict)
                ]
                if lines:
                    transcript = (
                        "\n<tool-outputs>\n" + "\n".join(lines) + "\n</tool-outputs>"
                    )
                    message["content"] = (
                        content + transcript if content else transcript.strip()
                    )
            out.append(message)
    return out


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token count (about 4 chars per token) of a message list.

    Only used until the provider reports real usage for the request.
    """
    try:
        return len(json.dumps(messages, ensure_ascii=False)) // 4
    except (TypeError, ValueError):
        return 0


def build_messages(
    system_prompt: str,
    tool_rules: bool,
    tools: list[Tool],
    history: list[dict[str, Any]],
    prompt: str,
) -> list[dict[str, Any]]:
    """Assemble the provider message list for one request."""
    system = str(system_prompt).strip()
    # The built-in rules always apply, even when the user prompt is empty
    rules = f"<rules>{RICH_HTML_RULES}</rules>"
    if tool_rules and tools:
        listing = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
        rules += f"\n<tool-rules>{TOOL_RULES.format(tools=listing)}</tool-rules>"
    if system:
        rules += "\nUser system prompt:\n<user-prompt>" + system + "</user-prompt>"

    messages: list[dict[str, Any]] = [{"role": "system", "content": rules}]
    messages.extend(history_to_messages(history))
    messages.append({"role": "user", "content": prompt})
    return messages
