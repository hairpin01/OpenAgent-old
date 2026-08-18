# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import math
import re
from typing import Any, Iterable

_TRANSIENT_HTTP_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_STATUS_RE = re.compile(r"\bHTTP\s+(\d{3})\b", re.IGNORECASE)


class ModelCallBudget:
    """Count real provider attempts and reserve capacity for finalization."""

    def __init__(self, limit: int) -> None:
        self.limit = max(1, int(limit))
        self.used = 0

    def reserve(self) -> None:
        if self.used >= self.limit:
            raise RuntimeError(f"Agent model-call budget exhausted ({self.limit})")
        self.used += 1

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def tool_rounds(
        self,
        *,
        thinking_enabled: bool,
        configured_steps: int,
        final_attempts: int = 1,
    ) -> int:
        available = self.limit - int(thinking_enabled) - max(1, final_attempts)
        return min(max(0, int(configured_steps)), max(0, available))


def estimate_text_tokens(value: Any) -> int:
    """Return a dependency-free, conservative token estimate.

    ASCII prose/code averages close to four characters per token. Non-ASCII
    text is estimated more conservatively so Cyrillic and other scripts do not
    silently overflow the provider context window.
    """

    text = str(value or "")
    if not text:
        return 0
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return max(1, math.ceil(ascii_chars / 4) + math.ceil(non_ascii_chars / 1.5))


def estimate_messages_tokens(messages: Iterable[dict[str, Any]]) -> int:
    """Estimate chat tokens, including a small per-message framing cost."""

    total = 0
    for message in messages:
        total += 4
        total += estimate_text_tokens(message.get("role", ""))
        total += estimate_text_tokens(message.get("content", ""))
    return total


def trim_messages_to_budget(
    messages: list[dict[str, Any]],
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Keep system messages and the newest conversation within max_tokens."""

    if max_tokens <= 0 or not messages:
        return []
    system = [item for item in messages if item.get("role") == "system"]
    body = [item for item in messages if item.get("role") != "system"]

    def truncate_message(
        item: dict[str, Any],
        budget: int,
    ) -> dict[str, Any] | None:
        if budget <= 8:
            return None
        content = str(item.get("content", ""))
        marker = "\n…[truncated to context budget]"
        target = max(1, budget - 5 - estimate_text_tokens(marker))
        current = max(1, estimate_text_tokens(content))
        size = max(1, int(len(content) * target / current))
        truncated = dict(item)
        truncated["content"] = content[:size].rstrip() + marker
        while size > 1 and estimate_messages_tokens([truncated]) > budget:
            size = max(1, int(size * 0.9))
            truncated["content"] = content[:size].rstrip() + marker
        return truncated if estimate_messages_tokens([truncated]) <= budget else None

    kept: list[dict[str, Any]] = []
    system_budget = max_tokens if not body else max(1, max_tokens * 2 // 3)
    for index, item in enumerate(system):
        cost = estimate_messages_tokens([item])
        remaining_system = system_budget - estimate_messages_tokens(kept)
        if cost > remaining_system:
            if index == 0:
                truncated = truncate_message(item, remaining_system)
                if truncated is not None:
                    kept.append(truncated)
            break
        kept.append(item)
    remaining = max_tokens - estimate_messages_tokens(kept)
    if remaining <= 0:
        return kept

    newest: list[dict[str, Any]] = []
    for item in reversed(body):
        cost = estimate_messages_tokens([item])
        if cost > remaining:
            if newest or remaining <= 8:
                break
            truncated = truncate_message(item, remaining)
            if truncated is not None:
                newest.append(truncated)
            break
        newest.append(item)
        remaining -= cost
    kept.extend(reversed(newest))
    return kept


def should_request_thinking(prompt: str, effort: str, *, flash_mode: bool) -> bool:
    """Decide whether the optional progress-note model call is worthwhile."""

    if flash_mode or effort == "off":
        return False
    if effort in {"medium", "high", "xhigh"}:
        return True
    text = str(prompt or "")
    return (
        len(text) >= 500
        or text.count("\n") >= 4
        or bool(
            re.search(
                r"\b(debug|refactor|migrat|implement|plan|analy[sz]|исправ|рефактор|"
                r"миграц|реализ|проанализ|план)\w*",
                text,
                re.IGNORECASE,
            )
        )
    )


def is_transient_provider_error(exc: BaseException) -> bool:
    """Return True only for failures that are normally safe to retry."""

    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, ConnectionError)):
        return True
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status in _TRANSIENT_HTTP_STATUSES
    text = str(exc)
    match = _STATUS_RE.search(text)
    if match:
        return int(match.group(1)) in _TRANSIENT_HTTP_STATUSES
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "timed out",
            "timeout",
            "connection reset",
            "connection refused",
            "cannot connect",
            "server disconnected",
            "temporarily unavailable",
        )
    )


def retry_delay(attempt: int, *, cap: float = 8.0) -> float:
    """Return deterministic exponential backoff for a one-based attempt."""

    return min(cap, 0.5 * (2 ** max(0, attempt - 1)))


_TOOL_GROUP_KEYWORDS = {
    "code": ("code", "python", "file", "module", "код", "файл", "модул"),
    "context": ("context", "history", "контекст", "истори"),
    "dialog": ("dialog", "chat", "диалог", "чат"),
    "file": ("file", "folder", "directory", "файл", "папк", "директор"),
    "mcub": ("mcub", "userbot", "юзербот"),
    "message": ("message", "send", "сообщ", "отправ"),
    "skills": ("skill", "knowledge", "скилл", "навык"),
    "terminal": ("terminal", "command", "shell", "терминал", "команд"),
    "todo": ("todo", "task", "задач", "план"),
    "web": ("web", "search", "internet", "сайт", "поиск", "интернет"),
}


def relevant_tool_names(
    prompt: str,
    names: Iterable[str],
    *,
    limit: int = 16,
) -> tuple[str, ...]:
    """Select a compact tool index; full discovery remains available on demand."""

    available = sorted(
        {str(name).strip().lower() for name in names if str(name).strip()}
    )
    lowered = str(prompt or "").lower()
    groups = {
        group
        for group, keywords in _TOOL_GROUP_KEYWORDS.items()
        if any(keyword in lowered for keyword in keywords)
    }
    essentials = {
        "thinking.note",
        "utility.list_tools",
        "utility.tool_help",
        "utility.plugin_docs",
    }
    selected = [
        name
        for name in available
        if name in essentials or name.split(".", 1)[0] in groups
    ]
    return tuple(selected[: max(1, limit)])


__all__ = [
    "ModelCallBudget",
    "estimate_messages_tokens",
    "estimate_text_tokens",
    "is_transient_provider_error",
    "relevant_tool_names",
    "retry_delay",
    "should_request_thinking",
    "trim_messages_to_budget",
]
