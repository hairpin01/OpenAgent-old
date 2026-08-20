# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import html
import json
import math
import re
from typing import Any, Iterable

_TRANSIENT_HTTP_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_STATUS_RE = re.compile(r"\bHTTP\s+(\d{3})\b", re.IGNORECASE)
_FINAL_FENCE_RE = re.compile(
    r"```final(?:_answer)?[ \t]*\r?\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_FINAL_TAG_RE = re.compile(
    r"<final(?:_answer)?>(.*?)</final(?:_answer)?>",
    re.IGNORECASE | re.DOTALL,
)


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


def extract_explicit_final(value: Any) -> str | None:
    """Extract an explicitly marked final answer from model output."""

    text = str(value or "").strip()
    if not text:
        return None
    has_final_marker = bool(
        re.search(
            r"```final(?:_answer)?[ \t]*\r?\n|<final(?:_answer)?>",
            text,
            re.IGNORECASE,
        )
    )
    if has_final_marker and re.search(
        r"```tool_call\b|<tool_call\b",
        text,
        re.IGNORECASE,
    ):
        return None
    for pattern in (_FINAL_FENCE_RE, _FINAL_TAG_RE):
        match = pattern.search(text)
        if match:
            final = match.group(1).strip()
            return final or None
    return None


def strip_explicit_final_regions(value: Any) -> str:
    """Remove final-answer regions before scanning output for executable tools."""

    text = str(value or "")
    if re.search(
        r"```final(?:_answer)?[ \t]*\r?\n|<final(?:_answer)?>",
        text,
        re.IGNORECASE,
    ) and re.search(r"```tool_call\b|<tool_call\b", text, re.IGNORECASE):
        return ""
    for pattern in (_FINAL_FENCE_RE, _FINAL_TAG_RE):
        text = pattern.sub("", text)
    return text


def intermediate_note(value: Any, *, limit: int = 1200) -> str:
    """Normalize non-final model prose for the user-visible thinking log."""

    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"```tool_call.*?```", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(
        r"<tool_call>.*?</tool_call>", "", text, flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text[: max(1, int(limit))]


def classify_agent_output(value: Any, *, has_tool_calls: bool) -> tuple[str, str]:
    """Classify one model turn as tools, explicit final, or intermediate prose."""

    if has_tool_calls:
        return "tools", ""
    final = extract_explicit_final(value)
    if final is not None:
        return "final", final
    return "intermediate", intermediate_note(value)


def accepts_completion_verdict(value: Any) -> bool:
    """Accept only the completion gate's exact affirmative token."""

    return bool(re.match(r"^\s*ACCEPT\s*$", str(value or ""), re.IGNORECASE))


def model_tool_reason(values: dict[str, Any], *, limit: int = 700) -> str:
    """Return an optional model-authored reason/comment for a tool call."""

    reason = str(
        values.get("reason") or values.get("comment") or values.get("purpose") or ""
    ).strip()
    return reason[: max(1, int(limit))]


def json_tool_payload_to_legacy(
    payload: dict[str, Any],
    allowed_names: Iterable[str],
) -> tuple[str, str, str] | None:
    """Convert JSON tool protocol to validated legacy attrs/body."""

    tool_name = str(payload.get("tool") or payload.get("name") or "").lower().strip()
    allowed = {str(name).strip().lower() for name in allowed_names}
    if tool_name not in allowed:
        return None
    args_raw = payload.get("args") or {}
    if not isinstance(args_raw, dict):
        args_raw = {}
    body_value = payload.get("body")
    if body_value is None:
        for key in ("body", "content", "text", "message", "command", "query", "prompt"):
            if key in args_raw:
                body_value = args_raw.get(key)
                break
    body = "" if body_value is None else str(body_value)
    attrs: list[str] = []
    for reason_key in ("reason", "comment", "purpose"):
        reason_value = payload.get(reason_key)
        if reason_value is not None and reason_key not in args_raw:
            attrs.append(f'{reason_key}="{html.escape(str(reason_value), quote=True)}"')
    for key, value in args_raw.items():
        if value is None or key == "body":
            continue
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, ensure_ascii=False)
        safe_key = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(key).strip())
        if safe_key:
            attrs.append(f'{safe_key}="{html.escape(str(value), quote=True)}"')
    return tool_name, " ".join(attrs), body


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


def relevant_tool_names(
    prompt: str,
    names: Iterable[str],
) -> tuple[str, ...]:
    """Return the complete stable tool index without keyword-based routing."""

    del prompt
    available = sorted(
        {str(name).strip().lower() for name in names if str(name).strip()}
    )
    return tuple(available)


def search_tool_docs(
    query: str,
    docs: dict[str, dict[str, Any]],
    *,
    limit: int = 8,
) -> tuple[str, ...]:
    """Rank tools by name and normalized documentation text."""

    needle = str(query or "").strip().lower()
    terms = tuple(dict.fromkeys(re.findall(r"[\w.-]+", needle, re.UNICODE)))
    if not terms:
        return ()
    ranked: list[tuple[int, str]] = []
    for raw_name, raw_doc in docs.items():
        name = str(raw_name).strip().lower()
        doc_text = json.dumps(raw_doc, ensure_ascii=False, default=str).lower()
        haystack = f"{name} {doc_text}"
        matched = sum(term in haystack for term in terms)
        if not matched:
            continue
        score = matched * 20
        if needle == name:
            score += 1000
        elif needle in name:
            score += 300
        score += sum(100 for term in terms if term in name)
        if matched == len(terms):
            score += 50
        ranked.append((score, name))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return tuple(name for _score, name in ranked[: max(1, int(limit))])


__all__ = [
    "ModelCallBudget",
    "accepts_completion_verdict",
    "classify_agent_output",
    "extract_explicit_final",
    "estimate_messages_tokens",
    "estimate_text_tokens",
    "is_transient_provider_error",
    "intermediate_note",
    "json_tool_payload_to_legacy",
    "model_tool_reason",
    "relevant_tool_names",
    "retry_delay",
    "search_tool_docs",
    "should_request_thinking",
    "strip_explicit_final_regions",
    "trim_messages_to_budget",
]
