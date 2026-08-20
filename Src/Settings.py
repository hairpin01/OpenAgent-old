# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any

# One switch for verbose OpenAgent runtime tracing. Keep enabled while debugging
# agent-loop/provider behavior; production builds can disable it here.
DEBUG = False
DEBUG_MAX_STRING_CHARS = 8_000
DEBUG_MAX_EVENT_CHARS = 32_000
DEBUG_MAX_COLLECTION_ITEMS = 100

_SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "cookie",
    "password",
    "secret",
    "telegram_bot_token",
    "token",
}
_CONTENT_KEYS = {
    "answer",
    "attachments",
    "attrs_raw",
    "body",
    "candidate",
    "content",
    "data",
    "messages",
    "note",
    "output",
    "outputs",
    "prompt",
    "raw_answer",
    "response",
    "result",
    "text",
    "thinking_notes",
    "tool_trace",
}

_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b", re.IGNORECASE),
    re.compile(
        r"(?i)([\"']?(?:api[_-]?key|access[_-]?token|telegram[_-]?bot[_-]?token|"
        r"authorization|cookie|password|secret)[\"']?\s*[:=]\s*[\"']?)"
        r"([^\"'\s&,;}]{4,})"
    ),
)


def configure_debug(enabled: bool) -> None:
    """Configure tracing for the loaded artifact without mutating source files."""

    global DEBUG
    DEBUG = bool(enabled)


def debug_for_artifact(path: Any) -> bool:
    """Return whether an artifact filename identifies a debug build."""

    return "debug" in Path(str(path or "")).stem.lower()


def _redact_text(value: str) -> str:
    text = value
    text = _SECRET_PATTERNS[0].sub(r"\1<redacted>", text)
    text = _SECRET_PATTERNS[1].sub("<redacted-telegram-token>", text)
    text = _SECRET_PATTERNS[2].sub("<redacted-api-key>", text)
    text = _SECRET_PATTERNS[3].sub(r"\1<redacted>", text)
    if len(text) > DEBUG_MAX_STRING_CHARS:
        return text[:DEBUG_MAX_STRING_CHARS] + f"…<truncated:{len(text)}>"
    return text


def _is_sensitive_key(value: Any) -> bool:
    key = str(value or "").strip().lower().replace("-", "_")
    return any(part in _SENSITIVE_KEYS for part in key.split("."))


def _is_content_key(value: Any) -> bool:
    key = str(value or "").strip().lower().replace("-", "_")
    return key in _CONTENT_KEYS


def _content_summary(value: str) -> dict[str, Any]:
    redacted = _redact_text(value)
    return {
        "length": len(value),
        "sha256": hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest(),
        "preview": redacted[:2_000],
    }


def _debug_value(value: Any, *, key: str = "", depth: int = 0) -> Any:
    if _is_sensitive_key(key):
        return "<redacted>"
    if depth >= 8:
        return "<max-depth>"
    if isinstance(value, dict):
        return {
            str(item_key): _debug_value(
                item_value,
                key=str(item_key),
                depth=depth + 1,
            )
            for item_key, item_value in list(value.items())[:DEBUG_MAX_COLLECTION_ITEMS]
        }
    if isinstance(value, (list, tuple, set)):
        return [
            _debug_value(item, key=key, depth=depth + 1)
            for item in list(value)[:DEBUG_MAX_COLLECTION_ITEMS]
        ]
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, str):
        if _is_content_key(key):
            return _content_summary(value)
        return _redact_text(value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return repr(value)


def debug_log(logger: Any, event: str, **fields: Any) -> None:
    """Write one structured OpenAgent trace line without leaking credentials."""

    if not DEBUG:
        return
    try:
        payload = {
            "event": str(event),
            **{
                str(key): _debug_value(value, key=str(key))
                for key, value in fields.items()
            },
        }
        text = "[OpenAgent DEBUG] " + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(text) > DEBUG_MAX_EVENT_CHARS:
            text = text[:DEBUG_MAX_EVENT_CHARS] + f"…<event-truncated:{len(text)}>"
        log_method = getattr(logger, "info", None) or getattr(logger, "warning", None)
        if callable(log_method):
            log_method(text)
    except Exception:
        return


__all__ = [
    "DEBUG",
    "DEBUG_MAX_COLLECTION_ITEMS",
    "DEBUG_MAX_EVENT_CHARS",
    "DEBUG_MAX_STRING_CHARS",
    "configure_debug",
    "debug_for_artifact",
    "debug_log",
]
