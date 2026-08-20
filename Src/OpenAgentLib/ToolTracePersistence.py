# SPDX-License-Identifier: MIT
"""Safe, terminal-only session envelopes for v2 tool execution."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from typing import Any, Mapping

TOOL_TRACE_SCHEMA_VERSION = 1
_TERMINAL_STATUSES = frozenset({"success", "error", "cancelled", "timed_out"})
_TERMINAL_STATES = {
    "success": "completed",
    "error": "failed",
    "cancelled": "cancelled",
    "timed_out": "timed_out",
}
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|credential|password|secret|token|cookie|"
    r"private[_-]?key|grant|capability|environment|stderr|stack|traceback)",
    re.IGNORECASE,
)
_SAFE_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MAX_DEPTH = 6
_MAX_ITEMS = 40
_MAX_TEXT = 2048
_MAX_ID = 256


class ToolTracePersistence:
    """Create, validate, merge, and render redacted terminal trace envelopes."""

    @staticmethod
    def _text(value: Any, *, limit: int) -> str | None:
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value[:limit] if value else None

    @classmethod
    def _json_safe(cls, value: Any, depth: int = 0) -> Any:
        if depth >= _MAX_DEPTH:
            return "[depth-limited]"
        if value is None or isinstance(value, (bool, int, float, str)):
            if isinstance(value, float) and (
                value != value or value in (float("inf"), float("-inf"))
            ):
                return "[non-json-number]"
            if isinstance(value, str):
                return value[:_MAX_TEXT]
            return value
        if isinstance(value, Mapping):
            safe: dict[str, Any] = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= _MAX_ITEMS:
                    safe["[truncated]"] = True
                    break
                key_text = str(key)[:128]
                safe[key_text] = (
                    "[redacted]"
                    if _SECRET_KEY.search(key_text)
                    else cls._json_safe(item, depth + 1)
                )
            return safe
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item, depth + 1) for item in value[:_MAX_ITEMS]]
        return "[non-json-value]"

    @classmethod
    def _timestamp(cls, value: Any) -> str:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @classmethod
    def from_terminal(cls, trace: Any, result: Any) -> dict[str, Any] | None:
        """Return a redacted terminal record, or reject non-terminal/malformed input."""
        status = cls._text(getattr(result, "status", None), limit=32)
        if hasattr(getattr(result, "status", None), "value"):
            status = getattr(result.status, "value", None)
        if status not in _TERMINAL_STATUSES:
            return None
        state = getattr(trace, "state", None)
        if hasattr(state, "value"):
            state = state.value
        if state != _TERMINAL_STATES[status]:
            return None
        call_id = cls._text(getattr(trace, "call_id", None), limit=_MAX_ID)
        correlation_id = cls._text(
            getattr(trace, "correlation_id", None), limit=_MAX_ID
        )
        if (
            not call_id
            or not correlation_id
            or getattr(result, "call_id", None) != call_id
        ):
            return None
        error = getattr(result, "error", None)
        error_code = getattr(error, "code", None)
        if hasattr(error_code, "value"):
            error_code = error_code.value
        error_code = cls._text(error_code, limit=64)
        if error_code and not _SAFE_CODE.fullmatch(error_code):
            error_code = "executor_failed"
        # Errors carry only a stable code: host stderr and handler exception text
        # must never cross the session boundary.
        output = (
            cls._json_safe(getattr(result, "output", None))
            if status == "success"
            else None
        )
        spill_ref = None
        if isinstance(output, Mapping):
            candidate = output.get("spill_ref") or output.get("spill_path")
            spill_ref = cls._text(candidate, limit=_MAX_TEXT)
            if spill_ref:
                output = {"spill_ref": spill_ref}
        record: dict[str, Any] = {
            "version": TOOL_TRACE_SCHEMA_VERSION,
            "call_id": call_id,
            "correlation_id": correlation_id,
            "status": status,
            "state": state,
            "updated_at": cls._timestamp(getattr(trace, "updated_at", None)),
        }
        if error_code:
            record["error_code"] = error_code
        if output is not None:
            record["output"] = output
        if spill_ref:
            record["spill_ref"] = spill_ref
        return record if cls.validate_record(record) else None

    @classmethod
    def validate_record(cls, record: Any) -> dict[str, Any] | None:
        """Validate persisted data strictly; invalid records are not restored."""
        if (
            not isinstance(record, Mapping)
            or record.get("version") != TOOL_TRACE_SCHEMA_VERSION
        ):
            return None
        call_id = cls._text(record.get("call_id"), limit=_MAX_ID)
        correlation_id = cls._text(record.get("correlation_id"), limit=_MAX_ID)
        status = cls._text(record.get("status"), limit=32)
        state = cls._text(record.get("state"), limit=32)
        updated_at = cls._text(record.get("updated_at"), limit=64)
        if not call_id or not correlation_id or status not in _TERMINAL_STATUSES:
            return None
        if state != _TERMINAL_STATES[status] or not updated_at:
            return None
        try:
            datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        except ValueError:
            return None
        error_code = cls._text(record.get("error_code"), limit=64)
        if error_code and not _SAFE_CODE.fullmatch(error_code):
            return None
        output = record.get("output")
        if output is not None:
            try:
                json.dumps(output, allow_nan=False)
            except (TypeError, ValueError):
                return None
            output = cls._json_safe(output)
        spill_ref = cls._text(record.get("spill_ref"), limit=_MAX_TEXT)
        safe = {
            "version": TOOL_TRACE_SCHEMA_VERSION,
            "call_id": call_id,
            "correlation_id": correlation_id,
            "status": status,
            "state": state,
            "updated_at": updated_at,
        }
        if error_code:
            safe["error_code"] = error_code
        if output is not None:
            safe["output"] = output
        if spill_ref:
            safe["spill_ref"] = spill_ref
        return safe

    @classmethod
    def restore(cls, value: Any) -> list[dict[str, Any]]:
        if (
            not isinstance(value, Mapping)
            or value.get("version") != TOOL_TRACE_SCHEMA_VERSION
        ):
            return []
        by_call: dict[str, dict[str, Any]] = {}
        for raw in value.get("records", []):
            record = cls.validate_record(raw)
            if record is not None:
                cls.merge(by_call, record)
        return [by_call[call_id] for call_id in sorted(by_call)]

    @staticmethod
    def merge(records: dict[str, dict[str, Any]], record: dict[str, Any]) -> None:
        """Keep one terminal record per call using timestamp then JSON ordering."""
        previous = records.get(record["call_id"])
        if previous is None:
            records[record["call_id"]] = record
            return
        previous_key = (
            previous["updated_at"],
            json.dumps(previous, sort_keys=True, separators=(",", ":")),
        )
        record_key = (
            record["updated_at"],
            json.dumps(record, sort_keys=True, separators=(",", ":")),
        )
        if record_key >= previous_key:
            records[record["call_id"]] = record

    @classmethod
    def session_field(cls, records: Any) -> dict[str, Any]:
        indexed: dict[str, dict[str, Any]] = {}
        for raw in records if isinstance(records, list) else []:
            record = cls.validate_record(raw)
            if record is not None:
                cls.merge(indexed, record)
        return {
            "version": TOOL_TRACE_SCHEMA_VERSION,
            "records": [indexed[key] for key in sorted(indexed)],
        }

    @classmethod
    def context_envelope(cls, record: Any) -> dict[str, str] | None:
        safe = cls.validate_record(record)
        if safe is None:
            return None
        text = f"Tool {safe['call_id']} {safe['status']}"
        if safe.get("error_code"):
            text += f" ({safe['error_code']})"
        if safe.get("spill_ref"):
            text += f"\nSaved tool output: {safe['spill_ref']}"
        elif safe.get("output") is not None:
            text += (
                "\nResult: "
                + json.dumps(safe["output"], ensure_ascii=False, separators=(",", ":"))[
                    :_MAX_TEXT
                ]
            )
        return {"role": "assistant", "content": text}

    @classmethod
    def response_summary(cls, records: Any) -> str:
        valid = [
            cls.validate_record(record)
            for record in records
            if isinstance(records, list)
        ]
        valid = [record for record in valid if record is not None]
        if not valid:
            return ""
        parts = []
        for record in valid:
            label = f"{record['call_id']}: {record['status']}"
            if record.get("error_code"):
                label += f" ({record['error_code']})"
            parts.append(label)
        return "Tool status: " + "; ".join(parts)


__all__ = ["TOOL_TRACE_SCHEMA_VERSION", "ToolTracePersistence"]
