# SPDX-License-Identifier: MIT
"""Strict, pure model-facing conversion into immutable v2 tool calls.

This module is intentionally independent of the legacy runtime and executor.
It parses only declared wire formats, resolves names through ``ToolRegistry``,
and returns data for a caller to execute later.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from .ToolKernel import ToolCall, ToolContext, ToolKernelError, ToolRegistry, ToolResult, ToolResultStatus


class ModelToolFormat(str, Enum):
    XML = "xml"
    FENCED_JSON = "fenced_json"
    RAW_JSON = "raw_json"
    HARMONY = "harmony"


class ModelToolErrorCode(str, Enum):
    INPUT_TOO_LARGE = "input_too_large"
    BLOCK_TOO_LARGE = "block_too_large"
    BODY_TOO_LARGE = "body_too_large"
    NESTING_TOO_DEEP = "nesting_too_deep"
    MALFORMED_BLOCK = "malformed_block"
    MALFORMED_ATTRIBUTES = "malformed_attributes"
    MALFORMED_JSON = "malformed_json"
    AMBIGUOUS_PAYLOAD = "ambiguous_payload"
    DUPLICATE_CALL_ID = "duplicate_call_id"
    UNKNOWN_TOOL = "unknown_tool"
    UNKNOWN_FIELD = "unknown_field"
    INVALID_ARGUMENT = "invalid_argument"
    VERSION_MISMATCH = "version_mismatch"


class ModelTurnKind(str, Enum):
    TOOLS = "tools"
    FINAL = "final"
    INTERMEDIATE = "intermediate"
    INVALID = "invalid"


@dataclass(frozen=True)
class ModelToolError:
    code: ModelToolErrorCode
    offset: int
    format: ModelToolFormat | None


@dataclass(frozen=True)
class ParsedToolCall:
    """A format-neutral call after syntax parsing but before registry creation."""

    requested_name: str
    arguments: Mapping[str, Any]
    api_version: str | None
    schema_version: str | None
    offset: int
    format: ModelToolFormat

    def __post_init__(self) -> None:
        if not isinstance(self.arguments, Mapping):
            raise TypeError("arguments must be an object")
        object.__setattr__(self, "arguments", _freeze_json(self.arguments))


@dataclass(frozen=True)
class ModelBoundaryOutput:
    kind: ModelTurnKind
    calls: tuple[ToolCall, ...] = ()
    parsed_calls: tuple[ParsedToolCall, ...] = ()
    errors: tuple[ModelToolError, ...] = ()
    final_answer: str = ""
    intermediate: str = ""

    def __post_init__(self) -> None:
        for field_name, item_type in (("calls", ToolCall), ("parsed_calls", ParsedToolCall), ("errors", ModelToolError)):
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or any(not isinstance(item, item_type) for item in values):
                raise TypeError(f"{field_name} must be an immutable tuple of {item_type.__name__}")


@dataclass(frozen=True)
class ModelBoundaryLimits:
    max_input_chars: int = 32_000
    max_block_chars: int = 12_000
    max_body_chars: int = 8_000
    max_argument_chars: int = 8_000
    max_depth: int = 12
    max_calls: int = 16
    max_docs_results: int = 8
    max_rendered_chars: int = 4_000

    def __post_init__(self) -> None:
        caps = {
            "max_input_chars": 1_000_000,
            "max_block_chars": 1_000_000,
            "max_body_chars": 1_000_000,
            "max_argument_chars": 1_000_000,
            "max_depth": 100,
            "max_calls": 1_000,
            "max_docs_results": 1_000,
            "max_rendered_chars": 1_000_000,
        }
        for name, cap in caps.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= cap:
                raise ValueError(f"{name} must be a positive integer no larger than {cap}")


_FENCE_OPENER_RE = re.compile(r"```(?P<label>tool_call|json)\b[^\r\n]*\r?\n", re.I)
_XML_RE = re.compile(
    r"<(?P<name>[a-z][a-z0-9_.-]*)(?P<attrs>[^<>]*?)(?:(?P<self>/)>|>(?P<body>.*?)</(?P=name)\s*>)",
    re.I | re.S,
)
_HARMONY_RE = re.compile(
    r"(?P<header>.*?)<\|message\|>(?P<body>.*?)<\|call\|>", re.S
)
_XML_MARKER_RE = re.compile(r"<(?P<close>/)?(?P<name>[a-z][a-z0-9_.-]*)\b", re.I)
_RAW_CALL_SHAPE_RE = re.compile(r"\{\s*\"(?:tool|name)\"\s*:\s*.*?\"(?:args|arguments)\"\s*:", re.S)
_FINAL_FENCE_RE = re.compile(r"```final(?:_answer)?[ \t]*\r?\n(.*?)```", re.I | re.S)
_FINAL_TAG_RE = re.compile(r"<final(?:_answer)?>(.*?)</final(?:_answer)?>", re.I | re.S)
_ATTR_RE = re.compile(r"\s*([a-zA-Z_][\w.-]*)\s*=\s*(['\"])(.*?)\2", re.S)
_SECRET_KEY_RE = re.compile(r"(?:pass(?:word)?|secret|token|api[_-]?key|authorization|cookie|credential)", re.I)
_BODY_FIELDS = ("body", "content", "text", "message", "command", "query", "prompt", "note")


def _freeze_json(value: Any) -> Any:
    """Return a recursive immutable JSON value without accepting Python objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise TypeError("numbers must be finite JSON values")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("object keys must be strings")
            frozen[key] = _freeze_json(nested)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(nested) for nested in value)
    raise TypeError(f"unsupported JSON value type {type(value).__name__}")


class ToolModelBoundary:
    """Pure parser and model-document view over one immutable v2 registry."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        call_id_factory: Callable[[], str],
        context: ToolContext,
        limits: ModelBoundaryLimits = ModelBoundaryLimits(),
    ) -> None:
        if not isinstance(registry, ToolRegistry):
            raise TypeError("registry must be a ToolRegistry")
        if not callable(call_id_factory):
            raise TypeError("call_id_factory must be callable")
        if not isinstance(context, ToolContext):
            raise TypeError("context must be a ToolContext")
        self._registry = registry
        self._call_id_factory = call_id_factory
        self._context = context
        self._limits = limits

    def parse(self, value: Any) -> ModelBoundaryOutput:
        text = value if isinstance(value, str) else ""
        if len(text) > self._limits.max_input_chars:
            return self._invalid(ModelToolErrorCode.INPUT_TOO_LARGE, 0, None)
        blocks = self._blocks(text)
        if isinstance(blocks, ModelToolError):
            return ModelBoundaryOutput(ModelTurnKind.INVALID, errors=(blocks,))
        if blocks:
            parsed: list[ParsedToolCall] = []
            for format_, offset, body, recipient in blocks:
                parsed_or_error = self._parse_block(format_, offset, body, recipient)
                if isinstance(parsed_or_error, ModelToolError):
                    return ModelBoundaryOutput(ModelTurnKind.INVALID, errors=(parsed_or_error,))
                parsed.extend(parsed_or_error)
            if len(parsed) > self._limits.max_calls:
                return self._invalid(ModelToolErrorCode.BLOCK_TOO_LARGE, 0, None)
            calls_or_error = self._create_calls(tuple(sorted(parsed, key=lambda item: item.offset)))
            if isinstance(calls_or_error, ModelToolError):
                return ModelBoundaryOutput(ModelTurnKind.INVALID, errors=(calls_or_error,))
            return ModelBoundaryOutput(ModelTurnKind.TOOLS, calls_or_error, tuple(parsed))
        final = self.explicit_final(text)
        if final is not None:
            return ModelBoundaryOutput(ModelTurnKind.FINAL, final_answer=final)
        return ModelBoundaryOutput(ModelTurnKind.INTERMEDIATE, intermediate=self._intermediate(text))

    def explicit_final(self, text: str) -> str | None:
        """Return only a sole explicit final marker, never embedded tool content."""
        if self._blocks(text):
            return None
        for pattern in (_FINAL_FENCE_RE, _FINAL_TAG_RE):
            match = pattern.search(text)
            if match:
                final = match.group(1).strip()
                return final or None
        return None

    def tool_docs(self) -> tuple[Mapping[str, Any], ...]:
        """Return deterministic, model-safe v2 specifications without internals."""
        return tuple(
            MappingProxyType(
                {
                    "id": spec.canonical_id,
                    "aliases": spec.aliases,
                    "description": spec.description,
                    "input_schema": self._json_value(spec.input_schema),
                    "api_version": spec.api_version,
                    "schema_version": spec.schema_version,
                    "capabilities": tuple(sorted(spec.capabilities)),
                    "confirmation": spec.confirmation.value,
                }
            )
            for spec in self._registry.specs()
        )

    def search_tool_docs(self, query: str, *, limit: int | None = None) -> tuple[Mapping[str, Any], ...]:
        maximum = min(self._limits.max_docs_results, max(1, limit or self._limits.max_docs_results))
        terms = tuple(dict.fromkeys(re.findall(r"[\w.-]+", str(query).lower())))
        if not terms:
            return ()
        ranked: list[tuple[int, Mapping[str, Any]]] = []
        for doc in self.tool_docs():
            text = json.dumps(doc, sort_keys=True, default=str).lower()
            score = sum(term in text for term in terms)
            if score:
                ranked.append((-score, doc))
        ranked.sort(key=lambda item: (item[0], str(item[1]["id"])))
        return tuple(doc for _score, doc in ranked[:maximum])

    def render_result(
        self,
        result: ToolResult,
        *,
        spill_reference: Callable[[ToolResult, str], str] | None = None,
    ) -> str:
        """Render a bounded redacted envelope; spilling is caller-owned."""
        if not isinstance(result, ToolResult):
            raise TypeError("result must be a ToolResult")
        envelope: dict[str, Any] = {"call_id": result.call_id, "status": result.status.value}
        if result.status is ToolResultStatus.SUCCESS:
            envelope["output"] = self._redact(self._json_value(result.output))
        else:
            error = result.error
            envelope["error_code"] = getattr(getattr(error, "code", None), "value", None) or "tool_error"
            envelope["retryable"] = result.retryable
        rendered = json.dumps(envelope, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        if len(rendered) <= self._limits.max_rendered_chars:
            return rendered
        if spill_reference is None:
            raise ValueError("oversized model result requires spill_reference")
        reference = spill_reference(result, rendered)
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("spill_reference must return a non-empty string")
        return json.dumps(
            {"call_id": result.call_id, "status": result.status.value, "spill_ref": reference.strip()},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _blocks(self, text: str) -> list[tuple[ModelToolFormat, int, str, str | None]] | ModelToolError:
        blocks: list[tuple[ModelToolFormat, int, str, str | None]] = []
        fence_spans: list[tuple[int, int]] = []
        for opener in _FENCE_OPENER_RE.finditer(text):
            close = text.find("```", opener.end())
            body = text[opener.end():] if close < 0 else text[opener.end():close]
            label = opener.group("label").lower()
            is_call = label == "tool_call" or bool(_RAW_CALL_SHAPE_RE.search(body))
            if close < 0 and is_call:
                return self._error(ModelToolErrorCode.MALFORMED_BLOCK, opener.start(), ModelToolFormat.FENCED_JSON)
            if is_call:
                fence_spans.append((opener.start(), close + 3))
                blocks.append((ModelToolFormat.FENCED_JSON, opener.start(), body, None))
        valid_fence_starts = {start for start, _end in fence_spans}
        for marker in re.finditer(r"```tool_call\b", text, re.I):
            if marker.start() not in valid_fence_starts:
                return self._error(ModelToolErrorCode.MALFORMED_BLOCK, marker.start(), ModelToolFormat.FENCED_JSON)
        for marker in re.finditer(r"```json\b", text, re.I):
            close = text.find("```", marker.end())
            segment = text[marker.end():] if close < 0 else text[marker.end():close]
            if _RAW_CALL_SHAPE_RE.search(segment) and marker.start() not in valid_fence_starts:
                return self._error(ModelToolErrorCode.MALFORMED_BLOCK, marker.start(), ModelToolFormat.FENCED_JSON)

        harmony_spans: list[tuple[int, int]] = []
        harmony_intent = "<|message|>" in text or "<|call|>" in text or bool(re.search(r"\bto=tool\.", text, re.I))
        for match in _HARMONY_RE.finditer(text):
            if "to=" not in match.group("header"):
                continue
            recipient = self._recipient(match.group("header"))
            if recipient is None:
                return self._error(ModelToolErrorCode.MALFORMED_BLOCK, match.start(), ModelToolFormat.HARMONY)
            blocks.append((ModelToolFormat.HARMONY, match.start(), match.group("body"), recipient))
            harmony_spans.append((match.start(), match.end()))
        if harmony_intent and not harmony_spans:
            return self._error(ModelToolErrorCode.MALFORMED_BLOCK, 0, ModelToolFormat.HARMONY)

        occupied = tuple((*fence_spans, *harmony_spans))
        xml_spans: list[tuple[int, int]] = []
        for match in _XML_RE.finditer(text):
            name = match.group("name").lower()
            if name in {"final", "final_answer"} or any(start <= match.start() <= end for start, end in occupied):
                continue
            if "." in name or name == "tool_call" or self._resolves(name):
                body = match.group("body") or ""
                blocks.append((ModelToolFormat.XML, match.start(), match.group("attrs") + "\n" + body, name))
                xml_spans.append((match.start(), match.end()))
        for marker in _XML_MARKER_RE.finditer(text):
            name = marker.group("name").lower()
            if name in {"final", "final_answer"} or any(start <= marker.start() < end for start, end in occupied):
                continue
            if name == "tool_call" or "." in name or self._resolves(name):
                if not any(start <= marker.start() < end for start, end in xml_spans):
                    return self._error(ModelToolErrorCode.MALFORMED_BLOCK, marker.start(), ModelToolFormat.XML)
        stripped = text.strip()
        raw_offset = text.index(stripped) if stripped else 0
        if not blocks and stripped[:1] in "[{" and _RAW_CALL_SHAPE_RE.match(stripped):
            blocks.append((ModelToolFormat.RAW_JSON, raw_offset, stripped, None))
        raw_shape = _RAW_CALL_SHAPE_RE.search(text)
        has_raw_block = any(format_ is ModelToolFormat.RAW_JSON for format_, _offset, _body, _recipient in blocks)
        if raw_shape and not (has_raw_block or any(start <= raw_shape.start() < end for start, end in occupied)):
            return self._error(ModelToolErrorCode.MALFORMED_BLOCK, raw_shape.start(), ModelToolFormat.RAW_JSON)
        if len({format_ for format_, _offset, _body, _recipient in blocks}) > 1:
            return self._error(ModelToolErrorCode.AMBIGUOUS_PAYLOAD, 0, None)
        return sorted(blocks, key=lambda item: item[1])

    def _parse_block(self, format_: ModelToolFormat, offset: int, body: str, recipient: str | None) -> tuple[ParsedToolCall, ...] | ModelToolError:
        if len(body) > self._limits.max_block_chars:
            return self._error(ModelToolErrorCode.BLOCK_TOO_LARGE, offset, format_)
        if format_ is ModelToolFormat.XML:
            assert recipient is not None
            attrs, xml_body = body.split("\n", 1)
            return self._parse_xml(offset, recipient, attrs, xml_body)
        try:
            payload = json.loads(body)
        except (TypeError, ValueError):
            return self._error(ModelToolErrorCode.MALFORMED_JSON, offset, format_)
        if self._depth(payload) > self._limits.max_depth:
            return self._error(ModelToolErrorCode.NESTING_TOO_DEEP, offset, format_)
        payloads = payload if isinstance(payload, list) else [payload]
        if not payloads or any(not isinstance(item, dict) for item in payloads):
            return self._error(ModelToolErrorCode.MALFORMED_JSON, offset, format_)
        parsed: list[ParsedToolCall] = []
        for item in payloads:
            call = self._parse_json_item(item, offset, format_, recipient)
            if isinstance(call, ModelToolError):
                return call
            parsed.append(call)
        return tuple(parsed)

    def _parse_json_item(self, item: Mapping[str, Any], offset: int, format_: ModelToolFormat, recipient: str | None) -> ParsedToolCall | ModelToolError:
        allowed = {"tool", "name", "args", "arguments", "body", "api_version", "schema_version"}
        unknown = set(item) - allowed
        if unknown:
            return self._error(ModelToolErrorCode.UNKNOWN_FIELD, offset, format_)
        names = [item[key] for key in ("tool", "name") if key in item]
        if len(names) > 1 or (not names and recipient is None):
            return self._error(ModelToolErrorCode.AMBIGUOUS_PAYLOAD, offset, format_)
        if "args" in item and "arguments" in item:
            return self._error(ModelToolErrorCode.AMBIGUOUS_PAYLOAD, offset, format_)
        name = recipient if not names else names[0]
        if recipient is not None and names and str(name).strip().lower() != recipient:
            return self._error(ModelToolErrorCode.AMBIGUOUS_PAYLOAD, offset, format_)
        args = item.get("args", item.get("arguments", {}))
        if not isinstance(args, dict):
            return self._error(ModelToolErrorCode.INVALID_ARGUMENT, offset, format_)
        if "body" in item:
            converted = self._add_body(str(name), dict(args), item["body"], offset, format_)
            if isinstance(converted, ModelToolError):
                return converted
            args = converted
        return ParsedToolCall(str(name), args, self._optional_text(item.get("api_version")), self._optional_text(item.get("schema_version")), offset, format_)

    def _parse_xml(self, offset: int, name: str, raw_attrs: str, body: str) -> tuple[ParsedToolCall, ...] | ModelToolError:
        if len(body) > self._limits.max_body_chars:
            return self._error(ModelToolErrorCode.BODY_TOO_LARGE, offset, ModelToolFormat.XML)
        raw_attrs = raw_attrs.rstrip()
        if raw_attrs.endswith("/"):
            raw_attrs = raw_attrs[:-1]
        attrs: dict[str, str] = {}
        cursor = 0
        while cursor < len(raw_attrs):
            match = _ATTR_RE.match(raw_attrs, cursor)
            if match is None:
                if raw_attrs[cursor:].strip():
                    return self._error(ModelToolErrorCode.MALFORMED_ATTRIBUTES, offset + cursor, ModelToolFormat.XML)
                break
            key = match.group(1).lower()
            if key in attrs:
                return self._error(ModelToolErrorCode.DUPLICATE_CALL_ID, offset + cursor, ModelToolFormat.XML)
            attrs[key] = match.group(3)
            cursor = match.end()
        if name == "tool_call":
            names = [attrs.pop(key) for key in ("tool", "name") if key in attrs]
            if len(names) != 1:
                return self._error(ModelToolErrorCode.AMBIGUOUS_PAYLOAD, offset, ModelToolFormat.XML)
            name = names[0]
        api_version = attrs.pop("api_version", None)
        schema_version = attrs.pop("schema_version", None)
        converted = self._convert_xml_args(name, attrs, body, offset)
        if isinstance(converted, ModelToolError):
            return converted
        return (ParsedToolCall(name, converted, api_version, schema_version, offset, ModelToolFormat.XML),)

    def _convert_xml_args(self, name: str, attrs: Mapping[str, str], body: str, offset: int) -> Mapping[str, Any] | ModelToolError:
        try:
            spec = self._registry.resolve(name)
        except ToolKernelError:
            return self._error(ModelToolErrorCode.UNKNOWN_TOOL, offset, ModelToolFormat.XML)
        properties = spec.input_schema.get("properties", {})
        if any(key not in properties for key in attrs):
            return self._error(ModelToolErrorCode.UNKNOWN_FIELD, offset, ModelToolFormat.XML)
        arguments: dict[str, Any] = {}
        for key, value in attrs.items():
            if len(value) > self._limits.max_argument_chars:
                return self._error(ModelToolErrorCode.BODY_TOO_LARGE, offset, ModelToolFormat.XML)
            converted = self._coerce_legacy(value, properties[key])
            if converted is _INVALID:
                return self._error(ModelToolErrorCode.INVALID_ARGUMENT, offset, ModelToolFormat.XML)
            arguments[key] = converted
        if body.strip():
            added = self._add_body(name, arguments, body, offset, ModelToolFormat.XML)
            if isinstance(added, ModelToolError):
                return added
            arguments = added
        return arguments

    def _add_body(self, name: str, arguments: dict[str, Any], body: Any, offset: int, format_: ModelToolFormat) -> dict[str, Any] | ModelToolError:
        if not isinstance(body, str) or len(body) > self._limits.max_body_chars:
            return self._error(ModelToolErrorCode.BODY_TOO_LARGE, offset, format_)
        try:
            properties = self._registry.resolve(name).input_schema.get("properties", {})
        except ToolKernelError:
            return self._error(ModelToolErrorCode.UNKNOWN_TOOL, offset, format_)
        candidates = [field for field in _BODY_FIELDS if field in properties and field not in arguments]
        if len(candidates) != 1:
            return self._error(ModelToolErrorCode.AMBIGUOUS_PAYLOAD, offset, format_)
        converted = self._coerce_legacy(body, properties[candidates[0]])
        if converted is _INVALID:
            return self._error(ModelToolErrorCode.INVALID_ARGUMENT, offset, format_)
        arguments[candidates[0]] = converted
        return arguments

    def _create_calls(self, parsed: Sequence[ParsedToolCall]) -> tuple[ToolCall, ...] | ModelToolError:
        calls: list[ToolCall] = []
        seen: set[str] = set()
        for item in parsed:
            try:
                spec = self._registry.resolve(item.requested_name)
            except ToolKernelError:
                return self._error(ModelToolErrorCode.UNKNOWN_TOOL, item.offset, item.format)
            try:
                call_id = self._call_id_factory()
                if not isinstance(call_id, str) or not call_id.strip() or call_id in seen:
                    return self._error(ModelToolErrorCode.DUPLICATE_CALL_ID, item.offset, item.format)
                seen.add(call_id)
                calls.append(self._registry.create_call(
                    call_id=call_id,
                    requested_name=item.requested_name,
                    arguments=item.arguments,
                    context=self._context,
                    api_version=item.api_version or spec.api_version,
                    schema_version=item.schema_version or spec.schema_version,
                ))
            except ToolKernelError as error:
                code = ModelToolErrorCode.VERSION_MISMATCH if "version" in str(getattr(error, "code", "")) else ModelToolErrorCode.INVALID_ARGUMENT
                return self._error(code, item.offset, item.format)
        return tuple(calls)

    def _resolves(self, name: str) -> bool:
        try:
            self._registry.resolve(name)
            return True
        except ToolKernelError:
            return False

    @staticmethod
    def _recipient(header: str) -> str | None:
        match = re.search(r"(?:^|\s)to=([^\s<]+)", header)
        if match is None:
            return None
        recipient = match.group(1).strip(" '\"").lower()
        return recipient[5:] if recipient.startswith("tool.") else recipient

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        return value if isinstance(value, str) and value.strip() else None

    @staticmethod
    def _depth(value: Any) -> int:
        if isinstance(value, Mapping):
            return 1 + max((ToolModelBoundary._depth(item) for item in value.values()), default=0)
        if isinstance(value, list):
            return 1 + max((ToolModelBoundary._depth(item) for item in value), default=0)
        return 0

    @staticmethod
    def _coerce_legacy(value: str, schema: Mapping[str, Any]) -> Any:
        type_ = schema.get("type")
        if type_ == "string":
            return value
        if type_ == "boolean":
            return {"true": True, "false": False}.get(value.strip().lower(), _INVALID)
        if type_ == "null":
            return None if value.strip().lower() == "null" else _INVALID
        if type_ in {"integer", "number", "array", "object"}:
            try:
                parsed = json.loads(value)
            except ValueError:
                return _INVALID
            if type_ == "integer" and (isinstance(parsed, bool) or not isinstance(parsed, int)):
                return _INVALID
            if type_ == "number" and (isinstance(parsed, bool) or not isinstance(parsed, (int, float))):
                return _INVALID
            if type_ == "array" and not isinstance(parsed, list):
                return _INVALID
            if type_ == "object" and not isinstance(parsed, dict):
                return _INVALID
            return parsed
        return _INVALID

    @staticmethod
    def _json_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): ToolModelBoundary._json_value(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [ToolModelBoundary._json_value(item) for item in value]
        return value

    @staticmethod
    def _redact(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): "[REDACTED]" if _SECRET_KEY_RE.search(str(key)) else ToolModelBoundary._redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [ToolModelBoundary._redact(item) for item in value]
        return value

    @staticmethod
    def _intermediate(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()[:1200]

    @staticmethod
    def _error(code: ModelToolErrorCode, offset: int, format_: ModelToolFormat | None) -> ModelToolError:
        return ModelToolError(code, offset, format_)

    def _invalid(self, code: ModelToolErrorCode, offset: int, format_: ModelToolFormat | None) -> ModelBoundaryOutput:
        return ModelBoundaryOutput(ModelTurnKind.INVALID, errors=(self._error(code, offset, format_),))


_INVALID = object()


__all__ = [
    "ModelBoundaryLimits",
    "ModelBoundaryOutput",
    "ModelToolError",
    "ModelToolErrorCode",
    "ModelToolFormat",
    "ModelTurnKind",
    "ParsedToolCall",
    "ToolModelBoundary",
]
