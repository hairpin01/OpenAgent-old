# SPDX-License-Identifier: MIT
"""OpenAI-compatible materialization for the immutable v2 tool registry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from .ToolKernel import MigrationDisposition, ToolSpec

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_PATH_RE = re.compile(r"(?:^|[\s'\"])(?:/[\w.~-]+|[A-Za-z]:[\\/])")
_PATH_GROUPS = ("file.", "terminal.", "code.", "ast_grep.")
_NATIVE_ERROR_PREFIX = "[NATIVE_TOOL_CALL_ERROR] "


def json_materialize(value: Any) -> Any:
    """Return ordinary JSON containers without changing the frozen schema."""

    if isinstance(value, Mapping):
        return {str(key): json_materialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_materialize(item) for item in value]
    if isinstance(value, frozenset):
        return [json_materialize(item) for item in sorted(value, key=repr)]
    return value


def json_schema_text(schema: Mapping[str, Any]) -> str:
    """Serialize one frozen schema exactly as model-readable JSON."""

    return json.dumps(
        json_materialize(schema),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def native_function_name(canonical_id: str) -> str:
    """Make a stable OpenAI-safe name while retaining a readable canonical stem."""

    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", canonical_id).strip("_") or "tool"
    digest = hashlib.sha256(canonical_id.encode("utf-8")).hexdigest()[:12]
    name = f"{stem[:51]}_{digest}"
    if _SAFE_NAME_RE.fullmatch(name) is None:
        raise ValueError(f"generated native function name is invalid: {name!r}")
    return name


@dataclass(frozen=True)
class NativeToolCatalog:
    tools: tuple[dict[str, Any], ...]
    native_to_canonical: Mapping[str, str]
    canonical_to_native: Mapping[str, str]
    specs_by_canonical: Mapping[str, ToolSpec]


def _score(spec: ToolSpec, prompt: str, terms: tuple[str, ...]) -> int:
    name = spec.canonical_id.lower()
    documentation = json.dumps(
        {
            "description": spec.description,
            "schema": json_materialize(spec.input_schema),
        },
        ensure_ascii=False,
        sort_keys=True,
    ).lower()
    score = sum(20 for term in terms if term in documentation)
    score += sum(80 for term in terms if term in name)
    if name in prompt:
        score += 500
    if _PATH_RE.search(prompt) and name.startswith(_PATH_GROUPS):
        score += 1000
    return score


def build_native_tool_catalog(
    specs: Sequence[ToolSpec], prompt: str, *, cap: int = 128
) -> NativeToolCatalog:
    """Select a bounded deterministic subset of migratable canonical specs."""

    migratable = [
        spec
        for spec in specs
        if spec.migration_disposition is MigrationDisposition.MIGRATE
    ]
    terms = tuple(dict.fromkeys(re.findall(r"[\w.-]+", prompt.lower())))
    ranked = sorted(
        migratable,
        key=lambda spec: (-_score(spec, prompt.lower(), terms), spec.canonical_id),
    )
    selected = ranked[: max(1, min(128, int(cap)))]
    search = next(
        (spec for spec in migratable if spec.canonical_id == "utility.search_tool"),
        None,
    )
    if search is not None and search not in selected:
        selected[-1:] = [search]
    selected.sort(key=lambda spec: spec.canonical_id)

    native_to_canonical: dict[str, str] = {}
    canonical_to_native: dict[str, str] = {}
    tools: list[dict[str, Any]] = []
    for spec in selected:
        native_name = native_function_name(spec.canonical_id)
        previous = native_to_canonical.get(native_name)
        if previous is not None and previous != spec.canonical_id:
            raise ValueError(f"native function name collision: {native_name}")
        native_to_canonical[native_name] = spec.canonical_id
        canonical_to_native[spec.canonical_id] = native_name
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": native_name,
                    "description": spec.description or spec.canonical_id,
                    "parameters": json_materialize(spec.input_schema),
                },
            }
        )
    selected_specs = {spec.canonical_id: spec for spec in selected}
    return NativeToolCatalog(
        tuple(tools), native_to_canonical, canonical_to_native, selected_specs
    )


def _has_undeclared_fields(arguments: Mapping[str, Any], spec: ToolSpec) -> bool:
    schema = spec.input_schema
    properties = schema.get("properties", {})
    return schema.get("additionalProperties") is False and any(
        name not in properties for name in arguments
    )


def native_response_to_fences(message: Any, catalog: NativeToolCatalog) -> str:
    """Convert a valid OpenAI ``message.tool_calls`` list to legacy fences.

    Invalid calls return a marker consumed by the response loop, so no partial or
    unadvertised native request reaches the v2 executor.
    """

    if not isinstance(message, Mapping):
        return _NATIVE_ERROR_PREFIX + "response message is not an object"
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        return _NATIVE_ERROR_PREFIX + "response content must be text or null"
    calls = message.get("tool_calls")
    if calls is None:
        return content or ""
    if not isinstance(calls, list) or not calls:
        return _NATIVE_ERROR_PREFIX + "tool_calls must be a non-empty array"
    seen_ids: set[str] = set()
    payloads: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, Mapping):
            return _NATIVE_ERROR_PREFIX + "tool call is not an object"
        call_id = call.get("id")
        function = call.get("function")
        if not isinstance(call_id, str) or not call_id or call_id in seen_ids:
            return (
                _NATIVE_ERROR_PREFIX + "tool call IDs must be unique non-empty strings"
            )
        seen_ids.add(call_id)
        if not isinstance(function, Mapping) or not isinstance(
            function.get("name"), str
        ):
            return _NATIVE_ERROR_PREFIX + "tool call function is malformed"
        canonical = catalog.native_to_canonical.get(function["name"])
        if canonical is None:
            return _NATIVE_ERROR_PREFIX + "unknown or unadvertised native function"
        raw_arguments = function.get("arguments")
        if not isinstance(raw_arguments, str):
            return _NATIVE_ERROR_PREFIX + "tool arguments must be a JSON string"
        try:
            arguments = json.loads(raw_arguments)
        except (TypeError, ValueError):
            return _NATIVE_ERROR_PREFIX + "tool arguments are malformed JSON"
        if not isinstance(arguments, dict):
            return _NATIVE_ERROR_PREFIX + "tool arguments must decode to an object"
        if _has_undeclared_fields(arguments, catalog.specs_by_canonical[canonical]):
            return _NATIVE_ERROR_PREFIX + "tool arguments contain undeclared fields"
        payloads.append(
            {"tool": canonical, "args": arguments, "provider_call_id": call_id}
        )
    fences = "".join(
        "```tool_call\n"
        + json.dumps(payload, ensure_ascii=True, sort_keys=True)
        + "\n```\n"
        for payload in payloads
    )
    return (content or "") + ("\n" if content and fences else "") + fences.rstrip()


def native_tool_call_error(value: str) -> str | None:
    return (
        value[len(_NATIVE_ERROR_PREFIX) :]
        if value.startswith(_NATIVE_ERROR_PREFIX)
        else None
    )


__all__ = [
    "NativeToolCatalog",
    "build_native_tool_catalog",
    "json_materialize",
    "json_schema_text",
    "native_function_name",
    "native_response_to_fences",
    "native_tool_call_error",
]
