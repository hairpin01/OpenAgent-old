# SPDX-License-Identifier: MIT
"""Pure, sandbox-safe declarations and clients for v2 plugins.

This module deliberately has no access to the application, plugin loader, or
legacy plugin classes.  A plugin can declare data and submit narrow JSON
capability requests; authority remains in the parent broker.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from .ToolKernel import (
    ConfirmationRequirement,
    ConcurrencyClass,
    IdempotencyClass,
    MigrationDisposition,
    TOOL_API_VERSION,
    TOOL_SCHEMA_VERSION,
    ToolSpec,
    normalize_tool_name,
    validate_schema,
)
from .ToolCompatibility import TOOL_COMPATIBILITY_MATRIX, _REJECTED_LEGACY_ALIASES


PLUGIN_MANIFEST_VERSION = "2"
PLUGIN_SDK_API_VERSION = "2"
_PLUGIN_ID_RE = re.compile(r"^[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)+$")
_ENTRYPOINT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
_CAPABILITY_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_KEY_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


class PluginManifestError(ValueError):
    """A declaration cannot be represented by the isolated v2 SDK."""


class LegacyPluginMigrationError(PluginManifestError):
    """Static legacy declarations are ambiguous or unsafe to migrate."""


class CapabilityFamily(str, Enum):
    TELEGRAM = "telegram"
    WORKSPACE_FS = "workspace-fs"
    PROCESS = "process"
    HTTPS_FETCH = "https-fetch"
    SCHEDULING = "scheduling"
    CONFIGURATION = "configuration"


def _json(value: Any, path: tuple[str | int, ...] = ()) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise PluginManifestError(f"non-finite JSON number at {path!r}")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise PluginManifestError(f"JSON object key at {path!r} must be a string")
            frozen[key] = _json(nested, path + (key,))
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_json(item, path + (index,)) for index, item in enumerate(value))
    raise PluginManifestError(f"non-JSON value at {path!r}")


def thaw_json(value: Any) -> Any:
    """Return an ordinary JSON-compatible copy of a frozen declaration."""

    if isinstance(value, Mapping):
        return {key: thaw_json(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def _required(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PluginManifestError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class PluginToolDeclaration:
    """One immutable v2 plugin tool declaration, independent of a handler."""

    canonical_id: str
    aliases: tuple[str, ...] = ()
    input_schema: Mapping[str, Any] = field(default_factory=lambda: {"type": "object"})
    output_schema: Mapping[str, Any] = field(default_factory=lambda: {"type": "object"})
    capabilities: frozenset[str] = field(default_factory=frozenset)
    description: str = ""
    confirmation: ConfirmationRequirement = ConfirmationRequirement.NONE
    concurrency: ConcurrencyClass = ConcurrencyClass.SERIAL
    idempotency: IdempotencyClass = IdempotencyClass.IDEMPOTENT
    migration_disposition: MigrationDisposition = MigrationDisposition.MIGRATE

    def __post_init__(self) -> None:
        canonical_id = normalize_tool_name(self.canonical_id, canonical=True)
        aliases = tuple(normalize_tool_name(alias) for alias in self.aliases)
        if canonical_id in aliases or len(aliases) != len(set(aliases)):
            raise PluginManifestError("tool aliases must be unique and exclude the canonical ID")
        capabilities = frozenset(str(item).strip().lower() for item in self.capabilities)
        if not capabilities or any(not _CAPABILITY_RE.fullmatch(item) for item in capabilities):
            raise PluginManifestError("tools require declared, normalized capabilities")
        object.__setattr__(self, "canonical_id", canonical_id)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "input_schema", validate_schema(self.input_schema))
        object.__setattr__(self, "output_schema", validate_schema(self.output_schema))
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "description", self.description.strip() if isinstance(self.description, str) else "")
        object.__setattr__(self, "confirmation", ConfirmationRequirement(self.confirmation))
        object.__setattr__(self, "concurrency", ConcurrencyClass(self.concurrency))
        object.__setattr__(self, "idempotency", IdempotencyClass(self.idempotency))
        object.__setattr__(self, "migration_disposition", MigrationDisposition(self.migration_disposition))

    def to_tool_spec(self, *, source_module: str = "") -> ToolSpec:
        return ToolSpec(
            canonical_id=self.canonical_id, aliases=self.aliases,
            input_schema=self.input_schema, output_schema=self.output_schema,
            api_version=TOOL_API_VERSION, schema_version=TOOL_SCHEMA_VERSION,
            capabilities=self.capabilities, confirmation=self.confirmation,
            concurrency=self.concurrency, idempotency=self.idempotency,
            migration_disposition=self.migration_disposition,
            description=self.description, source_family="plugin-v2", source_module=source_module,
        )


@dataclass(frozen=True)
class PluginManifest:
    """The complete immutable, versioned declaration accepted by a v2 host."""

    plugin_id: str
    version: str
    api_version: str
    entrypoint: str
    tools: tuple[PluginToolDeclaration, ...]
    capabilities: frozenset[str]
    manifest_version: str = PLUGIN_MANIFEST_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        plugin_id = _required(self.plugin_id, "plugin_id").lower()
        if not _PLUGIN_ID_RE.fullmatch(plugin_id):
            raise PluginManifestError("plugin_id must be a normalized dotted identifier")
        if self.manifest_version != PLUGIN_MANIFEST_VERSION or self.api_version != PLUGIN_SDK_API_VERSION:
            raise PluginManifestError("unsupported plugin manifest or SDK API version")
        entrypoint = _required(self.entrypoint, "entrypoint")
        if not _ENTRYPOINT_RE.fullmatch(entrypoint):
            raise PluginManifestError("entrypoint must be a dotted Python symbol name")
        if not isinstance(self.tools, tuple) or not self.tools:
            raise PluginManifestError("tools must be a non-empty tuple of declarations")
        if any(not isinstance(tool, PluginToolDeclaration) for tool in self.tools):
            raise PluginManifestError("tools must contain PluginToolDeclaration values only")
        capabilities = frozenset(str(item).strip().lower() for item in self.capabilities)
        known = frozenset(item.value for item in CapabilityFamily)
        if not capabilities or not capabilities.issubset(known):
            raise PluginManifestError("manifest declares an unknown capability")
        tool_ids = [tool.canonical_id for tool in self.tools]
        aliases = [alias for tool in self.tools for alias in tool.aliases]
        if len(tool_ids) != len(set(tool_ids)) or len(aliases) != len(set(aliases)):
            raise PluginManifestError("duplicate tool IDs or aliases are not allowed")
        if set(tool_ids).intersection(aliases):
            raise PluginManifestError("an alias cannot collide with a tool ID")
        if not set().union(*(tool.capabilities for tool in self.tools)).issubset(capabilities):
            raise PluginManifestError("tool declares a capability absent from its manifest")
        metadata = _json(self.metadata)
        if not isinstance(metadata, Mapping):
            raise PluginManifestError("metadata must be a JSON object")
        object.__setattr__(self, "plugin_id", plugin_id)
        object.__setattr__(self, "version", _required(self.version, "version"))
        object.__setattr__(self, "entrypoint", entrypoint)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "metadata", metadata)


def manifest_from_legacy_declarations(
    declarations: Mapping[str, Any], *, plugin_id: str, entrypoint: str, version: str = "0.0.0",
    source_module: str | None = None, compatibility_matrix: Sequence[Any] = TOOL_COMPATIBILITY_MATRIX,
) -> PluginManifest:
    """Convert explicit static legacy data; never imports or instantiates a plugin."""

    allowed = {"metadata", "tool_registry", "tool_map", "tool_docs", "tool_schemas", "dangerous_tools"}
    unknown = set(declarations) - allowed
    if unknown:
        raise LegacyPluginMigrationError(f"unknown legacy declaration fields: {sorted(unknown)!r}")
    registry = declarations.get("tool_registry", ())
    tool_map = declarations.get("tool_map", {})
    metadata = declarations.get("metadata", {})
    docs = declarations.get("tool_docs", {})
    schemas = declarations.get("tool_schemas", {})
    dangerous = frozenset(declarations.get("dangerous_tools", ()))
    if isinstance(registry, str) or not isinstance(registry, (tuple, list)):
        raise LegacyPluginMigrationError("tool_registry must be a literal sequence")
    if not isinstance(metadata, Mapping) or not isinstance(tool_map, Mapping) or not isinstance(docs, Mapping) or not isinstance(schemas, Mapping):
        raise LegacyPluginMigrationError("legacy maps must be literal objects")
    canonical = tuple(normalize_tool_name(item, canonical=True) for item in registry)
    if len(canonical) != len(set(canonical)):
        raise LegacyPluginMigrationError("legacy tool_registry has duplicate IDs")
    module = source_module or plugin_id.rsplit(".", 1)[-1]
    entries = tuple(entry for entry in compatibility_matrix if getattr(entry, "source_module", None) == module)
    frozen = {entry.canonical_id: entry for entry in entries}
    if set(canonical) != set(frozen):
        raise LegacyPluginMigrationError("legacy canonical tools do not match committed compatibility data")
    owners = {
        name: entry.canonical_id
        for entry in entries
        for name in (entry.canonical_id, *entry.aliases)
    }
    aliases_by_owner: dict[str, list[str]] = {name: [] for name in canonical}
    for alias, handler in tool_map.items():
        normalized_alias = normalize_tool_name(alias)
        if not isinstance(handler, str) or not handler.strip():
            raise LegacyPluginMigrationError(f"legacy handler for {normalized_alias} is missing")
        if normalized_alias in _REJECTED_LEGACY_ALIASES:
            raise LegacyPluginMigrationError(f"legacy alias is explicitly rejected: {normalized_alias}")
        owner = owners.get(normalized_alias)
        if owner is None:
            raise LegacyPluginMigrationError(f"legacy aliases require migration: {normalized_alias}")
        canonical_handler = tool_map.get(owner)
        if canonical_handler != handler:
            raise LegacyPluginMigrationError(f"legacy alias handler does not match frozen owner: {normalized_alias}")
        if normalized_alias != owner:
            aliases_by_owner[owner].append(normalized_alias)
    tools: list[PluginToolDeclaration] = []
    capability_set: set[str] = set()
    for name in canonical:
        schema = schemas.get(name, {"type": "object"})
        if not isinstance(schema, Mapping):
            raise LegacyPluginMigrationError(f"legacy schema for {name} is not an object")
        doc = docs.get(name, {})
        description = doc.get("description", "") if isinstance(doc, Mapping) else ""
        capability = _legacy_capability(name)
        capability_set.add(capability)
        tools.append(PluginToolDeclaration(
            canonical_id=name, aliases=tuple(aliases_by_owner[name]), input_schema=schema, output_schema={"type": "object"},
            capabilities=frozenset({capability}), description=str(description),
            confirmation=ConfirmationRequirement.REQUIRED if name in dangerous else ConfirmationRequirement.NONE,
        ))
    return PluginManifest(
        plugin_id, version, PLUGIN_SDK_API_VERSION, entrypoint, tuple(tools),
        frozenset(capability_set), metadata=metadata,
    )


def manifest_from_legacy_source(
    source: str | Path, *, plugin_id: str, entrypoint: str, version: str = "0.0.0"
) -> PluginManifest:
    """AST-only legacy conversion for source text or a source path."""

    text = Path(source).read_text(encoding="utf-8") if isinstance(source, Path) else source
    if not isinstance(text, str):
        raise TypeError("legacy source must be text or a Path")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise LegacyPluginMigrationError("legacy plugin source is invalid") from exc
    values: dict[str, Any] = {}
    wanted = {"metadata", "tool_registry", "tool_map", "tool_docs", "tool_schemas", "dangerous_tools"}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        for target in targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                try:
                    values[target.id] = ast.literal_eval(value)
                except (ValueError, TypeError) as exc:
                    raise LegacyPluginMigrationError(f"{target.id} must be a literal declaration") from exc
    return manifest_from_legacy_declarations(values, plugin_id=plugin_id, entrypoint=entrypoint, version=version)


def _legacy_capability(canonical_id: str) -> str:
    prefix = canonical_id.split(".", 1)[0]
    return {
        "terminal": CapabilityFamily.PROCESS.value,
        "filesystem": CapabilityFamily.WORKSPACE_FS.value,
        "download": CapabilityFamily.HTTPS_FETCH.value,
        "task": CapabilityFamily.SCHEDULING.value,
        "config": CapabilityFamily.CONFIGURATION.value,
        "chat": CapabilityFamily.TELEGRAM.value,
    }.get(prefix, CapabilityFamily.CONFIGURATION.value)


@dataclass(frozen=True)
class CapabilityCallContext:
    """Non-secret identity supplied by the host for one plugin tool invocation."""

    host_request_id: str
    call_id: str
    canonical_tool_id: str
    actor_scope: str
    grant_id: str

    def __post_init__(self) -> None:
        for name in ("host_request_id", "call_id", "actor_scope", "grant_id"):
            object.__setattr__(self, name, _required(getattr(self, name), name))
        object.__setattr__(self, "canonical_tool_id", normalize_tool_name(self.canonical_tool_id, canonical=True))


class CapabilityTransport(Protocol):
    def request(self, frame: Mapping[str, Any]) -> Mapping[str, Any]: ...


class CapabilityClient:
    """Sandbox-side JSON-only client; it cannot expose parent ambient objects."""

    def __init__(self, context: CapabilityCallContext, transport: CapabilityTransport) -> None:
        self._context = context
        self._transport = transport

    def request(self, capability: CapabilityFamily, operation: str, payload: Mapping[str, Any], request_id: str) -> Mapping[str, Any]:
        if not isinstance(payload, Mapping):
            raise TypeError("capability payload must be a JSON object")
        frame = {
            "version": PLUGIN_SDK_API_VERSION, "kind": "capability-request",
            "host_request_id": self._context.host_request_id, "call_id": self._context.call_id,
            "canonical_tool_id": self._context.canonical_tool_id, "actor_scope": self._context.actor_scope,
            "grant_id": self._context.grant_id, "capability": capability.value,
            "operation": _required(operation, "operation"), "capability_request_id": _required(request_id, "request_id"),
            "payload": thaw_json(_json(payload)),
        }
        response = self._transport.request(frame)
        if not isinstance(response, Mapping):
            raise PluginManifestError("capability transport returned a non-JSON response")
        return MappingProxyType(dict(_json(response)))

    def telegram(self, operation: str, data: Mapping[str, Any], request_id: str) -> Mapping[str, Any]:
        return self.request(CapabilityFamily.TELEGRAM, operation, {"data": data}, request_id)

    def filesystem(self, operation: str, path: str, request_id: str, **data: Any) -> Mapping[str, Any]:
        return self.request(CapabilityFamily.WORKSPACE_FS, operation, {"path": path, **data}, request_id)

    def process(self, argv: Sequence[str], request_id: str, **data: Any) -> Mapping[str, Any]:
        return self.request(CapabilityFamily.PROCESS, "run", {"argv": list(argv), **data}, request_id)

    def fetch(self, url: str, request_id: str, **data: Any) -> Mapping[str, Any]:
        return self.request(CapabilityFamily.HTTPS_FETCH, "fetch", {"url": url, **data}, request_id)

    def schedule(self, child_call: Mapping[str, Any], request_id: str) -> Mapping[str, Any]:
        return self.request(CapabilityFamily.SCHEDULING, "schedule", {"child_call": child_call}, request_id)

    def config(self, operation: str, key: str, request_id: str, **data: Any) -> Mapping[str, Any]:
        return self.request(CapabilityFamily.CONFIGURATION, operation, {"key": key, **data}, request_id)


__all__ = [name for name in globals() if name.startswith(("PLUGIN_", "Capability", "Plugin", "Legacy", "manifest_", "thaw_"))]
