# SPDX-License-Identifier: MIT
"""Immutable, generation-safe state for statically admitted v2 plugins.

The registry stores declarations and opaque ownership identifiers only.  It
does not import plugin modules or retain plugin instances, callbacks, futures,
or task objects.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from enum import Enum
from math import isfinite
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any

_PLUGIN_ID_RE = re.compile(r"^[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)+$")
_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)*$")
_CAPABILITY_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_ENTRYPOINT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_FAILURE_CODE_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_PLUGIN_MANIFEST_VERSION = "2"
_PLUGIN_SDK_API_VERSION = "2"


class InstalledPluginStatus(str, Enum):
    """Lifecycle states for an installed plugin generation."""

    ADMITTED = "admitted"
    INSTALLED = "installed"
    ACTIVE = "active"
    UNLOADING = "unloading"
    FAILED = "failed"
    DISABLED = "disabled"
    REMOVED = "removed"


VALID_INSTALLED_PLUGIN_TRANSITIONS: Mapping[
    InstalledPluginStatus, frozenset[InstalledPluginStatus]
] = MappingProxyType(
    {
        InstalledPluginStatus.ADMITTED: frozenset(
            {
                InstalledPluginStatus.INSTALLED,
                InstalledPluginStatus.DISABLED,
                InstalledPluginStatus.FAILED,
                InstalledPluginStatus.REMOVED,
            }
        ),
        InstalledPluginStatus.INSTALLED: frozenset(
            {
                InstalledPluginStatus.ACTIVE,
                InstalledPluginStatus.DISABLED,
                InstalledPluginStatus.FAILED,
                InstalledPluginStatus.REMOVED,
            }
        ),
        InstalledPluginStatus.ACTIVE: frozenset(
            {InstalledPluginStatus.UNLOADING, InstalledPluginStatus.FAILED}
        ),
        InstalledPluginStatus.UNLOADING: frozenset(
            {
                InstalledPluginStatus.INSTALLED,
                InstalledPluginStatus.DISABLED,
                InstalledPluginStatus.FAILED,
            }
        ),
        InstalledPluginStatus.FAILED: frozenset(
            {
                InstalledPluginStatus.INSTALLED,
                InstalledPluginStatus.DISABLED,
                InstalledPluginStatus.REMOVED,
            }
        ),
        InstalledPluginStatus.DISABLED: frozenset(
            {InstalledPluginStatus.INSTALLED, InstalledPluginStatus.REMOVED}
        ),
        InstalledPluginStatus.REMOVED: frozenset(),
    }
)


class InstalledPluginRegistryErrorCode(str, Enum):
    """Stable error codes for registry callers and command adapters."""

    MISSING = "missing"
    COLLISION = "collision"
    STALE_GENERATION = "stale_generation"
    INVALID_TRANSITION = "invalid_transition"
    INVALID_RECORD = "invalid_record"


class InstalledPluginRegistryError(ValueError):
    """Base error containing a stable code and safe record context."""

    def __init__(
        self,
        code: InstalledPluginRegistryErrorCode,
        message: str,
        *,
        plugin_id: str | None = None,
        expected_generation: int | None = None,
        actual_generation: int | None = None,
    ) -> None:
        self.code = code
        self.plugin_id = plugin_id
        self.expected_generation = expected_generation
        self.actual_generation = actual_generation
        super().__init__(f"{code.value}: {message}")


class InstalledPluginMissingError(InstalledPluginRegistryError):
    """A requested record or opaque ownership identifier is absent."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(InstalledPluginRegistryErrorCode.MISSING, message, **context)


class InstalledPluginCollisionError(InstalledPluginRegistryError):
    """A plugin ID, source path, tool name, or ownership ID is already owned."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(InstalledPluginRegistryErrorCode.COLLISION, message, **context)


class InstalledPluginStaleGenerationError(InstalledPluginRegistryError):
    """A mutation targets a replaced lifecycle generation."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(
            InstalledPluginRegistryErrorCode.STALE_GENERATION, message, **context
        )


class InstalledPluginTransitionError(InstalledPluginRegistryError):
    """A valid record cannot make the requested lifecycle transition."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(
            InstalledPluginRegistryErrorCode.INVALID_TRANSITION, message, **context
        )


class InstalledPluginRecordError(InstalledPluginRegistryError):
    """A record cannot be represented safely by the registry."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(
            InstalledPluginRegistryErrorCode.INVALID_RECORD, message, **context
        )


def normalize_plugin_id(value: object) -> str:
    """Return the canonical v2 plugin ID spelling used by PluginSDK."""

    if not isinstance(value, str):
        raise InstalledPluginRecordError("plugin ID must be a string")
    plugin_id = value.strip().lower()
    if not _PLUGIN_ID_RE.fullmatch(plugin_id):
        raise InstalledPluginRecordError("plugin ID must be a dotted identifier")
    return plugin_id


def _normalize_tool_name(value: object, *, canonical: bool = False) -> str:
    if not isinstance(value, str):
        raise InstalledPluginRecordError("tool names must be strings")
    name = value.strip().lower()
    pattern = _PLUGIN_ID_RE if canonical else _TOOL_NAME_RE
    if not pattern.fullmatch(name):
        kind = "canonical tool ID" if canonical else "tool name or alias"
        raise InstalledPluginRecordError(f"invalid {kind}: {value!r}")
    return name


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InstalledPluginRecordError(f"{name} must be a non-empty string")
    return value.strip()


def _freeze_json(
    value: Any,
    path: tuple[str | int, ...] = (),
    active_containers: set[int] | None = None,
) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise InstalledPluginRecordError(f"non-finite JSON number at {path!r}")
        return value
    if isinstance(value, Mapping):
        return _freeze_json_mapping(value, path, active_containers)
    if isinstance(value, (list, tuple)):
        return _freeze_json_sequence(value, path, active_containers)
    raise InstalledPluginRecordError(f"non-JSON value at {path!r}")


def _freeze_json_mapping(
    value: Mapping[object, Any],
    path: tuple[str | int, ...],
    active_containers: set[int] | None,
) -> Mapping[str, Any]:
    active = active_containers if active_containers is not None else set()
    identity = id(value)
    if identity in active:
        raise InstalledPluginRecordError(f"cyclic JSON value at {path!r}")
    active.add(identity)
    try:
        frozen: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise InstalledPluginRecordError(
                    f"JSON object key at {path!r} must be a string"
                )
            frozen[key] = _freeze_json(nested, path + (key,), active)
        return MappingProxyType(frozen)
    finally:
        active.remove(identity)


def _freeze_json_sequence(
    value: list[Any] | tuple[Any, ...],
    path: tuple[str | int, ...],
    active_containers: set[int] | None,
) -> tuple[Any, ...]:
    active = active_containers if active_containers is not None else set()
    identity = id(value)
    if identity in active:
        raise InstalledPluginRecordError(f"cyclic JSON value at {path!r}")
    active.add(identity)
    try:
        return tuple(
            _freeze_json(item, path + (index,), active)
            for index, item in enumerate(value)
        )
    finally:
        active.remove(identity)


def _freeze_capabilities(values: Iterable[object], name: str) -> frozenset[str]:
    if isinstance(values, str):
        raise InstalledPluginRecordError(f"{name} must be an iterable of capabilities")
    try:
        normalized = frozenset(str(value).strip().lower() for value in values)
    except TypeError as exc:
        raise InstalledPluginRecordError(
            f"{name} must be an iterable of capabilities"
        ) from exc
    if not normalized or any(
        not _CAPABILITY_RE.fullmatch(value) for value in normalized
    ):
        raise InstalledPluginRecordError(
            f"{name} must contain normalized capability identifiers"
        )
    return normalized


def _freeze_identifiers(values: Iterable[object], name: str) -> frozenset[str]:
    if isinstance(values, str):
        raise InstalledPluginRecordError(f"{name} must be an iterable of identifiers")
    try:
        identifiers = frozenset(_required_text(value, name) for value in values)
    except TypeError as exc:
        raise InstalledPluginRecordError(
            f"{name} must be an iterable of identifiers"
        ) from exc
    return identifiers


def _normalize_source_path(value: object) -> str:
    if not isinstance(value, (str, Path)):
        raise InstalledPluginRecordError("source path must be a string or Path")
    path = Path(value)
    if not path.is_absolute():
        raise InstalledPluginRecordError("source path must be absolute")
    normalized = path.resolve()
    if path != normalized:
        raise InstalledPluginRecordError("source path must already be normalized")
    return str(normalized)


def _normalize_digest(value: object) -> str:
    if not isinstance(value, str):
        raise InstalledPluginRecordError("source digest must be a string")
    digest = value.strip().lower()
    if not _DIGEST_RE.fullmatch(digest):
        raise InstalledPluginRecordError(
            "source digest must be a 64-character SHA-256 hex digest"
        )
    return digest


@dataclass(frozen=True)
class InstalledPluginSource:
    """The immutable static source identity approved for a plugin."""

    path: str
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_source_path(self.path))
        object.__setattr__(self, "digest", _normalize_digest(self.digest))


@dataclass(frozen=True)
class InstalledPluginTool:
    """An immutable tool declaration retained without a handler object."""

    canonical_id: str
    aliases: tuple[str, ...] = ()
    capabilities: frozenset[str] = field(default_factory=frozenset)
    description: str = ""
    input_schema: Mapping[str, Any] = field(default_factory=lambda: {"type": "object"})
    output_schema: Mapping[str, Any] = field(default_factory=lambda: {"type": "object"})
    confirmation: str = "none"
    concurrency: str = "serial"
    idempotency: str = "idempotent"
    migration_disposition: str = "migrate"

    def __post_init__(self) -> None:
        canonical_id = _normalize_tool_name(self.canonical_id, canonical=True)
        if isinstance(self.aliases, str):
            raise InstalledPluginRecordError("tool aliases must be an iterable")
        try:
            aliases = tuple(_normalize_tool_name(alias) for alias in self.aliases)
        except TypeError as exc:
            raise InstalledPluginRecordError(
                "tool aliases must be an iterable"
            ) from exc
        if canonical_id in aliases or len(aliases) != len(set(aliases)):
            raise InstalledPluginRecordError(
                "tool aliases must be unique and exclude the canonical ID"
            )
        input_schema = _freeze_json(self.input_schema, ("input_schema",))
        output_schema = _freeze_json(self.output_schema, ("output_schema",))
        if not isinstance(input_schema, Mapping) or not isinstance(
            output_schema, Mapping
        ):
            raise InstalledPluginRecordError("tool schemas must be JSON objects")
        object.__setattr__(self, "canonical_id", canonical_id)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(
            self,
            "capabilities",
            _freeze_capabilities(self.capabilities, "tool capabilities"),
        )
        object.__setattr__(
            self,
            "description",
            self.description.strip() if isinstance(self.description, str) else "",
        )
        object.__setattr__(self, "input_schema", input_schema)
        object.__setattr__(self, "output_schema", output_schema)
        object.__setattr__(
            self, "confirmation", _required_text(self.confirmation, "confirmation")
        )
        object.__setattr__(
            self, "concurrency", _required_text(self.concurrency, "concurrency")
        )
        object.__setattr__(
            self, "idempotency", _required_text(self.idempotency, "idempotency")
        )
        object.__setattr__(
            self,
            "migration_disposition",
            _required_text(self.migration_disposition, "migration disposition"),
        )

    @property
    def names(self) -> tuple[str, ...]:
        """Return the canonical ID followed by all declared aliases."""

        return (self.canonical_id, *self.aliases)


@dataclass(frozen=True)
class InstalledPluginManifest:
    """Immutable snapshot of the declaration data needed outside the host."""

    manifest_version: str
    api_version: str
    version: str
    entrypoint: str
    tools: tuple[InstalledPluginTool, ...]
    capabilities: frozenset[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    display_name: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.tools, str):
            raise InstalledPluginRecordError("manifest tools must be an iterable")
        try:
            tools = tuple(self.tools)
        except TypeError as exc:
            raise InstalledPluginRecordError(
                "manifest tools must be an iterable"
            ) from exc
        if not tools or any(
            not isinstance(tool, InstalledPluginTool) for tool in tools
        ):
            raise InstalledPluginRecordError(
                "manifest tools must contain InstalledPluginTool values"
            )
        tool_ids = tuple(tool.canonical_id for tool in tools)
        aliases = tuple(alias for tool in tools for alias in tool.aliases)
        if len(tool_ids) != len(set(tool_ids)) or len(aliases) != len(set(aliases)):
            raise InstalledPluginRecordError(
                "duplicate tool IDs or aliases are not allowed"
            )
        if set(tool_ids).intersection(aliases):
            raise InstalledPluginRecordError("an alias cannot collide with a tool ID")
        capabilities = _freeze_capabilities(self.capabilities, "manifest capabilities")
        if (
            not set()
            .union(*(tool.capabilities for tool in tools))
            .issubset(capabilities)
        ):
            raise InstalledPluginRecordError(
                "tool capabilities must be declared by the manifest"
            )
        metadata = _freeze_json(self.metadata, ("metadata",))
        if not isinstance(metadata, Mapping):
            raise InstalledPluginRecordError("manifest metadata must be a JSON object")
        display_name = self.display_name
        if not isinstance(display_name, str):
            raise InstalledPluginRecordError("display name must be a string")
        manifest_version = _required_text(self.manifest_version, "manifest version")
        api_version = _required_text(self.api_version, "API version")
        if (
            manifest_version != _PLUGIN_MANIFEST_VERSION
            or api_version != _PLUGIN_SDK_API_VERSION
        ):
            raise InstalledPluginRecordError(
                "unsupported plugin manifest or SDK API version"
            )
        object.__setattr__(self, "manifest_version", manifest_version)
        object.__setattr__(self, "api_version", api_version)
        object.__setattr__(
            self, "version", _required_text(self.version, "plugin version")
        )
        object.__setattr__(
            self, "entrypoint", _required_text(self.entrypoint, "entrypoint")
        )
        if not _ENTRYPOINT_RE.fullmatch(self.entrypoint):
            raise InstalledPluginRecordError(
                "entrypoint must be a dotted Python symbol"
            )
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "display_name", display_name.strip())

    @property
    def aliases(self) -> tuple[str, ...]:
        """Return all declared aliases in deterministic tool order."""

        return tuple(alias for tool in self.tools for alias in tool.aliases)


@dataclass(frozen=True)
class InstalledPluginRecord:
    """One immutable installed-plugin generation and its lifecycle metadata."""

    plugin_id: str
    source: InstalledPluginSource
    manifest: InstalledPluginManifest
    enabled: bool = True
    status: InstalledPluginStatus = InstalledPluginStatus.ADMITTED
    generation: int = 0
    action_ids: frozenset[str] = field(default_factory=frozenset)
    task_ids: frozenset[str] = field(default_factory=frozenset)
    failure_code: str | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        plugin_id = normalize_plugin_id(self.plugin_id)
        if not isinstance(self.source, InstalledPluginSource):
            raise InstalledPluginRecordError("source must be an InstalledPluginSource")
        if not isinstance(self.manifest, InstalledPluginManifest):
            raise InstalledPluginRecordError(
                "manifest must be an InstalledPluginManifest"
            )
        if plugin_id in self.manifest.aliases or any(
            tool.canonical_id == plugin_id for tool in self.manifest.tools
        ):
            raise InstalledPluginRecordError(
                "plugin ID cannot collide with a declared tool name or alias"
            )
        if not isinstance(self.enabled, bool):
            raise InstalledPluginRecordError("enabled must be a bool")
        try:
            status = InstalledPluginStatus(self.status)
        except ValueError as exc:
            raise InstalledPluginRecordError("unknown installed plugin status") from exc
        if isinstance(self.generation, bool) or not isinstance(self.generation, int):
            raise InstalledPluginRecordError("generation must be an integer")
        if self.generation < 0:
            raise InstalledPluginRecordError("generation cannot be negative")
        action_ids = _freeze_identifiers(self.action_ids, "action IDs")
        task_ids = _freeze_identifiers(self.task_ids, "task IDs")
        if status is InstalledPluginStatus.DISABLED and self.enabled:
            raise InstalledPluginRecordError("disabled records must have enabled=False")
        if (
            status
            in {
                InstalledPluginStatus.ADMITTED,
                InstalledPluginStatus.INSTALLED,
                InstalledPluginStatus.ACTIVE,
            }
            and not self.enabled
        ):
            raise InstalledPluginRecordError(
                f"{status.value} records must have enabled=True"
            )
        if status is InstalledPluginStatus.REMOVED:
            if self.enabled:
                raise InstalledPluginRecordError(
                    "removed records must have enabled=False"
                )
            if action_ids or task_ids:
                raise InstalledPluginRecordError(
                    "removed records cannot retain ownership IDs"
                )
        failure_code = self.failure_code
        failure_reason = self.failure_reason
        if status is InstalledPluginStatus.FAILED:
            if not isinstance(failure_code, str) or not _FAILURE_CODE_RE.fullmatch(
                failure_code.strip().lower()
            ):
                raise InstalledPluginRecordError(
                    "failed records require a safe failure code"
                )
            if not isinstance(failure_reason, str) or not failure_reason.strip():
                raise InstalledPluginRecordError(
                    "failed records require a failure reason"
                )
            failure_code = failure_code.strip().lower()
            failure_reason = failure_reason.strip()
            if (
                len(failure_reason) > 512
                or "\n" in failure_reason
                or "\r" in failure_reason
            ):
                raise InstalledPluginRecordError(
                    "failure reason must be a short single line"
                )
        elif failure_code is not None or failure_reason is not None:
            raise InstalledPluginRecordError(
                "failure details are only valid for failed records"
            )
        object.__setattr__(self, "plugin_id", plugin_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "action_ids", action_ids)
        object.__setattr__(self, "task_ids", task_ids)
        object.__setattr__(self, "failure_code", failure_code)
        object.__setattr__(self, "failure_reason", failure_reason)

    @classmethod
    def from_manifest(
        cls,
        source: object,
        manifest: object,
        *,
        enabled: bool = True,
        status: InstalledPluginStatus = InstalledPluginStatus.ADMITTED,
    ) -> InstalledPluginRecord:
        """Copy a static PluginSDK-like declaration without retaining it."""

        source_snapshot = InstalledPluginSource(
            getattr(source, "path", None), getattr(source, "digest", None)
        )
        plugin_id = getattr(manifest, "plugin_id", None)
        raw_tools = getattr(manifest, "tools", ())
        if isinstance(raw_tools, str):
            raise InstalledPluginRecordError("manifest tools must be an iterable")
        try:
            manifest_tools = []
            tool_declarations = tuple(raw_tools)
        except TypeError as exc:
            raise InstalledPluginRecordError(
                "manifest tools must be an iterable"
            ) from exc
        for tool in tool_declarations:
            manifest_tools.append(
                InstalledPluginTool(
                    canonical_id=getattr(tool, "canonical_id", None),
                    aliases=getattr(tool, "aliases", ()),
                    capabilities=getattr(tool, "capabilities", ()),
                    description=getattr(tool, "description", ""),
                    input_schema=getattr(tool, "input_schema", {}),
                    output_schema=getattr(tool, "output_schema", {}),
                    confirmation=getattr(
                        getattr(tool, "confirmation", "none"),
                        "value",
                        getattr(tool, "confirmation", "none"),
                    ),
                    concurrency=getattr(
                        getattr(tool, "concurrency", "serial"),
                        "value",
                        getattr(tool, "concurrency", "serial"),
                    ),
                    idempotency=getattr(
                        getattr(tool, "idempotency", "idempotent"),
                        "value",
                        getattr(tool, "idempotency", "idempotent"),
                    ),
                    migration_disposition=getattr(
                        getattr(tool, "migration_disposition", "migrate"),
                        "value",
                        getattr(tool, "migration_disposition", "migrate"),
                    ),
                )
            )
        metadata = getattr(manifest, "metadata", {})
        display_name = ""
        if isinstance(metadata, Mapping):
            name = metadata.get("display_name", metadata.get("name", ""))
            if isinstance(name, str):
                display_name = name
        status_value = InstalledPluginStatus(status)
        if not enabled and status_value is InstalledPluginStatus.ADMITTED:
            status_value = InstalledPluginStatus.DISABLED
        return cls(
            plugin_id=plugin_id,
            source=source_snapshot,
            manifest=InstalledPluginManifest(
                manifest_version=getattr(manifest, "manifest_version", None),
                api_version=getattr(manifest, "api_version", None),
                version=getattr(manifest, "version", None),
                entrypoint=getattr(manifest, "entrypoint", None),
                tools=tuple(manifest_tools),
                capabilities=getattr(manifest, "capabilities", ()),
                metadata=metadata,
                display_name=display_name,
            ),
            enabled=enabled,
            status=status_value,
        )

    @property
    def canonical_plugin_id(self) -> str:
        """Return the normalized plugin identity used for every registry key."""

        return self.plugin_id

    @property
    def source_path(self) -> str:
        return self.source.path

    @property
    def source_digest(self) -> str:
        return self.source.digest

    @property
    def manifest_version(self) -> str:
        return self.manifest.manifest_version

    @property
    def api_version(self) -> str:
        return self.manifest.api_version

    @property
    def version(self) -> str:
        return self.manifest.version

    @property
    def entrypoint(self) -> str:
        return self.manifest.entrypoint

    @property
    def display_name(self) -> str:
        return self.manifest.display_name

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.manifest.metadata

    @property
    def capabilities(self) -> frozenset[str]:
        return self.manifest.capabilities

    @property
    def tools(self) -> tuple[InstalledPluginTool, ...]:
        return self.manifest.tools

    @property
    def aliases(self) -> tuple[str, ...]:
        return self.manifest.aliases

    @property
    def callback_ids(self) -> frozenset[str]:
        """Return opaque callback token IDs owned by this generation."""

        return self.action_ids

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(name for tool in self.tools for name in tool.names)


class InstalledPluginRegistry:
    """Authoritative lock-safe registry for immutable installed-plugin records.

    Lifecycle revisions consume monotonically increasing generation numbers.
    Ownership changes keep the current generation so multiple independent action
    and task IDs can remain valid until the next lifecycle transition.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._records: dict[str, InstalledPluginRecord] = {}
        self._path_owners: dict[str, str] = {}
        self._name_owners: dict[str, str] = {}
        self._action_owners: dict[str, tuple[str, int]] = {}
        self._task_owners: dict[str, tuple[str, int]] = {}
        self._next_generation = 1

    @property
    def next_generation(self) -> int:
        """Return the next generation reserved for a lifecycle revision."""

        with self._lock:
            return self._next_generation

    def snapshot(
        self,
        status: InstalledPluginStatus | str | None = None,
        *,
        state: InstalledPluginStatus | str | None = None,
    ) -> tuple[InstalledPluginRecord, ...]:
        """Return a deterministic immutable snapshot of live records."""

        if status is not None and state is not None:
            raise InstalledPluginRecordError("provide either status or state, not both")
        selected = status if status is not None else state
        enabled_only = (
            isinstance(selected, str) and selected.strip().lower() == "enabled"
        )
        selected_status = None if enabled_only else self._status_filter(selected)
        with self._lock:
            records = tuple(
                record
                for _, record in sorted(self._records.items())
                if (selected_status is None or record.status is selected_status)
                and (not enabled_only or record.enabled)
            )
        return records

    def catalog_snapshot(self) -> tuple[InstalledPluginRecord, ...]:
        """Return all records, including disabled and failed declarations."""

        return self.snapshot()

    def get(self, plugin_id: object) -> InstalledPluginRecord:
        """Return one canonical record or raise a typed missing error."""

        canonical_id = normalize_plugin_id(plugin_id)
        with self._lock:
            record = self._records.get(canonical_id)
        if record is None:
            raise InstalledPluginMissingError(
                f"plugin {canonical_id!r} is not installed", plugin_id=canonical_id
            )
        return record

    def find(self, plugin_id: object) -> InstalledPluginRecord | None:
        """Return one record if present without converting absence to an error."""

        canonical_id = normalize_plugin_id(plugin_id)
        with self._lock:
            return self._records.get(canonical_id)

    def get_by_path(self, source_path: object) -> InstalledPluginRecord:
        """Return the record owning one canonical source path."""

        path = _normalize_source_path(source_path)
        with self._lock:
            plugin_id = self._path_owners.get(path)
            record = self._records.get(plugin_id) if plugin_id is not None else None
        if record is None:
            raise InstalledPluginMissingError(f"source path {path!r} is not installed")
        return record

    def find_by_path(self, source_path: object) -> InstalledPluginRecord | None:
        """Return the record owning one canonical source path, if present."""

        path = _normalize_source_path(source_path)
        with self._lock:
            plugin_id = self._path_owners.get(path)
            return self._records.get(plugin_id) if plugin_id is not None else None

    def get_active_tool_owner(self, tool_name: object) -> InstalledPluginRecord:
        """Return the active record declaring a canonical tool name or alias."""

        name = _normalize_tool_name(tool_name)
        with self._lock:
            plugin_id = self._name_owners.get(name)
            record = self._records.get(plugin_id) if plugin_id is not None else None
        if record is None or record.status is not InstalledPluginStatus.ACTIVE:
            raise InstalledPluginMissingError(
                f"tool {name!r} has no active plugin owner"
            )
        return record

    def admit(
        self,
        record: InstalledPluginRecord,
        *,
        expected_generation: int | None = None,
    ) -> InstalledPluginRecord:
        """Publish a statically validated source before it is installed."""

        candidate = self._require_candidate(record)
        if expected_generation is not None:
            raise InstalledPluginRecordError(
                "new admissions require expected_generation=None"
            )
        self._require_unowned_candidate(candidate)
        with self._lock:
            self._check_collisions(candidate)
            published = self._new_record(
                candidate,
                status=(
                    InstalledPluginStatus.ADMITTED
                    if candidate.enabled
                    else InstalledPluginStatus.DISABLED
                ),
                enabled=candidate.enabled,
            )
            self._publish(None, published)
            return published

    def install(
        self,
        record: InstalledPluginRecord,
        *,
        expected_generation: int | None = None,
    ) -> InstalledPluginRecord:
        """Publish a new install or promote its admitted generation."""

        candidate = self._require_candidate(record)
        self._require_unowned_candidate(candidate)
        with self._lock:
            current = self._records.get(candidate.plugin_id)
            if current is None:
                if expected_generation is not None:
                    raise InstalledPluginMissingError(
                        f"plugin {candidate.plugin_id!r} is not admitted",
                        plugin_id=candidate.plugin_id,
                    )
                self._check_collisions(candidate)
            else:
                self._require_generation(current, expected_generation)
                if current.status is not InstalledPluginStatus.ADMITTED:
                    raise InstalledPluginCollisionError(
                        f"plugin {candidate.plugin_id!r} is already installed",
                        plugin_id=candidate.plugin_id,
                    )
                if (
                    current.plugin_id != candidate.plugin_id
                    or current.source != candidate.source
                    or current.manifest != candidate.manifest
                ):
                    raise InstalledPluginRecordError(
                        "installation candidate does not match the admitted declaration",
                        plugin_id=candidate.plugin_id,
                    )
                self._require_unowned(current)
                self._check_collisions(candidate, replacing_plugin_id=current.plugin_id)
            published = self._new_record(
                candidate,
                status=(
                    InstalledPluginStatus.INSTALLED
                    if candidate.enabled
                    else InstalledPluginStatus.DISABLED
                ),
                enabled=candidate.enabled,
            )
            self._publish(current, published)
            return published

    def install_active(self, record: InstalledPluginRecord) -> InstalledPluginRecord:
        """Publish a new admitted source in its final executable state.

        Installing an enabled source does not expose a transient ``INSTALLED``
        generation.  Callers validate the isolated source before this mutation
        and bind the returned active record synchronously afterward.
        """

        candidate = self._require_candidate(record)
        self._require_unowned_candidate(candidate)
        with self._lock:
            if candidate.plugin_id in self._records:
                raise InstalledPluginCollisionError(
                    f"plugin {candidate.plugin_id!r} is already installed",
                    plugin_id=candidate.plugin_id,
                )
            self._check_collisions(candidate)
            published = self._new_record(
                candidate,
                status=(
                    InstalledPluginStatus.ACTIVE
                    if candidate.enabled
                    else InstalledPluginStatus.DISABLED
                ),
                enabled=candidate.enabled,
            )
            self._publish(None, published)
            return published

    def replace(
        self,
        record: InstalledPluginRecord,
        *,
        expected_generation: int,
    ) -> InstalledPluginRecord:
        """Atomically replace a quiesced lifecycle generation with a new source."""

        candidate = self._require_candidate(record)
        self._require_unowned_candidate(candidate)
        with self._lock:
            current = self._current(candidate.plugin_id)
            self._require_generation(current, expected_generation)
            if current.status not in {
                InstalledPluginStatus.ADMITTED,
                InstalledPluginStatus.INSTALLED,
                InstalledPluginStatus.DISABLED,
                InstalledPluginStatus.FAILED,
            }:
                raise InstalledPluginTransitionError(
                    f"cannot replace a {current.status.value} plugin generation",
                    plugin_id=current.plugin_id,
                    actual_generation=current.generation,
                )
            self._require_unowned(current)
            self._check_collisions(candidate, replacing_plugin_id=current.plugin_id)
            published = self._new_record(
                candidate,
                status=(
                    InstalledPluginStatus.INSTALLED
                    if candidate.enabled
                    else InstalledPluginStatus.DISABLED
                ),
                enabled=candidate.enabled,
            )
            self._publish(current, published)
            return published

    def replace_active(
        self,
        record: InstalledPluginRecord,
        *,
        expected_generation: int,
    ) -> InstalledPluginRecord:
        """Atomically replace one active generation with another active one.

        This intentionally skips the externally visible unloading state.  Task
        ownership must already be released, while callback ownership is retired
        with the old generation as the replacement is published.
        """

        candidate = self._require_candidate(record)
        self._require_unowned_candidate(candidate)
        with self._lock:
            current = self._current(candidate.plugin_id)
            self._require_generation(current, expected_generation)
            if current.status is not InstalledPluginStatus.ACTIVE:
                raise InstalledPluginTransitionError(
                    f"cannot replace a {current.status.value} plugin generation",
                    plugin_id=current.plugin_id,
                    actual_generation=current.generation,
                )
            self._require_no_tasks(current)
            self._check_collisions(candidate, replacing_plugin_id=current.plugin_id)
            published = self._new_record(
                candidate,
                enabled=current.enabled,
                status=InstalledPluginStatus.ACTIVE,
                action_ids=frozenset(),
                task_ids=frozenset(),
                failure_code=None,
                failure_reason=None,
            )
            self._publish(current, published)
            return published

    def set_enabled(
        self,
        plugin_id: object,
        enabled: bool,
        *,
        expected_generation: int,
    ) -> InstalledPluginRecord:
        """Persist the authoritative enabled intent for an installed record."""

        if not isinstance(enabled, bool):
            raise InstalledPluginRecordError("enabled must be a bool")
        canonical_id = normalize_plugin_id(plugin_id)
        with self._lock:
            current = self._current(canonical_id)
            self._require_generation(current, expected_generation)
            if current.enabled is enabled:
                return current
            if enabled:
                if current.status is not InstalledPluginStatus.DISABLED:
                    raise InstalledPluginTransitionError(
                        f"cannot enable a {current.status.value} plugin generation",
                        plugin_id=current.plugin_id,
                        actual_generation=current.generation,
                    )
                published = self._new_record(
                    current,
                    enabled=True,
                    status=InstalledPluginStatus.INSTALLED,
                )
            else:
                if current.status in {
                    InstalledPluginStatus.ADMITTED,
                    InstalledPluginStatus.INSTALLED,
                    InstalledPluginStatus.FAILED,
                }:
                    self._require_unowned(current)
                    published = self._new_record(
                        current,
                        enabled=False,
                        status=InstalledPluginStatus.DISABLED,
                        failure_code=None,
                        failure_reason=None,
                    )
                elif current.status is InstalledPluginStatus.ACTIVE:
                    self._require_no_tasks(current)
                    published = self._new_record(
                        current,
                        enabled=False,
                        status=InstalledPluginStatus.UNLOADING,
                        action_ids=frozenset(),
                    )
                else:
                    raise InstalledPluginTransitionError(
                        f"cannot disable a {current.status.value} plugin generation",
                        plugin_id=current.plugin_id,
                        actual_generation=current.generation,
                    )
            self._publish(current, published)
            return published

    def activate(
        self, plugin_id: object, *, expected_generation: int
    ) -> InstalledPluginRecord:
        """Mark an installed enabled source as eligible for isolated execution."""

        canonical_id = normalize_plugin_id(plugin_id)
        with self._lock:
            current = self._current(canonical_id)
            self._require_generation(current, expected_generation)
            if (
                current.status is not InstalledPluginStatus.INSTALLED
                or not current.enabled
            ):
                raise InstalledPluginTransitionError(
                    f"cannot activate a {current.status.value} plugin generation",
                    plugin_id=current.plugin_id,
                    actual_generation=current.generation,
                )
            self._require_unowned(current)
            published = self._new_record(
                current,
                status=InstalledPluginStatus.ACTIVE,
            )
            self._publish(current, published)
            return published

    def begin_unload(
        self, plugin_id: object, *, expected_generation: int
    ) -> InstalledPluginRecord:
        """Quiesce an active record after its owned tasks have been released."""

        canonical_id = normalize_plugin_id(plugin_id)
        with self._lock:
            current = self._current(canonical_id)
            self._require_generation(current, expected_generation)
            if current.status is not InstalledPluginStatus.ACTIVE:
                raise InstalledPluginTransitionError(
                    f"cannot unload a {current.status.value} plugin generation",
                    plugin_id=current.plugin_id,
                    actual_generation=current.generation,
                )
            self._require_no_tasks(current)
            published = self._new_record(
                current,
                status=InstalledPluginStatus.UNLOADING,
                action_ids=frozenset(),
            )
            self._publish(current, published)
            return published

    def complete_unload(
        self, plugin_id: object, *, expected_generation: int
    ) -> InstalledPluginRecord:
        """Complete quiescing and return to installed or disabled state."""

        canonical_id = normalize_plugin_id(plugin_id)
        with self._lock:
            current = self._current(canonical_id)
            self._require_generation(current, expected_generation)
            if current.status is not InstalledPluginStatus.UNLOADING:
                raise InstalledPluginTransitionError(
                    f"cannot complete unload for a {current.status.value} plugin generation",
                    plugin_id=current.plugin_id,
                    actual_generation=current.generation,
                )
            self._require_unowned(current)
            published = self._new_record(
                current,
                status=(
                    InstalledPluginStatus.INSTALLED
                    if current.enabled
                    else InstalledPluginStatus.DISABLED
                ),
            )
            self._publish(current, published)
            return published

    def fail(
        self,
        plugin_id: object,
        failure_code: object,
        failure_reason: object,
        *,
        expected_generation: int,
    ) -> InstalledPluginRecord:
        """Record a safe lifecycle failure after owned work has stopped."""

        code = _required_text(failure_code, "failure code").lower()
        reason = _required_text(failure_reason, "failure reason")
        if not _FAILURE_CODE_RE.fullmatch(code):
            raise InstalledPluginRecordError("failure code must be a safe identifier")
        if len(reason) > 512 or "\n" in reason or "\r" in reason:
            raise InstalledPluginRecordError(
                "failure reason must be a short single line"
            )
        canonical_id = normalize_plugin_id(plugin_id)
        with self._lock:
            current = self._current(canonical_id)
            self._require_generation(current, expected_generation)
            if current.status in {
                InstalledPluginStatus.DISABLED,
                InstalledPluginStatus.REMOVED,
            }:
                raise InstalledPluginTransitionError(
                    f"cannot fail a {current.status.value} plugin generation",
                    plugin_id=current.plugin_id,
                    actual_generation=current.generation,
                )
            self._require_no_tasks(current)
            published = self._new_record(
                current,
                status=InstalledPluginStatus.FAILED,
                action_ids=frozenset(),
                failure_code=code,
                failure_reason=reason,
            )
            self._publish(current, published)
            return published

    def remove(
        self, plugin_id: object, *, expected_generation: int
    ) -> InstalledPluginRecord:
        """Remove a non-active, fully released record and its collision indexes."""

        canonical_id = normalize_plugin_id(plugin_id)
        with self._lock:
            current = self._current(canonical_id)
            self._require_generation(current, expected_generation)
            if current.status not in {
                InstalledPluginStatus.ADMITTED,
                InstalledPluginStatus.INSTALLED,
                InstalledPluginStatus.DISABLED,
                InstalledPluginStatus.FAILED,
            }:
                raise InstalledPluginTransitionError(
                    f"cannot remove a {current.status.value} plugin generation",
                    plugin_id=current.plugin_id,
                    actual_generation=current.generation,
                )
            self._require_unowned(current)
            removed = self._new_record(
                current,
                enabled=False,
                status=InstalledPluginStatus.REMOVED,
                failure_code=None,
                failure_reason=None,
            )
            self._remove_indexes(current)
            del self._records[current.plugin_id]
            return removed

    def remove_active(
        self, plugin_id: object, *, expected_generation: int
    ) -> InstalledPluginRecord:
        """Remove an active generation after its owned tasks are quiescent.

        Callback ownership is retired with the returned tombstone.  Live
        indexes are removed in the same lock-held mutation, so the old
        generation cannot be resolved or reused after this method returns.
        """

        canonical_id = normalize_plugin_id(plugin_id)
        with self._lock:
            current = self._current(canonical_id)
            self._require_generation(current, expected_generation)
            if current.status is not InstalledPluginStatus.ACTIVE:
                raise InstalledPluginTransitionError(
                    f"cannot remove a {current.status.value} plugin generation",
                    plugin_id=current.plugin_id,
                    actual_generation=current.generation,
                )
            self._require_no_tasks(current)
            removed = self._new_record(
                current,
                enabled=False,
                status=InstalledPluginStatus.REMOVED,
                action_ids=frozenset(),
                task_ids=frozenset(),
            )
            self._remove_indexes(current)
            del self._records[current.plugin_id]
            return removed

    def bind_action(
        self,
        plugin_id: object,
        action_id: object,
        *,
        expected_generation: int,
    ) -> InstalledPluginRecord:
        """Bind an opaque callback token to the current plugin generation."""

        canonical_id = normalize_plugin_id(plugin_id)
        token = _required_text(action_id, "action ID")
        with self._lock:
            current = self._current(canonical_id)
            self._require_generation(current, expected_generation)
            if current.status in {
                InstalledPluginStatus.UNLOADING,
                InstalledPluginStatus.REMOVED,
            }:
                raise InstalledPluginTransitionError(
                    f"cannot bind an action to a {current.status.value} plugin generation",
                    plugin_id=current.plugin_id,
                    actual_generation=current.generation,
                )
            owner = self._action_owners.get(token)
            if owner is not None:
                raise InstalledPluginCollisionError(
                    f"action ID {token!r} is already owned by {owner[0]!r}",
                    plugin_id=current.plugin_id,
                )
            updated = dataclass_replace(
                current, action_ids=current.action_ids | frozenset({token})
            )
            self._records[current.plugin_id] = updated
            self._action_owners[token] = (updated.plugin_id, updated.generation)
            return updated

    def release_action(
        self,
        plugin_id: object,
        action_id: object,
        *,
        expected_generation: int,
    ) -> InstalledPluginRecord:
        """Release a callback token after it has been consumed or expired."""

        canonical_id = normalize_plugin_id(plugin_id)
        token = _required_text(action_id, "action ID")
        with self._lock:
            current = self._current(canonical_id)
            self._require_generation(current, expected_generation)
            if token not in current.action_ids:
                raise InstalledPluginMissingError(
                    f"action ID {token!r} is not owned by {current.plugin_id!r}",
                    plugin_id=current.plugin_id,
                    actual_generation=current.generation,
                )
            if self._action_owners.get(token) != (
                current.plugin_id,
                current.generation,
            ):
                raise InstalledPluginMissingError(
                    f"action ID {token!r} has no matching owner",
                    plugin_id=current.plugin_id,
                    actual_generation=current.generation,
                )
            updated = dataclass_replace(
                current, action_ids=current.action_ids - frozenset({token})
            )
            self._records[current.plugin_id] = updated
            self._action_owners.pop(token, None)
            return updated

    def release_actions(
        self,
        plugin_id: object,
        action_ids: Iterable[object],
        *,
        expected_generation: int,
    ) -> InstalledPluginRecord:
        """Atomically release every callback owned by one exact generation."""

        canonical_id = normalize_plugin_id(plugin_id)
        tokens = frozenset(
            _required_text(action_id, "action ID") for action_id in action_ids
        )
        with self._lock:
            current = self._current(canonical_id)
            self._require_generation(current, expected_generation)
            if tokens != current.action_ids:
                raise InstalledPluginMissingError(
                    f"action IDs do not match ownership for {current.plugin_id!r}",
                    plugin_id=current.plugin_id,
                    actual_generation=current.generation,
                )
            expected_owner = (current.plugin_id, current.generation)
            if any(
                self._action_owners.get(token) != expected_owner for token in tokens
            ):
                raise InstalledPluginMissingError(
                    f"action IDs have no matching owner for {current.plugin_id!r}",
                    plugin_id=current.plugin_id,
                    actual_generation=current.generation,
                )
            updated = dataclass_replace(current, action_ids=current.action_ids - tokens)
            self._records[current.plugin_id] = updated
            for token in tokens:
                self._action_owners.pop(token)
            return updated

    def own_task(
        self,
        plugin_id: object,
        task_id: object,
        *,
        expected_generation: int,
    ) -> InstalledPluginRecord:
        """Bind an opaque task ID without retaining its Future or task object."""

        canonical_id = normalize_plugin_id(plugin_id)
        token = _required_text(task_id, "task ID")
        with self._lock:
            current = self._current(canonical_id)
            self._require_generation(current, expected_generation)
            if current.status is not InstalledPluginStatus.ACTIVE:
                raise InstalledPluginTransitionError(
                    f"cannot own a task for a {current.status.value} plugin generation",
                    plugin_id=current.plugin_id,
                    actual_generation=current.generation,
                )
            owner = self._task_owners.get(token)
            if owner is not None:
                raise InstalledPluginCollisionError(
                    f"task ID {token!r} is already owned by {owner[0]!r}",
                    plugin_id=current.plugin_id,
                )
            updated = dataclass_replace(
                current, task_ids=current.task_ids | frozenset({token})
            )
            self._records[current.plugin_id] = updated
            self._task_owners[token] = (updated.plugin_id, updated.generation)
            return updated

    def release_task(
        self,
        plugin_id: object,
        task_id: object,
        *,
        expected_generation: int,
    ) -> InstalledPluginRecord:
        """Release an opaque task ID after cancellation or completion."""

        canonical_id = normalize_plugin_id(plugin_id)
        token = _required_text(task_id, "task ID")
        with self._lock:
            current = self._current(canonical_id)
            self._require_generation(current, expected_generation)
            if token not in current.task_ids:
                raise InstalledPluginMissingError(
                    f"task ID {token!r} is not owned by {current.plugin_id!r}",
                    plugin_id=current.plugin_id,
                    actual_generation=current.generation,
                )
            updated = dataclass_replace(
                current, task_ids=current.task_ids - frozenset({token})
            )
            self._records[current.plugin_id] = updated
            self._task_owners.pop(token, None)
            return updated

    def get_action_owner(
        self, action_id: object, *, expected_generation: int
    ) -> InstalledPluginRecord:
        """Resolve an action token only for its exact bound generation."""

        token = _required_text(action_id, "action ID")
        with self._lock:
            owner = self._action_owners.get(token)
            if owner is None:
                raise InstalledPluginMissingError(f"action ID {token!r} is not bound")
            record = self._records.get(owner[0])
            if record is None or record.generation != owner[1]:
                raise InstalledPluginStaleGenerationError(
                    f"action ID {token!r} belongs to a retired generation",
                    plugin_id=owner[0],
                    expected_generation=expected_generation,
                    actual_generation=record.generation if record is not None else None,
                )
            self._require_generation(record, expected_generation)
            return record

    def get_task_owner(
        self, task_id: object, *, expected_generation: int
    ) -> InstalledPluginRecord:
        """Resolve a task owner only for its exact bound generation."""

        token = _required_text(task_id, "task ID")
        with self._lock:
            owner = self._task_owners.get(token)
            if owner is None:
                raise InstalledPluginMissingError(f"task ID {token!r} is not owned")
            record = self._records.get(owner[0])
            if record is None or record.generation != owner[1]:
                raise InstalledPluginStaleGenerationError(
                    f"task ID {token!r} belongs to a retired generation",
                    plugin_id=owner[0],
                    expected_generation=expected_generation,
                    actual_generation=record.generation if record is not None else None,
                )
            self._require_generation(record, expected_generation)
            return record

    def _status_filter(
        self, status: InstalledPluginStatus | str | None
    ) -> InstalledPluginStatus | None:
        if status is None:
            return None
        try:
            return InstalledPluginStatus(status)
        except ValueError as exc:
            raise InstalledPluginRecordError("unknown installed plugin status") from exc

    def _require_candidate(
        self, record: InstalledPluginRecord
    ) -> InstalledPluginRecord:
        if not isinstance(record, InstalledPluginRecord):
            raise InstalledPluginRecordError(
                "registry mutations require InstalledPluginRecord"
            )
        return record

    def _require_unowned_candidate(self, record: InstalledPluginRecord) -> None:
        if record.action_ids or record.task_ids:
            raise InstalledPluginRecordError(
                "new or replacement records cannot carry ownership IDs"
            )

    def _current(self, plugin_id: str) -> InstalledPluginRecord:
        record = self._records.get(plugin_id)
        if record is None:
            raise InstalledPluginMissingError(
                f"plugin {plugin_id!r} is not installed", plugin_id=plugin_id
            )
        return record

    def _require_generation(
        self, record: InstalledPluginRecord, expected_generation: int | None
    ) -> None:
        if (
            isinstance(expected_generation, bool)
            or not isinstance(expected_generation, int)
            or expected_generation <= 0
        ):
            raise InstalledPluginRecordError(
                "expected generation must be a positive integer"
            )
        if record.generation != expected_generation:
            raise InstalledPluginStaleGenerationError(
                f"plugin {record.plugin_id!r} is generation {record.generation}, not {expected_generation}",
                plugin_id=record.plugin_id,
                expected_generation=expected_generation,
                actual_generation=record.generation,
            )

    def _require_unowned(self, record: InstalledPluginRecord) -> None:
        if record.action_ids or record.task_ids:
            raise InstalledPluginTransitionError(
                "release callback and task ownership before this transition",
                plugin_id=record.plugin_id,
                actual_generation=record.generation,
            )

    def _require_no_tasks(self, record: InstalledPluginRecord) -> None:
        if record.task_ids:
            raise InstalledPluginTransitionError(
                "release owned tasks before beginning unload or failure handling",
                plugin_id=record.plugin_id,
                actual_generation=record.generation,
            )

    def _new_record(
        self, record: InstalledPluginRecord, **changes: Any
    ) -> InstalledPluginRecord:
        changes["generation"] = self._next_generation
        self._next_generation += 1
        return dataclass_replace(record, **changes)

    def _check_collisions(
        self,
        candidate: InstalledPluginRecord,
        *,
        replacing_plugin_id: str | None = None,
    ) -> None:
        existing = self._records.get(candidate.plugin_id)
        if existing is not None and candidate.plugin_id != replacing_plugin_id:
            raise InstalledPluginCollisionError(
                f"plugin ID {candidate.plugin_id!r} is already installed",
                plugin_id=candidate.plugin_id,
            )
        plugin_name_owner = self._name_owners.get(candidate.plugin_id)
        if plugin_name_owner is not None and plugin_name_owner != replacing_plugin_id:
            raise InstalledPluginCollisionError(
                f"plugin ID {candidate.plugin_id!r} collides with a tool owned by {plugin_name_owner!r}",
                plugin_id=candidate.plugin_id,
            )
        path_owner = self._path_owners.get(candidate.source.path)
        if path_owner is not None and path_owner != replacing_plugin_id:
            raise InstalledPluginCollisionError(
                f"source path {candidate.source.path!r} is already owned by {path_owner!r}",
                plugin_id=candidate.plugin_id,
            )
        for tool_name in candidate.tool_names:
            owner = self._name_owners.get(tool_name)
            if owner is not None and owner != replacing_plugin_id:
                raise InstalledPluginCollisionError(
                    f"tool name {tool_name!r} is already owned by {owner!r}",
                    plugin_id=candidate.plugin_id,
                )
            plugin_owner = self._records.get(tool_name)
            if plugin_owner is not None and tool_name != replacing_plugin_id:
                raise InstalledPluginCollisionError(
                    f"tool name {tool_name!r} collides with plugin ID {tool_name!r}",
                    plugin_id=candidate.plugin_id,
                )

    def _publish(
        self,
        current: InstalledPluginRecord | None,
        published: InstalledPluginRecord,
    ) -> None:
        if current is not None:
            self._remove_indexes(current)
        self._records[published.plugin_id] = published
        self._path_owners[published.source.path] = published.plugin_id
        for tool_name in published.tool_names:
            self._name_owners[tool_name] = published.plugin_id
        for action_id in published.action_ids:
            self._action_owners[action_id] = (
                published.plugin_id,
                published.generation,
            )
        for task_id in published.task_ids:
            self._task_owners[task_id] = (published.plugin_id, published.generation)

    def _remove_indexes(self, record: InstalledPluginRecord) -> None:
        if self._path_owners.get(record.source.path) == record.plugin_id:
            self._path_owners.pop(record.source.path, None)
        for tool_name in record.tool_names:
            if self._name_owners.get(tool_name) == record.plugin_id:
                self._name_owners.pop(tool_name, None)
        for action_id in record.action_ids:
            if self._action_owners.get(action_id) == (
                record.plugin_id,
                record.generation,
            ):
                self._action_owners.pop(action_id, None)
        for task_id in record.task_ids:
            if self._task_owners.get(task_id) == (
                record.plugin_id,
                record.generation,
            ):
                self._task_owners.pop(task_id, None)
