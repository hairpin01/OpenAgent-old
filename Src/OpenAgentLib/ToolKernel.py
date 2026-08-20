# SPDX-License-Identifier: MIT
"""Immutable v2 tool contracts, schema validation, and registry metadata.

This module deliberately contains no dispatch or execution code.  It is the
boundary where tool declarations become immutable, validated v2 metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


TOOL_API_VERSION = "2"
TOOL_SCHEMA_VERSION = "2"

_CANONICAL_ID_RE = re.compile(r"^[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)+$")
_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)*$")
_CAPABILITY_RE = re.compile(r"^[a-z][a-z0-9-]*$")


class ToolErrorCode(str, Enum):
    """Stable error codes for contract and registry failures."""

    ADAPTER_DRIFT = "adapter_drift"
    API_VERSION_MISMATCH = "api_version_mismatch"
    DUPLICATE_ALIAS = "duplicate_alias"
    DUPLICATE_TOOL = "duplicate_tool"
    INVALID_ARGUMENT = "invalid_argument"
    INVALID_CALL = "invalid_call"
    INVALID_NAME = "invalid_name"
    INVALID_SCHEMA = "invalid_schema"
    INVALID_SPEC = "invalid_spec"
    POLICY_DENIED = "policy_denied"
    CONFIRMATION_REQUIRED = "confirmation_required"
    HOOK_CANCELLED = "hook_cancelled"
    HOOK_FAILED = "hook_failed"
    HANDLER_FAILED = "handler_failed"
    HOST_FAILED = "host_failed"
    OUTPUT_SCHEMA_INVALID = "output_schema_invalid"
    SPILL_FAILED = "spill_failed"
    EXECUTOR_FAILED = "executor_failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    BATCH_LENGTH_MISMATCH = "batch_length_mismatch"
    DUPLICATE_CALL_ID = "duplicate_call_id"
    SCHEMA_VERSION_MISMATCH = "schema_version_mismatch"
    UNDECLARED_ALIAS = "undeclared_alias"


class ConfirmationRequirement(str, Enum):
    NONE = "none"
    REQUIRED = "required"


class ConcurrencyClass(str, Enum):
    SERIAL = "serial"
    PARALLEL_READ = "parallel-read"


class IdempotencyClass(str, Enum):
    IDEMPOTENT = "idempotent"
    NON_IDEMPOTENT = "non-idempotent"


class MigrationDisposition(str, Enum):
    MIGRATE = "migrate"
    REJECT = "reject"


class ToolResultStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class ToolTraceState(str, Enum):
    CREATED = "created"
    VALIDATED = "validated"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


def _normalise_error_name(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("tool error names must be strings")
    return value.strip().lower()


@dataclass(frozen=True)
class ToolError:
    """Serializable error details retained by exceptions and failed results."""

    code: ToolErrorCode
    message: str
    canonical_id: str | None = None
    requested_name: str | None = None
    field_path: tuple[str | int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.code, ToolErrorCode):
            object.__setattr__(self, "code", ToolErrorCode(self.code))
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("tool error message is required")
        object.__setattr__(self, "canonical_id", _normalise_error_name(self.canonical_id))
        object.__setattr__(self, "requested_name", _normalise_error_name(self.requested_name))
        path = tuple(self.field_path)
        if any(not isinstance(part, (str, int)) for part in path):
            raise TypeError("tool error field paths contain only strings and integers")
        object.__setattr__(self, "field_path", path)

    @property
    def canonical_name(self) -> str | None:
        return self.canonical_id


class ToolKernelError(ValueError):
    """Base error carrying machine-readable v2 failure details."""

    def __init__(
        self,
        code: ToolErrorCode,
        message: str,
        *,
        canonical_id: str | None = None,
        requested_name: str | None = None,
        field_path: Sequence[str | int] = (),
    ) -> None:
        self.details = ToolError(
            code=code,
            message=message,
            canonical_id=canonical_id,
            requested_name=requested_name,
            field_path=tuple(field_path),
        )
        super().__init__(message)

    @property
    def code(self) -> ToolErrorCode:
        return self.details.code

    @property
    def canonical_id(self) -> str | None:
        return self.details.canonical_id

    @property
    def canonical_name(self) -> str | None:
        return self.details.canonical_id

    @property
    def requested_name(self) -> str | None:
        return self.details.requested_name

    @property
    def field_path(self) -> tuple[str | int, ...]:
        return self.details.field_path


class ToolNameError(ToolKernelError):
    """A tool canonical ID or alias is malformed."""


class ToolSchemaError(ToolKernelError):
    """A bounded v2 JSON-schema-like declaration is malformed."""


class ToolArgumentError(ToolKernelError):
    """Arguments do not conform to a tool input schema."""


class ToolVersionError(ToolKernelError):
    """Tool API or schema versions do not match."""


class ToolRegistryError(ToolKernelError):
    """Registry construction or call acceptance failed."""


class ToolUndeclaredAliasError(ToolRegistryError):
    """The requested name is not a canonical ID or declared alias."""


class SystemToolAdapterError(ToolKernelError):
    """A legacy system descriptor cannot be represented by a v2 spec."""


def _required_text(value: object, field_name: str, error_type: type[ToolKernelError]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error_type(
            ToolErrorCode.INVALID_SPEC,
            f"{field_name} must be a non-empty string",
            field_path=(field_name,),
        )
    return value.strip()


def normalize_tool_name(value: object, *, canonical: bool = False) -> str:
    """Return the one allowed spelling for a canonical ID or alias."""

    if not isinstance(value, str):
        raise ToolNameError(
            ToolErrorCode.INVALID_NAME,
            "tool names must be strings",
            field_path=("tool_name",),
        )
    normalized = value.strip().lower()
    pattern = _CANONICAL_ID_RE if canonical else _TOOL_NAME_RE
    if not pattern.fullmatch(normalized):
        kind = "canonical tool ID" if canonical else "tool name or alias"
        raise ToolNameError(
            ToolErrorCode.INVALID_NAME,
            f"invalid {kind}: {value!r}",
            requested_name=normalized or None,
            field_path=("tool_name",),
        )
    return normalized


def _freeze_json_value(
    value: Any,
    *,
    error_type: type[ToolKernelError],
    canonical_id: str | None = None,
    requested_name: str | None = None,
    field_path: tuple[str | int, ...] = (),
) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise error_type(
                ToolErrorCode.INVALID_ARGUMENT,
                "numbers must be finite",
                canonical_id=canonical_id,
                requested_name=requested_name,
                field_path=field_path,
            )
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise error_type(
                    ToolErrorCode.INVALID_ARGUMENT,
                    "object keys must be strings",
                    canonical_id=canonical_id,
                    requested_name=requested_name,
                    field_path=field_path,
                )
            frozen[key] = _freeze_json_value(
                nested_value,
                error_type=error_type,
                canonical_id=canonical_id,
                requested_name=requested_name,
                field_path=field_path + (key,),
            )
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json_value(
                nested_value,
                error_type=error_type,
                canonical_id=canonical_id,
                requested_name=requested_name,
                field_path=field_path + (index,),
            )
            for index, nested_value in enumerate(value)
        )
    raise error_type(
        ToolErrorCode.INVALID_ARGUMENT,
        f"unsupported JSON value type {type(value).__name__}",
        canonical_id=canonical_id,
        requested_name=requested_name,
        field_path=field_path,
    )


def _schema_error(message: str, path: tuple[str | int, ...]) -> ToolSchemaError:
    return ToolSchemaError(ToolErrorCode.INVALID_SCHEMA, message, field_path=path)


def _validate_schema_definition(schema: Any, path: tuple[str | int, ...] = ()) -> None:
    if not isinstance(schema, Mapping):
        raise _schema_error("schemas must be objects", path)
    if not schema:
        raise _schema_error("schemas must declare a type or enum", path)

    allowed = {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
    }
    unknown = set(schema) - allowed
    if unknown:
        raise _schema_error(f"unsupported schema keyword {sorted(unknown)[0]!r}", path)
    if any(not isinstance(key, str) for key in schema):
        raise _schema_error("schema keys must be strings", path)

    schema_type = schema.get("type")
    if schema_type is not None:
        if not isinstance(schema_type, str) or schema_type not in {
            "object",
            "string",
            "integer",
            "number",
            "boolean",
            "array",
            "null",
        }:
            raise _schema_error("unsupported schema type", path + ("type",))
    if schema_type is None and "enum" not in schema:
        raise _schema_error("schemas must declare a type or enum", path)

    object_keywords = {"properties", "required", "additionalProperties"}
    if object_keywords & set(schema) and schema_type != "object":
        raise _schema_error("object keywords require type 'object'", path)
    if "items" in schema and schema_type != "array":
        raise _schema_error("items requires type 'array'", path)

    if schema_type == "object":
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise _schema_error("properties must be an object", path + ("properties",))
        for property_name, property_schema in properties.items():
            if not isinstance(property_name, str) or not property_name:
                raise _schema_error("property names must be non-empty strings", path + ("properties",))
            _validate_schema_definition(
                property_schema, path + ("properties", property_name)
            )
        required = schema.get("required", ())
        if isinstance(required, str) or not isinstance(required, (list, tuple)):
            raise _schema_error("required must be an array of property names", path + ("required",))
        if len(required) != len(set(required)) or any(
            not isinstance(property_name, str) for property_name in required
        ):
            raise _schema_error("required must contain unique string property names", path + ("required",))
        if any(property_name not in properties for property_name in required):
            raise _schema_error("required properties must be declared", path + ("required",))
        additional = schema.get("additionalProperties", True)
        if not isinstance(additional, bool) and not isinstance(additional, Mapping):
            raise _schema_error(
                "additionalProperties must be a boolean or schema",
                path + ("additionalProperties",),
            )
        if isinstance(additional, Mapping):
            _validate_schema_definition(
                additional, path + ("additionalProperties",)
            )

    if schema_type == "array" and "items" in schema:
        _validate_schema_definition(schema["items"], path + ("items",))

    if "enum" in schema:
        values = schema["enum"]
        if isinstance(values, str) or not isinstance(values, (list, tuple)) or not values:
            raise _schema_error("enum must be a non-empty array", path + ("enum",))
        for index, value in enumerate(values):
            _freeze_json_value(
                value,
                error_type=ToolSchemaError,
                field_path=path + ("enum", index),
            )
            if schema_type is not None and not _matches_schema_type(value, schema_type):
                raise _schema_error(
                    "enum values must match the declared type",
                    path + ("enum", index),
                )


def validate_schema(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate and defensively freeze the supported schema subset."""

    _validate_schema_definition(schema)
    frozen = _freeze_json_value(schema, error_type=ToolSchemaError)
    assert isinstance(frozen, Mapping)
    return frozen


def _json_values_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return (
            set(left) == set(right)
            and all(_json_values_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _json_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def _matches_schema_type(value: Any, schema_type: str) -> bool:
    if schema_type == "object":
        return isinstance(value, Mapping)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "array":
        return isinstance(value, (list, tuple))
    return value is None


def _validate_schema_value(
    schema: Mapping[str, Any],
    value: Any,
    *,
    canonical_id: str | None,
    requested_name: str | None,
    field_path: tuple[str | int, ...],
) -> None:
    schema_type = schema.get("type")
    if schema_type is not None and not _matches_schema_type(value, schema_type):
        raise ToolArgumentError(
            ToolErrorCode.INVALID_ARGUMENT,
            f"expected {schema_type}",
            canonical_id=canonical_id,
            requested_name=requested_name,
            field_path=field_path,
        )

    if schema_type == "object":
        assert isinstance(value, Mapping)
        properties = schema.get("properties", {})
        required = schema.get("required", ())
        for property_name in required:
            if property_name not in value:
                raise ToolArgumentError(
                    ToolErrorCode.INVALID_ARGUMENT,
                    "required property is missing",
                    canonical_id=canonical_id,
                    requested_name=requested_name,
                    field_path=field_path + (property_name,),
                )
        additional = schema.get("additionalProperties", True)
        for property_name, property_value in value.items():
            if not isinstance(property_name, str):
                raise ToolArgumentError(
                    ToolErrorCode.INVALID_ARGUMENT,
                    "object keys must be strings",
                    canonical_id=canonical_id,
                    requested_name=requested_name,
                    field_path=field_path,
                )
            property_schema = properties.get(property_name)
            if property_schema is not None:
                _validate_schema_value(
                    property_schema,
                    property_value,
                    canonical_id=canonical_id,
                    requested_name=requested_name,
                    field_path=field_path + (property_name,),
                )
            elif additional is False:
                raise ToolArgumentError(
                    ToolErrorCode.INVALID_ARGUMENT,
                    "undeclared property is not allowed",
                    canonical_id=canonical_id,
                    requested_name=requested_name,
                    field_path=field_path + (property_name,),
                )
            elif isinstance(additional, Mapping):
                _validate_schema_value(
                    additional,
                    property_value,
                    canonical_id=canonical_id,
                    requested_name=requested_name,
                    field_path=field_path + (property_name,),
                )

    if schema_type == "array" and "items" in schema:
        assert isinstance(value, (list, tuple))
        for index, item in enumerate(value):
            _validate_schema_value(
                schema["items"],
                item,
                canonical_id=canonical_id,
                requested_name=requested_name,
                field_path=field_path + (index,),
            )

    if "enum" in schema and not any(
        _json_values_equal(value, candidate) for candidate in schema["enum"]
    ):
        raise ToolArgumentError(
            ToolErrorCode.INVALID_ARGUMENT,
            "value is not in the declared enum",
            canonical_id=canonical_id,
            requested_name=requested_name,
            field_path=field_path,
        )


def validate_schema_value(
    schema: Mapping[str, Any],
    value: Any,
    *,
    canonical_id: str | None = None,
    requested_name: str | None = None,
) -> Any:
    """Validate and freeze one JSON-compatible value against a declared schema."""

    schema = validate_schema(schema)
    _validate_schema_value(
        schema,
        value,
        canonical_id=canonical_id,
        requested_name=requested_name,
        field_path=(),
    )
    return _freeze_json_value(
        value,
        error_type=ToolArgumentError,
        canonical_id=canonical_id,
        requested_name=requested_name,
    )


def validate_arguments(
    schema: Mapping[str, Any],
    arguments: Mapping[str, Any],
    *,
    canonical_id: str | None = None,
    requested_name: str | None = None,
) -> Mapping[str, Any]:
    """Validate arguments and return an immutable deep copy of their JSON value."""

    schema = validate_schema(schema)
    if not isinstance(arguments, Mapping):
        raise ToolArgumentError(
            ToolErrorCode.INVALID_ARGUMENT,
            "tool arguments must be an object",
            canonical_id=canonical_id,
            requested_name=requested_name,
            field_path=("arguments",),
        )
    frozen = validate_schema_value(
        schema,
        arguments,
        canonical_id=canonical_id,
        requested_name=requested_name,
    )
    assert isinstance(frozen, Mapping)
    return frozen


def _coerce_enum(
    value: object,
    enum_type: type[Enum],
    field_name: str,
    canonical_id: str,
) -> Enum:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise ToolRegistryError(
            ToolErrorCode.INVALID_SPEC,
            f"{field_name} must be a supported string value",
            canonical_id=canonical_id,
            field_path=(field_name,),
        )
    try:
        return enum_type(value.strip().lower())
    except ValueError as exc:
        raise ToolRegistryError(
            ToolErrorCode.INVALID_SPEC,
            f"unsupported {field_name}: {value!r}",
            canonical_id=canonical_id,
            field_path=(field_name,),
        ) from exc


@dataclass(frozen=True)
class ToolSpec:
    """The canonical immutable declaration for one v2 tool."""

    canonical_id: str
    aliases: tuple[str, ...]
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    api_version: str
    schema_version: str
    capabilities: frozenset[str]
    confirmation: ConfirmationRequirement
    concurrency: ConcurrencyClass
    idempotency: IdempotencyClass
    migration_disposition: MigrationDisposition
    description: str = ""
    source_family: str = ""
    source_module: str = ""

    def __post_init__(self) -> None:
        canonical_id = normalize_tool_name(self.canonical_id, canonical=True)
        object.__setattr__(self, "canonical_id", canonical_id)
        if isinstance(self.aliases, str) or not isinstance(self.aliases, (list, tuple)):
            raise ToolRegistryError(
                ToolErrorCode.INVALID_SPEC,
                "aliases must be an array of declared names",
                canonical_id=canonical_id,
                field_path=("aliases",),
            )
        aliases = tuple(normalize_tool_name(alias) for alias in self.aliases)
        if canonical_id in aliases or len(aliases) != len(set(aliases)):
            raise ToolRegistryError(
                ToolErrorCode.DUPLICATE_ALIAS,
                "aliases must be unique and cannot repeat the canonical ID",
                canonical_id=canonical_id,
                field_path=("aliases",),
            )
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "input_schema", validate_schema(self.input_schema))
        object.__setattr__(self, "output_schema", validate_schema(self.output_schema))
        object.__setattr__(self, "api_version", _required_text(self.api_version, "api_version", ToolRegistryError))
        object.__setattr__(self, "schema_version", _required_text(self.schema_version, "schema_version", ToolRegistryError))

        if isinstance(self.capabilities, str) or not isinstance(
            self.capabilities, (set, frozenset, list, tuple)
        ):
            raise ToolRegistryError(
                ToolErrorCode.INVALID_SPEC,
                "capabilities must be a non-empty set of names",
                canonical_id=canonical_id,
                field_path=("capabilities",),
            )
        capabilities = frozenset(self.capabilities)
        if not capabilities or any(
            not isinstance(capability, str)
            or not _CAPABILITY_RE.fullmatch(capability.strip().lower())
            for capability in capabilities
        ):
            raise ToolRegistryError(
                ToolErrorCode.INVALID_SPEC,
                "capabilities must be non-empty normalized names",
                canonical_id=canonical_id,
                field_path=("capabilities",),
            )
        object.__setattr__(
            self,
            "capabilities",
            frozenset(capability.strip().lower() for capability in capabilities),
        )
        object.__setattr__(
            self,
            "confirmation",
            _coerce_enum(
                self.confirmation,
                ConfirmationRequirement,
                "confirmation",
                canonical_id,
            ),
        )
        object.__setattr__(
            self,
            "concurrency",
            _coerce_enum(self.concurrency, ConcurrencyClass, "concurrency", canonical_id),
        )
        object.__setattr__(
            self,
            "idempotency",
            _coerce_enum(self.idempotency, IdempotencyClass, "idempotency", canonical_id),
        )
        object.__setattr__(
            self,
            "migration_disposition",
            _coerce_enum(
                self.migration_disposition,
                MigrationDisposition,
                "migration_disposition",
                canonical_id,
            ),
        )
        for field_name in ("description", "source_family", "source_module"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise ToolRegistryError(
                    ToolErrorCode.INVALID_SPEC,
                    f"{field_name} must be a string",
                    canonical_id=canonical_id,
                    field_path=(field_name,),
                )
            object.__setattr__(self, field_name, value.strip())

    @property
    def name(self) -> str:
        """Compatibility spelling for consumers that call a spec's ID its name."""

        return self.canonical_id


@dataclass(frozen=True)
class ToolContext:
    """Pure call context passed between later policy and execution layers."""

    correlation_id: str
    actor_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "correlation_id", _required_text(self.correlation_id, "correlation_id", ToolRegistryError)
        )
        if self.actor_id is not None:
            object.__setattr__(self, "actor_id", _required_text(self.actor_id, "actor_id", ToolRegistryError))
        frozen = _freeze_json_value(self.metadata, error_type=ToolRegistryError)
        if not isinstance(frozen, Mapping):
            raise ToolRegistryError(
                ToolErrorCode.INVALID_SPEC,
                "context metadata must be an object",
                field_path=("metadata",),
            )
        object.__setattr__(self, "metadata", frozen)


@dataclass(frozen=True)
class ToolCall:
    """A schema-validated, immutable request for one resolved tool spec."""

    call_id: str
    spec: ToolSpec
    requested_name: str
    arguments: Mapping[str, Any]
    context: ToolContext | None = None
    api_version: str = TOOL_API_VERSION
    schema_version: str = TOOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "call_id", _required_text(self.call_id, "call_id", ToolRegistryError))
        if not isinstance(self.spec, ToolSpec):
            raise ToolRegistryError(
                ToolErrorCode.INVALID_CALL,
                "calls require a ToolSpec",
                field_path=("spec",),
            )
        requested_name = normalize_tool_name(self.requested_name)
        if requested_name not in {self.spec.canonical_id, *self.spec.aliases}:
            raise ToolUndeclaredAliasError(
                ToolErrorCode.UNDECLARED_ALIAS,
                "requested name is not declared by the tool spec",
                canonical_id=self.spec.canonical_id,
                requested_name=requested_name,
                field_path=("requested_name",),
            )
        object.__setattr__(self, "requested_name", requested_name)
        api_version = _required_text(self.api_version, "api_version", ToolRegistryError)
        schema_version = _required_text(self.schema_version, "schema_version", ToolRegistryError)
        if api_version != self.spec.api_version:
            raise ToolVersionError(
                ToolErrorCode.API_VERSION_MISMATCH,
                "call API version does not match the tool spec",
                canonical_id=self.spec.canonical_id,
                requested_name=requested_name,
                field_path=("api_version",),
            )
        if schema_version != self.spec.schema_version:
            raise ToolVersionError(
                ToolErrorCode.SCHEMA_VERSION_MISMATCH,
                "call schema version does not match the tool spec",
                canonical_id=self.spec.canonical_id,
                requested_name=requested_name,
                field_path=("schema_version",),
            )
        object.__setattr__(self, "api_version", api_version)
        object.__setattr__(self, "schema_version", schema_version)
        if self.context is not None and not isinstance(self.context, ToolContext):
            raise ToolRegistryError(
                ToolErrorCode.INVALID_CALL,
                "call context must be a ToolContext",
                canonical_id=self.spec.canonical_id,
                requested_name=requested_name,
                field_path=("context",),
            )
        object.__setattr__(
            self,
            "arguments",
            validate_arguments(
                self.spec.input_schema,
                self.arguments,
                canonical_id=self.spec.canonical_id,
                requested_name=requested_name,
            ),
        )

    @property
    def canonical_id(self) -> str:
        return self.spec.canonical_id

    @property
    def tool_name(self) -> str:
        return self.spec.canonical_id


@dataclass(frozen=True)
class ToolTraceEvent:
    """One in-memory trace transition with no side effects."""

    state: ToolTraceState
    timestamp: datetime
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", ToolTraceState(self.state))
        if not isinstance(self.timestamp, datetime):
            raise TypeError("trace timestamps must be datetimes")
        frozen = _freeze_json_value(self.details, error_type=ToolRegistryError)
        if not isinstance(frozen, Mapping):
            raise ToolRegistryError(
                ToolErrorCode.INVALID_SPEC,
                "trace event details must be an object",
                field_path=("details",),
            )
        object.__setattr__(self, "details", frozen)


@dataclass(frozen=True)
class ToolTrace:
    """Immutable correlation, call, state, timestamps, and trace events."""

    call_id: str
    correlation_id: str
    state: ToolTraceState
    created_at: datetime
    updated_at: datetime
    events: tuple[ToolTraceEvent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "call_id", _required_text(self.call_id, "call_id", ToolRegistryError))
        object.__setattr__(
            self, "correlation_id", _required_text(self.correlation_id, "correlation_id", ToolRegistryError)
        )
        object.__setattr__(self, "state", ToolTraceState(self.state))
        if not isinstance(self.created_at, datetime) or not isinstance(self.updated_at, datetime):
            raise TypeError("trace timestamps must be datetimes")
        if self.updated_at < self.created_at:
            raise ToolRegistryError(
                ToolErrorCode.INVALID_SPEC,
                "trace updated_at cannot precede created_at",
                field_path=("updated_at",),
            )
        events = tuple(self.events)
        if any(not isinstance(event, ToolTraceEvent) for event in events):
            raise ToolRegistryError(
                ToolErrorCode.INVALID_SPEC,
                "trace events must be ToolTraceEvent instances",
                field_path=("events",),
            )
        object.__setattr__(self, "events", events)

    @classmethod
    def created(cls, call: ToolCall, now: datetime | None = None) -> ToolTrace:
        timestamp = now or datetime.now(timezone.utc)
        return cls(
            call_id=call.call_id,
            correlation_id=call.context.correlation_id if call.context else call.call_id,
            state=ToolTraceState.CREATED,
            created_at=timestamp,
            updated_at=timestamp,
        )


@dataclass(frozen=True)
class ToolResult:
    """A terminal outcome.  ``retryable`` is data only; no retry logic lives here."""

    call_id: str
    status: ToolResultStatus
    output: Any = None
    error: ToolError | ToolKernelError | None = None
    retryable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "call_id", _required_text(self.call_id, "call_id", ToolRegistryError))
        object.__setattr__(self, "status", ToolResultStatus(self.status))
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be a boolean")
        object.__setattr__(
            self, "output", _freeze_json_value(self.output, error_type=ToolRegistryError)
        )
        error = self.error.details if isinstance(self.error, ToolKernelError) else self.error
        if error is not None and not isinstance(error, ToolError):
            raise TypeError("result errors must be ToolError or ToolKernelError")
        if self.status is ToolResultStatus.SUCCESS and error is not None:
            raise ToolRegistryError(
                ToolErrorCode.INVALID_SPEC,
                "successful results cannot carry an error",
                field_path=("error",),
            )
        if self.status is not ToolResultStatus.SUCCESS and error is None:
            raise ToolRegistryError(
                ToolErrorCode.INVALID_SPEC,
                "non-successful results require an error",
                field_path=("error",),
            )
        object.__setattr__(self, "error", error)


class ToolRegistry:
    """A functional immutable registry of canonical specs and declared aliases."""

    def __init__(
        self,
        specs: Iterable[ToolSpec] = (),
        *,
        api_version: str = TOOL_API_VERSION,
        schema_version: str = TOOL_SCHEMA_VERSION,
    ) -> None:
        self._api_version = _required_text(api_version, "api_version", ToolRegistryError)
        self._schema_version = _required_text(schema_version, "schema_version", ToolRegistryError)
        collected = tuple(specs)
        if any(not isinstance(spec, ToolSpec) for spec in collected):
            raise ToolRegistryError(
                ToolErrorCode.INVALID_SPEC,
                "registries contain only ToolSpec values",
                field_path=("specs",),
            )
        sorted_specs = tuple(sorted(collected, key=lambda spec: spec.canonical_id))
        by_id: dict[str, ToolSpec] = {}
        for spec in sorted_specs:
            if spec.api_version != self._api_version:
                raise ToolVersionError(
                    ToolErrorCode.API_VERSION_MISMATCH,
                    "tool spec API version does not match the registry",
                    canonical_id=spec.canonical_id,
                    field_path=("api_version",),
                )
            if spec.schema_version != self._schema_version:
                raise ToolVersionError(
                    ToolErrorCode.SCHEMA_VERSION_MISMATCH,
                    "tool spec schema version does not match the registry",
                    canonical_id=spec.canonical_id,
                    field_path=("schema_version",),
                )
            if spec.canonical_id in by_id:
                raise ToolRegistryError(
                    ToolErrorCode.DUPLICATE_TOOL,
                    "duplicate canonical tool ID",
                    canonical_id=spec.canonical_id,
                    field_path=("canonical_id",),
                )
            by_id[spec.canonical_id] = spec

        aliases: dict[str, ToolSpec] = {}
        for spec in sorted_specs:
            for alias in spec.aliases:
                if alias in by_id:
                    raise ToolRegistryError(
                        ToolErrorCode.DUPLICATE_ALIAS,
                        "an alias collides with a canonical tool ID",
                        canonical_id=spec.canonical_id,
                        requested_name=alias,
                        field_path=("aliases",),
                    )
                if alias in aliases:
                    raise ToolRegistryError(
                        ToolErrorCode.DUPLICATE_ALIAS,
                        "duplicate alias declared by multiple tool specs",
                        canonical_id=spec.canonical_id,
                        requested_name=alias,
                        field_path=("aliases",),
                    )
                aliases[alias] = spec

        names = dict(by_id)
        names.update(aliases)
        self._specs = sorted_specs
        self._by_id = MappingProxyType(by_id)
        self._names = MappingProxyType(names)

    @property
    def api_version(self) -> str:
        return self._api_version

    @property
    def schema_version(self) -> str:
        return self._schema_version

    def specs(self) -> tuple[ToolSpec, ...]:
        """Enumerate canonical specs in lexical canonical-ID order."""

        return self._specs

    def register(self, spec: ToolSpec) -> ToolRegistry:
        """Return a new registry containing ``spec`` without mutating this one."""

        return ToolRegistry(
            (*self._specs, spec),
            api_version=self._api_version,
            schema_version=self._schema_version,
        )

    def resolve(self, requested_name: str) -> ToolSpec:
        normalized = normalize_tool_name(requested_name)
        spec = self._names.get(normalized)
        if spec is None:
            raise ToolUndeclaredAliasError(
                ToolErrorCode.UNDECLARED_ALIAS,
                "tool name is not a canonical ID or declared alias",
                requested_name=normalized,
                field_path=("requested_name",),
            )
        return spec

    def create_call(
        self,
        *,
        call_id: str,
        requested_name: str,
        arguments: Mapping[str, Any],
        context: ToolContext | None = None,
        api_version: str | None = None,
        schema_version: str | None = None,
    ) -> ToolCall:
        spec = self.resolve(requested_name)
        return ToolCall(
            call_id=call_id,
            spec=spec,
            requested_name=requested_name,
            arguments=arguments,
            context=context,
            api_version=api_version if api_version is not None else self._api_version,
            schema_version=(
                schema_version if schema_version is not None else self._schema_version
            ),
        )

    def validate_call(self, call: ToolCall) -> ToolCall:
        if not isinstance(call, ToolCall):
            raise ToolRegistryError(
                ToolErrorCode.INVALID_CALL,
                "registry accepts only ToolCall values",
                field_path=("call",),
            )
        resolved = self.resolve(call.requested_name)
        if resolved is not call.spec:
            raise ToolRegistryError(
                ToolErrorCode.INVALID_CALL,
                "call spec is not the resolved registry spec",
                canonical_id=resolved.canonical_id,
                requested_name=call.requested_name,
                field_path=("spec",),
            )
        return call


DeterministicToolRegistry = ToolRegistry


def _default_compatibility_matrix() -> tuple[Any, ...]:
    try:
        from .ToolCompatibility import TOOL_COMPATIBILITY_MATRIX
    except ImportError:
        from ToolCompatibility import TOOL_COMPATIBILITY_MATRIX

    return tuple(TOOL_COMPATIBILITY_MATRIX)


class SystemToolAdapter:
    """Convert a ``SystemTool``-shaped descriptor using frozen compatibility data."""

    def __init__(self, compatibility_matrix: Iterable[Any] | None = None) -> None:
        entries = tuple(
            _default_compatibility_matrix()
            if compatibility_matrix is None
            else compatibility_matrix
        )
        self._entries = MappingProxyType(
            {
                normalize_tool_name(entry.canonical_id, canonical=True): entry
                for entry in entries
                if getattr(entry, "source_family", None) == "system"
            }
        )

    @staticmethod
    def _descriptor_id(descriptor: Any) -> str:
        tool_class = getattr(descriptor, "tool_class", None)
        name = getattr(descriptor, "name", None)
        if not isinstance(tool_class, str) or not isinstance(name, str):
            raise SystemToolAdapterError(
                ToolErrorCode.ADAPTER_DRIFT,
                "SystemTool descriptors require string tool_class and name",
                field_path=("descriptor",),
            )
        return normalize_tool_name(f"{tool_class.strip().lower()}.{name.strip().lower()}", canonical=True)

    def to_spec(self, descriptor: Any) -> ToolSpec:
        canonical_id = self._descriptor_id(descriptor)
        entry = self._entries.get(canonical_id)
        if entry is None:
            raise SystemToolAdapterError(
                ToolErrorCode.ADAPTER_DRIFT,
                "no committed compatibility entry exists for the SystemTool",
                canonical_id=canonical_id,
                field_path=("canonical_id",),
            )
        descriptor_aliases = getattr(descriptor, "aliases", ())
        if isinstance(descriptor_aliases, str) or not isinstance(
            descriptor_aliases, (list, tuple)
        ):
            raise SystemToolAdapterError(
                ToolErrorCode.ADAPTER_DRIFT,
                "SystemTool aliases must be an array",
                canonical_id=canonical_id,
                field_path=("aliases",),
            )
        aliases = tuple(normalize_tool_name(alias) for alias in descriptor_aliases)
        expected_aliases = tuple(normalize_tool_name(alias) for alias in entry.aliases)
        if aliases != expected_aliases:
            raise SystemToolAdapterError(
                ToolErrorCode.ADAPTER_DRIFT,
                "SystemTool aliases drift from the committed compatibility entry",
                canonical_id=canonical_id,
                field_path=("aliases",),
            )
        descriptor_version = getattr(descriptor, "api_version", None)
        if not isinstance(descriptor_version, str) or descriptor_version.strip() != "1":
            raise ToolVersionError(
                ToolErrorCode.API_VERSION_MISMATCH,
                "SystemTool descriptors must use legacy API version 1",
                canonical_id=canonical_id,
                field_path=("api_version",),
            )
        docs = getattr(descriptor, "docs", None)
        if not isinstance(docs, Mapping) or not isinstance(docs.get("desc"), str):
            raise SystemToolAdapterError(
                ToolErrorCode.ADAPTER_DRIFT,
                "SystemTool docs.desc is required",
                canonical_id=canonical_id,
                field_path=("docs", "desc"),
            )
        input_schema = getattr(descriptor, "input_schema", None)
        output_schema = getattr(descriptor, "output_schema", None)
        if not isinstance(input_schema, Mapping) or not isinstance(output_schema, Mapping):
            raise SystemToolAdapterError(
                ToolErrorCode.ADAPTER_DRIFT,
                "SystemTool schemas must be objects",
                canonical_id=canonical_id,
                field_path=("input_schema",),
            )
        return ToolSpec(
            canonical_id=canonical_id,
            aliases=aliases,
            input_schema=(
                input_schema
                if input_schema
                else {"type": "object", "additionalProperties": True}
            ),
            output_schema=(
                output_schema
                if output_schema
                else {"type": "object", "additionalProperties": True}
            ),
            api_version=TOOL_API_VERSION,
            schema_version=TOOL_SCHEMA_VERSION,
            capabilities=frozenset({entry.capability_class}),
            confirmation=entry.confirmation_class,
            concurrency=entry.concurrency_class,
            idempotency=entry.idempotency_class,
            migration_disposition=entry.migration_disposition,
            description=docs["desc"],
            source_family=entry.source_family,
            source_module=entry.source_module,
        )

    def to_specs(self, descriptors: Mapping[str, Any]) -> tuple[ToolSpec, ...]:
        """Adapt an alias-indexed legacy discovery map into canonical v2 specs."""

        if not isinstance(descriptors, Mapping):
            raise SystemToolAdapterError(
                ToolErrorCode.ADAPTER_DRIFT,
                "SystemTool discovery output must be a mapping",
                field_path=("descriptors",),
            )
        specs_by_id: dict[str, ToolSpec] = {}
        for mapping_key, descriptor in descriptors.items():
            try:
                requested_name = normalize_tool_name(mapping_key)
            except ToolNameError as exc:
                raise SystemToolAdapterError(
                    ToolErrorCode.ADAPTER_DRIFT,
                    "SystemTool discovery keys must be canonical IDs or declared aliases",
                    requested_name=mapping_key if isinstance(mapping_key, str) else None,
                    field_path=("descriptors",),
                ) from exc
            spec = self.to_spec(descriptor)
            if requested_name not in {spec.canonical_id, *spec.aliases}:
                raise SystemToolAdapterError(
                    ToolErrorCode.ADAPTER_DRIFT,
                    "SystemTool discovery key is not declared by its descriptor",
                    canonical_id=spec.canonical_id,
                    requested_name=requested_name,
                    field_path=("descriptors", requested_name),
                )
            existing = specs_by_id.get(spec.canonical_id)
            if existing is not None and existing != spec:
                raise SystemToolAdapterError(
                    ToolErrorCode.ADAPTER_DRIFT,
                    "conflicting SystemTool descriptors claim one canonical ID",
                    canonical_id=spec.canonical_id,
                    requested_name=requested_name,
                    field_path=("descriptors", requested_name),
                )
            specs_by_id.setdefault(spec.canonical_id, spec)
        return tuple(specs_by_id[canonical_id] for canonical_id in sorted(specs_by_id))

    def to_registry(self, descriptors: Mapping[str, Any]) -> ToolRegistry:
        """Build a fully validated immutable v2 registry from discovery output."""

        return ToolRegistry(self.to_specs(descriptors))


def system_tool_to_spec(
    descriptor: Any, compatibility_matrix: Iterable[Any] | None = None
) -> ToolSpec:
    """Adapt one descriptor without resolving or invoking its handler."""

    return SystemToolAdapter(compatibility_matrix).to_spec(descriptor)


adapt_system_tool = system_tool_to_spec


__all__ = [
    "ConfirmationRequirement",
    "ConcurrencyClass",
    "DeterministicToolRegistry",
    "IdempotencyClass",
    "MigrationDisposition",
    "SystemToolAdapter",
    "SystemToolAdapterError",
    "TOOL_API_VERSION",
    "TOOL_SCHEMA_VERSION",
    "ToolArgumentError",
    "ToolCall",
    "ToolContext",
    "ToolError",
    "ToolErrorCode",
    "ToolKernelError",
    "ToolNameError",
    "ToolRegistry",
    "ToolRegistryError",
    "ToolResult",
    "ToolResultStatus",
    "ToolSchemaError",
    "ToolSpec",
    "ToolTrace",
    "ToolTraceEvent",
    "ToolTraceState",
    "ToolUndeclaredAliasError",
    "ToolVersionError",
    "adapt_system_tool",
    "normalize_tool_name",
    "system_tool_to_spec",
    "validate_arguments",
    "validate_schema",
]
