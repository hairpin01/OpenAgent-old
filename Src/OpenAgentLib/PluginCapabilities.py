# SPDX-License-Identifier: MIT
"""Parent-side, fail-closed broker for narrow v2 plugin capabilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import ipaddress
from math import isfinite
from pathlib import Path, PurePosixPath
import socket
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlparse

from .PluginSDK import CapabilityFamily, PLUGIN_SDK_API_VERSION, thaw_json
from .ToolKernel import ToolCall, normalize_tool_name
from .ToolPolicy import PolicyDecisionKind, ToolPolicyEngine, ToolPolicyRequest, tool_scope_for


class CapabilityErrorCode(str, Enum):
    DENIED = "denied"
    INVALID_FRAME = "invalid-frame"
    INVALID_GRANT = "invalid-grant"
    REPLAYED = "replayed"
    UNKNOWN_CAPABILITY = "unknown-capability"
    UNKNOWN_OPERATION = "unknown-operation"
    INVALID_REQUEST = "invalid-request"
    BACKEND_ERROR = "backend-error"


class CapabilityProtocolError(ValueError):
    """A capability frame violates the versioned parent-child protocol."""


def _required(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CapabilityProtocolError(f"{field_name} must be a non-empty string")
    return value.strip()


def _json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise CapabilityProtocolError("numbers must be finite")
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _json(nested) for key, nested in value.items() if isinstance(key, str)}) if all(isinstance(key, str) for key in value) else (_ for _ in ()).throw(CapabilityProtocolError("JSON keys must be strings"))
    if isinstance(value, (list, tuple)):
        return tuple(_json(item) for item in value)
    raise CapabilityProtocolError("values must be JSON-compatible")


def _scope(value: object) -> str:
    return _required(value, "actor_scope")


@dataclass(frozen=True)
class CapabilityGrant:
    """An immutable parent-issued grant for exactly one active tool call."""

    grant_id: str
    host_request_id: str
    call_id: str
    canonical_tool_id: str
    actor_scope: str
    capability: CapabilityFamily
    operations: frozenset[str]
    constraints: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("grant_id", "host_request_id", "call_id"):
            object.__setattr__(self, field_name, _required(getattr(self, field_name), field_name))
        object.__setattr__(self, "canonical_tool_id", normalize_tool_name(self.canonical_tool_id, canonical=True))
        object.__setattr__(self, "actor_scope", _scope(self.actor_scope))
        object.__setattr__(self, "capability", CapabilityFamily(self.capability))
        operations = frozenset(_required(operation, "operation").lower() for operation in self.operations)
        if not operations:
            raise CapabilityProtocolError("grants require at least one operation")
        object.__setattr__(self, "operations", operations)
        constraints = _json(self.constraints)
        if not isinstance(constraints, Mapping):
            raise CapabilityProtocolError("grant constraints must be an object")
        object.__setattr__(self, "constraints", constraints)

    @classmethod
    def for_call(
        cls, grant_id: str, host_request_id: str, call: ToolCall, capability: CapabilityFamily,
        operations: frozenset[str], constraints: Mapping[str, Any] | None = None,
    ) -> "CapabilityGrant":
        return cls(grant_id, host_request_id, call.call_id, call.spec.canonical_id, tool_scope_for(call), capability, operations, constraints or {})


@dataclass(frozen=True)
class CapabilityRequest:
    """One correlated child request, with no avenue for ambient object transfer."""

    host_request_id: str
    call_id: str
    canonical_tool_id: str
    actor_scope: str
    grant_id: str
    capability: CapabilityFamily
    operation: str
    capability_request_id: str
    payload: Mapping[str, Any]
    version: str = PLUGIN_SDK_API_VERSION

    def __post_init__(self) -> None:
        if self.version != PLUGIN_SDK_API_VERSION:
            raise CapabilityProtocolError("unsupported capability protocol version")
        for field_name in ("host_request_id", "call_id", "grant_id", "capability_request_id"):
            object.__setattr__(self, field_name, _required(getattr(self, field_name), field_name))
        object.__setattr__(self, "canonical_tool_id", normalize_tool_name(self.canonical_tool_id, canonical=True))
        object.__setattr__(self, "actor_scope", _scope(self.actor_scope))
        object.__setattr__(self, "capability", CapabilityFamily(self.capability))
        object.__setattr__(self, "operation", _required(self.operation, "operation").lower())
        payload = _json(self.payload)
        if not isinstance(payload, Mapping):
            raise CapabilityProtocolError("capability payload must be an object")
        object.__setattr__(self, "payload", payload)

    def to_envelope(self) -> dict[str, Any]:
        return {
            "version": self.version, "kind": "capability-request", "host_request_id": self.host_request_id,
            "call_id": self.call_id, "canonical_tool_id": self.canonical_tool_id,
            "actor_scope": self.actor_scope, "grant_id": self.grant_id,
            "capability": self.capability.value, "operation": self.operation,
            "capability_request_id": self.capability_request_id, "payload": thaw_json(self.payload),
        }

    @classmethod
    def from_envelope(cls, value: Any) -> "CapabilityRequest":
        expected = {"version", "kind", "host_request_id", "call_id", "canonical_tool_id", "actor_scope", "grant_id", "capability", "operation", "capability_request_id", "payload"}
        if not isinstance(value, Mapping) or set(value) != expected or value.get("kind") != "capability-request":
            raise CapabilityProtocolError("capability request has unknown or missing fields")
        return cls(**{key: value[key] for key in expected if key != "kind"})


@dataclass(frozen=True)
class CapabilityResponse:
    host_request_id: str
    call_id: str
    capability_request_id: str
    ok: bool
    data: Mapping[str, Any] | None = None
    error: CapabilityErrorCode | None = None

    def __post_init__(self) -> None:
        for field_name in ("host_request_id", "call_id", "capability_request_id"):
            object.__setattr__(self, field_name, _required(getattr(self, field_name), field_name))
        if not isinstance(self.ok, bool):
            raise TypeError("capability response ok must be bool")
        if self.ok == (self.error is not None):
            raise CapabilityProtocolError("responses require exactly success data or an error")
        data = _json(self.data or {})
        if not isinstance(data, Mapping):
            raise CapabilityProtocolError("response data must be an object")
        object.__setattr__(self, "data", data)
        if self.error is not None:
            object.__setattr__(self, "error", CapabilityErrorCode(self.error))

    @classmethod
    def denied(cls, request: CapabilityRequest, error: CapabilityErrorCode) -> "CapabilityResponse":
        return cls(request.host_request_id, request.call_id, request.capability_request_id, False, {}, error)

    def to_envelope(self) -> dict[str, Any]:
        return {"version": PLUGIN_SDK_API_VERSION, "kind": "capability-response", "host_request_id": self.host_request_id, "call_id": self.call_id, "capability_request_id": self.capability_request_id, "ok": self.ok, "data": thaw_json(self.data), "error": self.error.value if self.error else None}


class CapabilityBackend(Protocol):
    def invoke(self, operation: str, payload: Mapping[str, Any], grant: CapabilityGrant) -> Mapping[str, Any]: ...


class TelegramBackend(CapabilityBackend, Protocol):
    """Receives opaque Telegram IDs/data only, never a client or session."""


class WorkspaceFilesystemBackend(CapabilityBackend, Protocol):
    """Must enforce normalized paths and atomically compare any expected_hash."""


class ProcessBackend(CapabilityBackend, Protocol):
    """Executes validated argv arrays without a shell."""


class PublicHttpsBackend(CapabilityBackend, Protocol):
    """Must call ``validate_public_https_url`` for the initial URL and redirects."""


class SchedulingBackend(CapabilityBackend, Protocol):
    """Schedules only the canonical reduced-budget child call data."""


class ConfigurationBackend(CapabilityBackend, Protocol):
    """Reads and writes only grant-namespaced JSON settings."""


def resolve_workspace_path(root: str | Path, relative_path: str) -> Path:
    """Resolve a grant-relative path and reject traversal or symlink escape."""

    relative = _relative_path(relative_path)
    return _resolve_workspace_path(root, relative)


def _resolve_workspace_path(root: str | Path, relative: str) -> Path:
    try:
        resolved_root = Path(root).resolve(strict=True)
        resolved_path = (resolved_root / relative).resolve(strict=False)
        resolved_path.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise CapabilityProtocolError("filesystem path escapes its granted root") from exc
    return resolved_path


def resolve_workspace_directory(root: str | Path, relative_path: str) -> Path:
    """Resolve an explicit grant-relative directory, including the grant root."""

    relative = _relative_directory(relative_path)
    if relative == ".":
        try:
            return Path(root).resolve(strict=True)
        except OSError as exc:
            raise CapabilityProtocolError("filesystem root is unavailable") from exc
    return _resolve_workspace_path(root, relative)


def validate_public_https_url(url: str, resolver: Any = socket.getaddrinfo) -> str:
    """Validate HTTPS and every DNS result; call again after each redirect."""

    parsed = urlparse(_required(url, "url"))
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise CapabilityProtocolError("only credential-free HTTPS URLs are allowed")
    try:
        records = resolver(parsed.hostname, 443, type=socket.SOCK_STREAM)
    except (OSError, socket.gaierror) as exc:
        raise CapabilityProtocolError("could not resolve HTTPS host") from exc
    if not records:
        raise CapabilityProtocolError("HTTPS host has no addresses")
    for record in records:
        address = ipaddress.ip_address(record[4][0])
        if not address.is_global:
            raise CapabilityProtocolError("HTTPS host resolves to a non-public address")
    return parsed.geturl()


class CapabilityBroker:
    """Checks policy, exact grant binding, schemas, and replay before invoking fakes/backends."""

    def __init__(self, policy: ToolPolicyEngine, backends: Mapping[CapabilityFamily, CapabilityBackend], *, resolver: Callable[..., Any] = socket.getaddrinfo) -> None:
        self._policy = policy
        self._backends = MappingProxyType(dict(backends))
        self._resolver = resolver
        self._used_request_ids: set[tuple[str, str]] = set()
        self._scheduled_child_ids: set[str] = set()

    def dispatch(self, call: ToolCall, policy_request: ToolPolicyRequest, grant: CapabilityGrant, request: CapabilityRequest) -> CapabilityResponse:
        if self._policy.evaluate(call, policy_request).kind is not PolicyDecisionKind.ALLOW:
            return CapabilityResponse.denied(request, CapabilityErrorCode.DENIED)
        if not self._grant_matches(call, grant, request):
            return CapabilityResponse.denied(request, CapabilityErrorCode.INVALID_GRANT)
        replay_key = (grant.grant_id, request.capability_request_id)
        if replay_key in self._used_request_ids:
            return CapabilityResponse.denied(request, CapabilityErrorCode.REPLAYED)
        if request.operation not in grant.operations or not _operation_allowed(request.capability, request.operation):
            return CapabilityResponse.denied(request, CapabilityErrorCode.UNKNOWN_OPERATION)
        try:
            normalized_payload = _normalize_payload(request, grant, self._resolver)
        except CapabilityProtocolError:
            return CapabilityResponse.denied(request, CapabilityErrorCode.INVALID_REQUEST)
        if request.capability is CapabilityFamily.SCHEDULING:
            child_id = normalized_payload["call_id"]
            if child_id in self._scheduled_child_ids:
                return CapabilityResponse.denied(request, CapabilityErrorCode.INVALID_REQUEST)
        backend = self._backends.get(request.capability)
        if backend is None:
            return CapabilityResponse.denied(request, CapabilityErrorCode.UNKNOWN_CAPABILITY)
        self._used_request_ids.add(replay_key)
        if request.capability is CapabilityFamily.SCHEDULING:
            self._scheduled_child_ids.add(normalized_payload["call_id"])
        try:
            data = _json(backend.invoke(request.operation, normalized_payload, grant))
            if not isinstance(data, Mapping):
                raise CapabilityProtocolError("backend result must be an object")
            for redirect in data.get("redirect_urls", ()):
                validate_public_https_url(redirect, self._resolver)
        except (CapabilityProtocolError, TypeError, ValueError, OSError, socket.gaierror):
            return CapabilityResponse.denied(request, CapabilityErrorCode.BACKEND_ERROR)
        return CapabilityResponse(request.host_request_id, request.call_id, request.capability_request_id, True, data)

    @staticmethod
    def _grant_matches(call: ToolCall, grant: CapabilityGrant, request: CapabilityRequest) -> bool:
        return (
            grant.grant_id == request.grant_id and grant.host_request_id == request.host_request_id
            and grant.call_id == call.call_id == request.call_id
            and grant.canonical_tool_id == call.spec.canonical_id == request.canonical_tool_id
            and grant.actor_scope == tool_scope_for(call) == request.actor_scope
            and grant.capability is request.capability
        )


_TELEGRAM_OPERATION_FIELDS = MappingProxyType({
    # The broker accepts only these named operations and opaque JSON reference
    # fields.  Plugins never get a Telegram client, peer resolver, or session.
    "get-message": frozenset({"peer_id", "message_id"}),
    "send-message": frozenset({"peer_id", "text", "reply_to_message_id"}),
    "edit-message": frozenset({"peer_id", "message_id", "text"}),
    "delete-message": frozenset({"peer_id", "message_id"}),
    "react": frozenset({"peer_id", "message_id", "reaction"}),
    "chat-info": frozenset({"peer_id"}), "chat-participants": frozenset({"peer_id", "limit"}),
    "chat-admins": frozenset({"peer_id"}), "chat-permissions": frozenset({"peer_id"}),
    "chat-common-with-user": frozenset({"user_id", "limit"}), "chat-set-title": frozenset({"peer_id", "title"}),
    "chat-set-about": frozenset({"peer_id", "about"}), "chat-set-username": frozenset({"peer_id", "username"}),
    "chat-slowmode": frozenset({"peer_id", "seconds"}), "chat-invite-link": frozenset({"peer_id"}),
    "contacts-add": frozenset({"user_id", "first_name", "last_name", "phone"}),
    "contacts-delete": frozenset({"user_id"}), "contacts-block": frozenset({"user_id"}),
    "contacts-unblock": frozenset({"user_id"}), "contacts-entity": frozenset({"user_id"}),
    "creation-channel": frozenset({"title", "about"}), "creation-group": frozenset({"title", "about"}),
    "creation-bot": frozenset({"name", "username", "about"}), "creation-private-invite": frozenset({"invite_id"}),
    "dialog-list-private": frozenset({"limit"}), "dialog-list-groups": frozenset({"limit"}),
    "dialog-list-all": frozenset({"limit"}), "dialog-search": frozenset({"query", "limit"}),
    "dialog-archive": frozenset({"peer_id"}), "dialog-unarchive": frozenset({"peer_id"}),
    "dialog-leave": frozenset({"peer_id"}), "dialog-export-invite": frozenset({"peer_id"}),
    "dialog-get-photo": frozenset({"peer_id"}), "dialog-set-photo": frozenset({"peer_id", "media_id"}),
    "message-reply": frozenset({"peer_id", "message_id", "text"}),
    "message-forward": frozenset({"peer_id", "message_id", "destination_peer_id"}),
    "message-pin": frozenset({"peer_id", "message_id"}), "message-search": frozenset({"peer_id", "query", "limit"}),
    "message-history": frozenset({"peer_id", "limit"}), "message-mark-read": frozenset({"peer_id", "message_id"}),
    "message-typing": frozenset({"peer_id"}), "message-schedule": frozenset({"peer_id", "text", "schedule_at"}),
    "message-draft": frozenset({"peer_id", "text"}),
    "moderation-mute": frozenset({"peer_id", "user_id", "until_seconds"}),
    "moderation-unmute": frozenset({"peer_id", "user_id"}), "moderation-ban": frozenset({"peer_id", "user_id", "reason"}),
    "moderation-unban": frozenset({"peer_id", "user_id"}), "moderation-kick": frozenset({"peer_id", "user_id"}),
    "moderation-promote": frozenset({"peer_id", "user_id", "rights"}), "moderation-demote": frozenset({"peer_id", "user_id"}),
    "moderation-pin": frozenset({"peer_id", "message_id"}), "moderation-delete-messages": frozenset({"peer_id", "message_id"}),
    "moderation-get-admins": frozenset({"peer_id"}),
    "profile-get": frozenset({"user_id"}), "profile-get-full": frozenset({"user_id"}), "profile-get-me": frozenset(),
    "profile-update-name": frozenset({"first_name", "last_name"}), "profile-update-bio": frozenset({"bio"}),
    "profile-update-username": frozenset({"username"}), "profile-set-photo": frozenset({"media_id"}),
    "profile-download-photo": frozenset({"user_id"}), "profile-get-photos": frozenset({"user_id", "limit"}),
    "profile-common-chats": frozenset({"user_id", "limit"}),
    "send-media": frozenset({"peer_id", "media_id", "caption"}), "download-media": frozenset({"peer_id", "message_id"}),
})

_TELEGRAM_REQUIRED_FIELDS = MappingProxyType({
    "get-message": frozenset({"message_id"}), "send-message": frozenset({"peer_id", "text"}),
    "edit-message": frozenset({"message_id", "text"}), "delete-message": frozenset({"message_id"}),
    "react": frozenset({"message_id", "reaction"}), "send-media": frozenset({"peer_id", "media_id"}),
    "download-media": frozenset({"peer_id", "message_id"}),
})

_OPERATIONS = MappingProxyType({
    CapabilityFamily.TELEGRAM: frozenset(_TELEGRAM_OPERATION_FIELDS),
    CapabilityFamily.WORKSPACE_FS: frozenset({"read", "write", "list"}),
    CapabilityFamily.PROCESS: frozenset({"run"}),
    CapabilityFamily.HTTPS_FETCH: frozenset({"fetch"}),
    CapabilityFamily.SCHEDULING: frozenset({"schedule"}),
    CapabilityFamily.CONFIGURATION: frozenset({"get", "set"}),
    CapabilityFamily.MCUB_CONTROL: frozenset({"module-list", "module-install", "module-reload", "config-get", "config-set"}),
})


def _operation_allowed(capability: CapabilityFamily, operation: str) -> bool:
    return operation in _OPERATIONS[capability]


def _only(payload: Mapping[str, Any], allowed: frozenset[str]) -> None:
    if set(payload) - allowed:
        raise CapabilityProtocolError("request has unknown fields")


def _relative_path(value: Any) -> str:
    path = _required(value, "path")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or path in {".", ""}:
        raise CapabilityProtocolError("path must be a grant-relative path")
    return path


def _relative_directory(value: Any) -> str:
    """Accept an explicit grant root only for directory and cwd operations."""

    path = _required(value, "path")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise CapabilityProtocolError("path must be a grant-relative path")
    return path


def _has_ambient_key(value: Any) -> bool:
    forbidden = {"client", "event", "session", "token", "password", "api_hash", "api_id", "credential", "secret"}
    if isinstance(value, Mapping):
        return any(key.strip().lower() in forbidden or _has_ambient_key(nested) for key, nested in value.items())
    if isinstance(value, tuple):
        return any(_has_ambient_key(item) for item in value)
    return False


def _positive_bound(value: Any, name: str, maximum: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CapabilityProtocolError(f"{name} must be positive")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0 or value > maximum:
        raise CapabilityProtocolError(f"{name} exceeds grant")
    return value


def _normalize_payload(request: CapabilityRequest, grant: CapabilityGrant, resolver: Callable[..., Any]) -> Mapping[str, Any]:
    payload = request.payload
    if request.capability is CapabilityFamily.TELEGRAM:
        _only(payload, frozenset({"data"}))
        if not isinstance(payload.get("data"), Mapping):
            raise CapabilityProtocolError("telegram data must be an opaque JSON object")
        if _has_ambient_key(payload["data"]):
            raise CapabilityProtocolError("telegram payload contains ambient credentials")
        allowed = _TELEGRAM_OPERATION_FIELDS[request.operation]
        _only(payload["data"], allowed)
        required = _TELEGRAM_REQUIRED_FIELDS.get(request.operation, frozenset())
        if not required.issubset(payload["data"]):
            raise CapabilityProtocolError("telegram operation is missing required opaque fields")
        for key, value in payload["data"].items():
            if key.endswith("_id") and (not isinstance(value, str) or not value):
                raise CapabilityProtocolError("telegram references must be opaque non-empty IDs")
        return MappingProxyType({"data": payload["data"]})
    elif request.capability is CapabilityFamily.WORKSPACE_FS:
        allowed = frozenset({"path", "content", "mode", "expected_hash"}) if request.operation == "write" else frozenset({"path"})
        _only(payload, allowed)
        path = _relative_directory(payload.get("path")) if request.operation == "list" else _relative_path(payload.get("path"))
        if request.operation == "write" and not isinstance(payload.get("content"), str):
            raise CapabilityProtocolError("writes require string content")
        if request.operation == "write":
            mode = payload.get("mode", "overwrite")
            if mode not in {"overwrite", "append"}:
                raise CapabilityProtocolError("write mode must be overwrite or append")
            expected_hash = payload.get("expected_hash")
            if expected_hash is not None and (not isinstance(expected_hash, str) or not expected_hash):
                raise CapabilityProtocolError("write expected_hash must be a non-empty string")
        root = grant.constraints.get("root")
        if not isinstance(root, str) or not root:
            raise CapabilityProtocolError("filesystem grants require a root")
        resolved = resolve_workspace_directory(root, path) if request.operation == "list" else resolve_workspace_path(root, path)
        normalized = {"path": str(resolved)}
        if request.operation == "write":
            normalized["content"] = payload["content"]
            normalized["mode"] = mode
            if expected_hash is not None:
                normalized["expected_hash"] = expected_hash
        return MappingProxyType(normalized)
    elif request.capability is CapabilityFamily.PROCESS:
        _only(payload, frozenset({"argv", "cwd", "timeout_seconds", "max_output_bytes", "env"}))
        argv = payload.get("argv")
        if not isinstance(argv, tuple) or not argv or any(not isinstance(arg, str) or not arg for arg in argv):
            raise CapabilityProtocolError("process requests require a non-empty argv array")
        required_constraints = {"executables", "cwd_root", "max_timeout_seconds", "max_output_bytes", "max_args", "max_arg_length", "env_allowlist"}
        if not required_constraints.issubset(grant.constraints):
            raise CapabilityProtocolError("process grant lacks mandatory constraints")
        if argv[0] not in grant.constraints["executables"]:
            raise CapabilityProtocolError("process executable is not granted")
        _positive_bound(len(argv), "argv count", grant.constraints["max_args"])
        if any(len(arg) > grant.constraints["max_arg_length"] for arg in argv):
            raise CapabilityProtocolError("argv entry exceeds grant")
        cwd = resolve_workspace_directory(grant.constraints["cwd_root"], payload.get("cwd", "."))
        timeout = payload.get("timeout_seconds")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0 or timeout > grant.constraints["max_timeout_seconds"]:
            raise CapabilityProtocolError("process timeout exceeds grant")
        max_output = _positive_bound(payload.get("max_output_bytes"), "max output", grant.constraints["max_output_bytes"])
        if "env" in payload:
            if not isinstance(payload["env"], Mapping) or not set(payload["env"]).issubset(set(grant.constraints.get("env_allowlist", ()))) or any(not isinstance(value, str) for value in payload["env"].values()):
                raise CapabilityProtocolError("process environment exceeds grant")
        return MappingProxyType({"argv": argv, "cwd": str(cwd), "timeout_seconds": timeout, "max_output_bytes": max_output, "env": payload.get("env", MappingProxyType({}))})
    elif request.capability is CapabilityFamily.HTTPS_FETCH:
        _only(payload, frozenset({"url", "timeout_seconds", "max_bytes"}))
        if not {"max_timeout_seconds", "max_bytes"}.issubset(grant.constraints):
            raise CapabilityProtocolError("HTTPS grant lacks mandatory bounds")
        url = validate_public_https_url(payload.get("url"), resolver)
        timeout = payload.get("timeout_seconds")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0 or timeout > grant.constraints["max_timeout_seconds"]:
            raise CapabilityProtocolError("HTTPS timeout exceeds grant")
        max_bytes = _positive_bound(payload.get("max_bytes"), "HTTPS bytes", grant.constraints["max_bytes"])
        return MappingProxyType({"url": url, "timeout_seconds": timeout, "max_bytes": max_bytes, "redirects": "broker-validated-only"})
    elif request.capability is CapabilityFamily.SCHEDULING:
        _only(payload, frozenset({"child_call"}))
        child = payload.get("child_call")
        if not isinstance(child, Mapping) or set(child) != {"call_id", "canonical_tool_id", "arguments", "remaining_calls", "remaining_token_budget", "remaining_depth", "parent_call_id", "cancellation_parent_id"}:
            raise CapabilityProtocolError("child call must be canonical bounded data")
        if child["parent_call_id"] != request.call_id or child["cancellation_parent_id"] != grant.constraints.get("cancellation_parent_id"):
            raise CapabilityProtocolError("child call is not bound to parent budget")
        if child["call_id"] == request.call_id:
            raise CapabilityProtocolError("child call cannot reuse its parent identity")
        for budget in ("remaining_calls", "remaining_token_budget", "remaining_depth"):
            if not isinstance(child[budget], int) or child[budget] < 0 or child[budget] >= grant.constraints.get(budget, -1):
                raise CapabilityProtocolError("child budget is not strictly reduced")
        canonical_tool_id = normalize_tool_name(child["canonical_tool_id"], canonical=True)
        ancestors = grant.constraints.get("ancestor_call_ids", ())
        if not isinstance(ancestors, (list, tuple)) or any(not isinstance(item, str) for item in ancestors):
            raise CapabilityProtocolError("scheduling ancestry must be a string sequence")
        if child["call_id"] in ancestors:
            raise CapabilityProtocolError("child call creates an identity cycle")
        ancestor_tools = grant.constraints.get("ancestor_tool_ids", ())
        if not isinstance(ancestor_tools, (list, tuple)) or any(not isinstance(item, str) for item in ancestor_tools):
            raise CapabilityProtocolError("scheduling tool ancestry must be a string sequence")
        if canonical_tool_id in ancestor_tools:
            raise CapabilityProtocolError("child call creates a tool cycle")
        _json(child["arguments"])
        return MappingProxyType(dict(child))
    elif request.capability is CapabilityFamily.MCUB_CONTROL:
        if request.operation in {"module-list", "module-reload"}:
            _only(payload, frozenset())
            return MappingProxyType({})
        if request.operation == "module-install":
            _only(payload, frozenset({"module_url"}))
            module_url = _required(payload.get("module_url"), "module_url")
            parsed = urlparse(module_url)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or len(module_url) > 2_048:
                raise CapabilityProtocolError("module installation requires a bounded credential-free HTTPS URL")
            return MappingProxyType({"module_url": module_url})
        _only(payload, frozenset({"key", "value"}) if request.operation == "config-set" else frozenset({"key"}))
        key = _required(payload.get("key"), "key")
        allowed_keys = grant.constraints.get("keys")
        if not isinstance(allowed_keys, (list, tuple, frozenset)) or key not in allowed_keys:
            raise CapabilityProtocolError("MCUB configuration key is not granted")
        if request.operation == "config-set" and "value" not in payload:
            raise CapabilityProtocolError("MCUB configuration writes need a value")
        return MappingProxyType(dict(payload))
    else:
        _only(payload, frozenset({"key", "value"}) if request.operation == "set" else frozenset({"key"}))
        key = _required(payload.get("key"), "key")
        namespace = grant.constraints.get("namespace")
        if not isinstance(namespace, str) or not (key == namespace or key.startswith(namespace + ".")):
            raise CapabilityProtocolError("configuration key lies outside grant namespace")
        if request.operation == "set" and "value" not in payload:
            raise CapabilityProtocolError("configuration writes need a value")
        return MappingProxyType(dict(payload))


__all__ = [name for name in globals() if name.startswith(("Capability", "_OPERATIONS"))]
