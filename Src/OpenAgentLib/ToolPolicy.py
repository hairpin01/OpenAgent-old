# SPDX-License-Identifier: MIT
"""Authoritative authorization and scheduling policy for v2 tool calls.

This module only decides whether a validated call may run.  It deliberately does
not invoke handlers, hooks, confirmations, or any other runtime integration.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import AsyncIterator, Iterable, Mapping

from .ToolCompatibility import ToolCompatibility, compatibility_matrix
from .ToolKernel import (
    ConcurrencyClass,
    ConfirmationRequirement,
    IdempotencyClass,
    MigrationDisposition,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    normalize_tool_name,
)


class ToolPolicyCatalogError(ValueError):
    """The reviewed compatibility contract cannot form a complete catalog."""


class PolicyDecisionKind(str, Enum):
    ALLOW = "allow"
    CONFIRMATION_REQUIRED = "confirmation-required"
    DENY = "deny"


class PolicyReasonCode(str, Enum):
    ALLOWED = "allowed"
    UNKNOWN_TOOL = "unknown-tool"
    TOOL_DISABLED = "tool-disabled"
    MIGRATION_REJECTED = "migration-rejected"
    SPEC_METADATA_DRIFT = "spec-metadata-drift"
    SCHEMA_INVALID = "schema-invalid"
    CAPABILITY_NOT_GRANTED = "capability-not-granted"
    CONFIRMATION_REQUIRED = "confirmation-required"
    CONFIRMATION_REJECTED = "confirmation-rejected"
    INVALID_CONFIRMATION_GRANT = "invalid-confirmation-grant"
    INVALID_TIMEOUT = "invalid-timeout"
    TIMEOUT_EXCEEDED = "timeout-exceeded"
    CALL_BUDGET_EXHAUSTED = "call-budget-exhausted"
    TOKEN_BUDGET_EXHAUSTED = "token-budget-exhausted"
    INVALID_BUDGET = "invalid-budget"
    RETRY_NOT_ELIGIBLE = "retry-not-eligible"


class ConfirmationState(str, Enum):
    MISSING = "missing"
    APPROVED = "approved"
    REJECTED = "rejected"


class ToolExecutionLane(str, Enum):
    SERIAL = "serial"
    PARALLEL_READ = "parallel-read"


def tool_scope_for(call: ToolCall) -> str:
    """Return the stable authorization and serial-lane scope for a call."""

    context = call.context
    if context is not None and context.actor_id:
        return f"actor:{context.actor_id}"
    if context is not None:
        return f"session:{context.correlation_id}"
    return f"call:{call.call_id}"


@dataclass(frozen=True)
class ToolConfirmationGrant:
    """One immutable approval, bound to exactly one tool call and scope."""

    token: str
    call_id: str
    canonical_id: str
    scope: str
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        token = self.token.strip() if isinstance(self.token, str) else ""
        if not token:
            raise ValueError("confirmation grants require a non-empty token")
        if not isinstance(self.call_id, str) or not self.call_id.strip():
            raise ValueError("confirmation grants require a call ID")
        if not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError("confirmation grants require a scope")
        if self.expires_at is not None and (
            not isinstance(self.expires_at, datetime) or self.expires_at.tzinfo is None
        ):
            raise ValueError("confirmation grant expiry must be timezone-aware")
        object.__setattr__(self, "token", token)
        object.__setattr__(self, "call_id", self.call_id.strip())
        object.__setattr__(
            self, "canonical_id", normalize_tool_name(self.canonical_id, canonical=True)
        )
        object.__setattr__(self, "scope", self.scope.strip())

    @classmethod
    def for_call(
        cls, token: str, call: ToolCall, *, expires_at: datetime | None = None
    ) -> "ToolConfirmationGrant":
        return cls(
            token,
            call.call_id,
            call.spec.canonical_id,
            tool_scope_for(call),
            expires_at,
        )


@dataclass(frozen=True)
class ToolPolicyRule:
    """Reviewed policy metadata, keyed exclusively by a canonical tool ID."""

    canonical_id: str
    capabilities: frozenset[str]
    confirmation: ConfirmationRequirement
    concurrency: ConcurrencyClass
    idempotency: IdempotencyClass
    migration_disposition: MigrationDisposition

    def __post_init__(self) -> None:
        canonical_id = normalize_tool_name(self.canonical_id, canonical=True)
        capabilities = frozenset(
            str(capability).strip().lower() for capability in self.capabilities
        )
        if not capabilities or any(not capability for capability in capabilities):
            raise ToolPolicyCatalogError("policy rules require reviewed capabilities")
        object.__setattr__(self, "canonical_id", canonical_id)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(
            self, "confirmation", ConfirmationRequirement(self.confirmation)
        )
        object.__setattr__(self, "concurrency", ConcurrencyClass(self.concurrency))
        object.__setattr__(self, "idempotency", IdempotencyClass(self.idempotency))
        object.__setattr__(
            self,
            "migration_disposition",
            MigrationDisposition(self.migration_disposition),
        )

    @classmethod
    def from_compatibility(cls, entry: ToolCompatibility) -> "ToolPolicyRule":
        return cls(
            canonical_id=entry.canonical_id,
            capabilities=frozenset({entry.capability_class}),
            confirmation=ConfirmationRequirement(entry.confirmation_class),
            concurrency=ConcurrencyClass(entry.concurrency_class),
            idempotency=IdempotencyClass(entry.idempotency_class),
            migration_disposition=MigrationDisposition(entry.migration_disposition),
        )


class ToolPolicyCatalog:
    """Immutable canonical-ID policy lookup, optionally checked against the matrix."""

    def __init__(
        self,
        rules: Iterable[ToolPolicyRule],
        *,
        matrix: Iterable[ToolCompatibility] | None = None,
    ) -> None:
        indexed: dict[str, ToolPolicyRule] = {}
        for rule in rules:
            if not isinstance(rule, ToolPolicyRule):
                raise ToolPolicyCatalogError(
                    "policy catalogs contain only ToolPolicyRule values"
                )
            if rule.canonical_id in indexed:
                raise ToolPolicyCatalogError(
                    f"duplicate canonical policy ID {rule.canonical_id!r}"
                )
            indexed[rule.canonical_id] = rule
        if matrix is not None:
            expected: dict[str, ToolPolicyRule] = {}
            for entry in matrix:
                rule = ToolPolicyRule.from_compatibility(entry)
                if rule.canonical_id in expected:
                    raise ToolPolicyCatalogError(
                        f"duplicate matrix canonical ID {rule.canonical_id!r}"
                    )
                expected[rule.canonical_id] = rule
            missing = sorted(set(expected) - set(indexed))
            unknown = sorted(set(indexed) - set(expected))
            drifted = sorted(
                canonical_id
                for canonical_id in set(indexed) & set(expected)
                if indexed[canonical_id] != expected[canonical_id]
            )
            if missing or unknown or drifted:
                raise ToolPolicyCatalogError(
                    "catalog must exactly match the committed matrix: "
                    f"missing={missing}, unknown={unknown}, drifted={drifted}"
                )
        self._rules: Mapping[str, ToolPolicyRule] = MappingProxyType(
            dict(sorted(indexed.items()))
        )

    @classmethod
    def from_compatibility_matrix(
        cls, matrix: Iterable[ToolCompatibility] | None = None
    ) -> "ToolPolicyCatalog":
        entries = tuple(compatibility_matrix() if matrix is None else matrix)
        return cls(
            (ToolPolicyRule.from_compatibility(entry) for entry in entries),
            matrix=entries,
        )

    @property
    def rules(self) -> Mapping[str, ToolPolicyRule]:
        return self._rules

    def get(self, canonical_id: str) -> ToolPolicyRule | None:
        try:
            return self._rules[normalize_tool_name(canonical_id, canonical=True)]
        except (KeyError, ValueError):
            return None


@dataclass(frozen=True)
class ToolPolicyRequest:
    """Execution environment supplied to the pure policy evaluator."""

    enabled_tool_ids: frozenset[str]
    granted_capabilities: frozenset[str]
    confirmation: ConfirmationState = ConfirmationState.MISSING
    confirmation_grant: ToolConfirmationGrant | None = None
    requested_timeout: float = 30.0
    maximum_timeout: float = 30.0
    remaining_calls: int = 1
    remaining_token_budget: int = 0
    estimated_tokens: int = 0
    schema_valid: bool = True
    retry_attempt: int = 1
    maximum_attempts: int = 1
    now: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "enabled_tool_ids",
            frozenset(str(value).strip().lower() for value in self.enabled_tool_ids),
        )
        object.__setattr__(
            self,
            "granted_capabilities",
            frozenset(
                str(value).strip().lower() for value in self.granted_capabilities
            ),
        )
        object.__setattr__(self, "confirmation", ConfirmationState(self.confirmation))
        if self.confirmation_grant is not None and not isinstance(
            self.confirmation_grant, ToolConfirmationGrant
        ):
            raise TypeError("confirmation_grant must be a ToolConfirmationGrant")
        if self.now is not None and (
            not isinstance(self.now, datetime) or self.now.tzinfo is None
        ):
            raise TypeError("policy request time must be timezone-aware")


@dataclass(frozen=True)
class ToolPolicyDecision:
    kind: PolicyDecisionKind
    canonical_id: str
    reason: PolicyReasonCode
    lane: ToolExecutionLane | None = None

    @property
    def allowed(self) -> bool:
        return self.kind is PolicyDecisionKind.ALLOW


class ToolPolicyEngine:
    """Fail-closed evaluator over immutable calls and reviewed policy data."""

    def __init__(self, catalog: ToolPolicyCatalog) -> None:
        self.catalog = catalog

    def evaluate(
        self, call: ToolCall, request: ToolPolicyRequest
    ) -> ToolPolicyDecision:
        canonical_id = call.spec.canonical_id
        rule = self.catalog.get(canonical_id)
        if rule is None:
            return self._deny(canonical_id, PolicyReasonCode.UNKNOWN_TOOL)
        if canonical_id not in request.enabled_tool_ids:
            return self._deny(canonical_id, PolicyReasonCode.TOOL_DISABLED)
        if rule.migration_disposition is MigrationDisposition.REJECT:
            return self._deny(canonical_id, PolicyReasonCode.MIGRATION_REJECTED)
        if not self._matches_rule(call, rule):
            return self._deny(canonical_id, PolicyReasonCode.SPEC_METADATA_DRIFT)
        if not request.schema_valid:
            return self._deny(canonical_id, PolicyReasonCode.SCHEMA_INVALID)
        if not rule.capabilities.issubset(request.granted_capabilities):
            return self._deny(canonical_id, PolicyReasonCode.CAPABILITY_NOT_GRANTED)
        if rule.confirmation is ConfirmationRequirement.REQUIRED:
            confirmation_reason = self._confirmation_reason(call, request)
            if confirmation_reason is not None:
                if confirmation_reason is PolicyReasonCode.CONFIRMATION_REQUIRED:
                    return ToolPolicyDecision(
                        PolicyDecisionKind.CONFIRMATION_REQUIRED,
                        canonical_id,
                        confirmation_reason,
                    )
                return self._deny(canonical_id, confirmation_reason)
        budget_reason = self._budget_reason(request)
        if budget_reason is not None:
            return self._deny(canonical_id, budget_reason)
        return ToolPolicyDecision(
            PolicyDecisionKind.ALLOW,
            canonical_id,
            PolicyReasonCode.ALLOWED,
            self.lane_for(call),
        )

    def retry_eligible(
        self, call: ToolCall, result: ToolResult, request: ToolPolicyRequest
    ) -> bool:
        rule = self.catalog.get(call.spec.canonical_id)
        return bool(
            rule is not None
            and result.call_id == call.call_id
            and self.evaluate(call, request).kind is PolicyDecisionKind.ALLOW
            and self._matches_rule(call, rule)
            and rule.migration_disposition is MigrationDisposition.MIGRATE
            and rule.idempotency is IdempotencyClass.IDEMPOTENT
            and result.status is ToolResultStatus.ERROR
            and result.retryable
            and request.retry_attempt >= 1
            and request.maximum_attempts >= 1
            and request.retry_attempt < request.maximum_attempts
        )

    def lane_for(self, call: ToolCall) -> ToolExecutionLane:
        rule = self.catalog.get(call.spec.canonical_id)
        if (
            rule is not None
            and self._matches_rule(call, rule)
            and rule.confirmation is ConfirmationRequirement.NONE
            and rule.concurrency is ConcurrencyClass.PARALLEL_READ
            and rule.idempotency is IdempotencyClass.IDEMPOTENT
        ):
            return ToolExecutionLane.PARALLEL_READ
        return ToolExecutionLane.SERIAL

    @staticmethod
    def _matches_rule(call: ToolCall, rule: ToolPolicyRule) -> bool:
        spec = call.spec
        return (
            spec.canonical_id == rule.canonical_id
            and spec.capabilities == rule.capabilities
            and spec.confirmation is rule.confirmation
            and spec.concurrency is rule.concurrency
            and spec.idempotency is rule.idempotency
            and spec.migration_disposition is rule.migration_disposition
        )

    @staticmethod
    def _confirmation_reason(
        call: ToolCall, request: ToolPolicyRequest
    ) -> PolicyReasonCode | None:
        if request.confirmation is ConfirmationState.MISSING:
            return PolicyReasonCode.CONFIRMATION_REQUIRED
        if request.confirmation is ConfirmationState.REJECTED:
            return PolicyReasonCode.CONFIRMATION_REJECTED
        grant = request.confirmation_grant
        if grant is None:
            return PolicyReasonCode.INVALID_CONFIRMATION_GRANT
        now = request.now or datetime.now(timezone.utc)
        if (
            grant.call_id != call.call_id
            or grant.canonical_id != call.spec.canonical_id
            or grant.scope != tool_scope_for(call)
            or (grant.expires_at is not None and grant.expires_at < now)
        ):
            return PolicyReasonCode.INVALID_CONFIRMATION_GRANT
        return None

    @staticmethod
    def _budget_reason(request: ToolPolicyRequest) -> PolicyReasonCode | None:
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
            for value in (request.requested_timeout, request.maximum_timeout)
        ):
            return PolicyReasonCode.INVALID_TIMEOUT
        if request.requested_timeout <= 0 or request.maximum_timeout <= 0:
            return PolicyReasonCode.INVALID_TIMEOUT
        if request.requested_timeout > request.maximum_timeout:
            return PolicyReasonCode.TIMEOUT_EXCEEDED
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (
                request.remaining_calls,
                request.remaining_token_budget,
                request.estimated_tokens,
            )
        ):
            return PolicyReasonCode.INVALID_BUDGET
        if request.remaining_calls <= 0:
            return PolicyReasonCode.CALL_BUDGET_EXHAUSTED
        if request.remaining_token_budget < 0 or request.estimated_tokens < 0:
            return PolicyReasonCode.INVALID_BUDGET
        if request.estimated_tokens > request.remaining_token_budget:
            return PolicyReasonCode.TOKEN_BUDGET_EXHAUSTED
        return None

    @staticmethod
    def _deny(canonical_id: str, reason: PolicyReasonCode) -> ToolPolicyDecision:
        return ToolPolicyDecision(PolicyDecisionKind.DENY, canonical_id, reason)


@dataclass
class _ScopeEntry:
    lock: asyncio.Lock
    holders: int = 0
    waiters: int = 0


class ToolConcurrencyGate:
    """Per-actor/session serial lanes with centrally approved parallel reads."""

    def __init__(self, policy: ToolPolicyEngine) -> None:
        self.policy = policy
        self._scope_entries: dict[str, _ScopeEntry] = {}

    @property
    def active_scope_count(self) -> int:
        """Number of scopes currently held or waiting, without exposing locks."""

        return len(self._scope_entries)

    @asynccontextmanager
    async def acquire(self, call: ToolCall) -> AsyncIterator[ToolExecutionLane]:
        lane = self.policy.lane_for(call)
        if lane is ToolExecutionLane.PARALLEL_READ:
            yield lane
            return

        scope = tool_scope_for(call)
        entry = self._scope_entries.get(scope)
        if entry is None:
            entry = _ScopeEntry(asyncio.Lock())
            self._scope_entries[scope] = entry
        entry.waiters += 1
        try:
            await entry.lock.acquire()
        except BaseException:
            entry.waiters -= 1
            self._discard_idle_entry(scope, entry)
            raise
        entry.waiters -= 1
        entry.holders += 1
        try:
            yield lane
        finally:
            entry.holders -= 1
            entry.lock.release()
            self._discard_idle_entry(scope, entry)

    def _discard_idle_entry(self, scope: str, entry: _ScopeEntry) -> None:
        if (
            self._scope_entries.get(scope) is entry
            and entry.holders == 0
            and entry.waiters == 0
            and not entry.lock.locked()
        ):
            del self._scope_entries[scope]

    @staticmethod
    def scope_for(call: ToolCall) -> str:
        return tool_scope_for(call)


DEFAULT_TOOL_POLICY_CATALOG = ToolPolicyCatalog.from_compatibility_matrix()


__all__ = [
    "ConfirmationState",
    "DEFAULT_TOOL_POLICY_CATALOG",
    "PolicyDecisionKind",
    "PolicyReasonCode",
    "ToolConcurrencyGate",
    "ToolConfirmationGrant",
    "ToolExecutionLane",
    "ToolPolicyCatalog",
    "ToolPolicyCatalogError",
    "ToolPolicyDecision",
    "ToolPolicyEngine",
    "ToolPolicyRequest",
    "ToolPolicyRule",
    "tool_scope_for",
]
