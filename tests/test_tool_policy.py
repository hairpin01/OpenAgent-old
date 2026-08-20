from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "Src"))

from OpenAgentLib.ToolKernel import (  # noqa: E402
    ConfirmationRequirement,
    ConcurrencyClass,
    IdempotencyClass,
    MigrationDisposition,
    ToolCall,
    ToolContext,
    ToolError,
    ToolErrorCode,
    ToolResult,
    ToolResultStatus,
    ToolSpec,
)
from OpenAgentLib.ToolPolicy import (  # noqa: E402
    ConfirmationState,
    PolicyDecisionKind,
    PolicyReasonCode,
    ToolConcurrencyGate,
    ToolConfirmationGrant,
    ToolExecutionLane,
    ToolPolicyCatalog,
    ToolPolicyEngine,
    ToolPolicyRequest,
    ToolPolicyRule,
)


def _rule(
    canonical_id: str = "sample.inspect",
    *,
    capabilities: frozenset[str] = frozenset({"read-only"}),
    confirmation: ConfirmationRequirement = ConfirmationRequirement.NONE,
    concurrency: ConcurrencyClass = ConcurrencyClass.PARALLEL_READ,
    idempotency: IdempotencyClass = IdempotencyClass.IDEMPOTENT,
    disposition: MigrationDisposition = MigrationDisposition.MIGRATE,
) -> ToolPolicyRule:
    return ToolPolicyRule(
        canonical_id=canonical_id,
        capabilities=capabilities,
        confirmation=confirmation,
        concurrency=concurrency,
        idempotency=idempotency,
        migration_disposition=disposition,
    )


def _call(
    rule: ToolPolicyRule,
    *,
    requested_name: str | None = None,
    actor_id: str | None = "actor-a",
    correlation_id: str = "session-a",
    capabilities: frozenset[str] | None = None,
    confirmation: ConfirmationRequirement | None = None,
    concurrency: ConcurrencyClass | None = None,
    idempotency: IdempotencyClass | None = None,
    disposition: MigrationDisposition | None = None,
) -> ToolCall:
    aliases = ("inspect",) if requested_name == "inspect" else ()
    spec = ToolSpec(
        canonical_id=rule.canonical_id,
        aliases=aliases,
        input_schema={"type": "object", "additionalProperties": False},
        output_schema={"type": "object", "additionalProperties": True},
        api_version="2",
        schema_version="2",
        capabilities=capabilities if capabilities is not None else rule.capabilities,
        confirmation=confirmation if confirmation is not None else rule.confirmation,
        concurrency=concurrency if concurrency is not None else rule.concurrency,
        idempotency=idempotency if idempotency is not None else rule.idempotency,
        migration_disposition=(
            disposition if disposition is not None else rule.migration_disposition
        ),
    )
    return ToolCall(
        call_id=f"call-{correlation_id}-{actor_id or 'none'}",
        spec=spec,
        requested_name=requested_name or rule.canonical_id,
        arguments={},
        context=ToolContext(correlation_id=correlation_id, actor_id=actor_id),
    )


def _request(rule: ToolPolicyRule, **overrides: object) -> ToolPolicyRequest:
    values: dict[str, object] = {
        "enabled_tool_ids": frozenset({rule.canonical_id}),
        "granted_capabilities": rule.capabilities,
        "requested_timeout": 5.0,
        "maximum_timeout": 10.0,
        "remaining_calls": 1,
        "remaining_token_budget": 100,
        "estimated_tokens": 10,
    }
    values.update(overrides)
    return ToolPolicyRequest(**values)  # type: ignore[arg-type]


def _engine(rule: ToolPolicyRule) -> ToolPolicyEngine:
    return ToolPolicyEngine(ToolPolicyCatalog([rule]))


def test_alias_bypass_uses_canonical_policy_and_denies_missing_capability() -> None:
    rule = _rule(capabilities=frozenset({"filesystem-write"}))
    decision = _engine(rule).evaluate(
        _call(rule, requested_name="inspect"),
        _request(rule, granted_capabilities=frozenset()),
    )

    assert decision.kind is PolicyDecisionKind.DENY
    assert decision.canonical_id == rule.canonical_id
    assert decision.reason is PolicyReasonCode.CAPABILITY_NOT_GRANTED


def test_forged_safe_spec_metadata_is_denied_and_cannot_use_parallel_lane() -> None:
    rule = _rule(
        capabilities=frozenset({"filesystem-write"}),
        confirmation=ConfirmationRequirement.REQUIRED,
        concurrency=ConcurrencyClass.SERIAL,
        idempotency=IdempotencyClass.NON_IDEMPOTENT,
    )
    call = _call(
        rule,
        capabilities=frozenset({"read-only"}),
        confirmation=ConfirmationRequirement.NONE,
        concurrency=ConcurrencyClass.PARALLEL_READ,
        idempotency=IdempotencyClass.IDEMPOTENT,
    )
    engine = _engine(rule)
    decision = engine.evaluate(
        call, _request(rule, granted_capabilities=frozenset({"read-only"}))
    )

    assert decision.reason is PolicyReasonCode.SPEC_METADATA_DRIFT
    assert engine.lane_for(call) is ToolExecutionLane.SERIAL


def test_confirmation_is_required_before_allowing_a_privileged_call() -> None:
    rule = _rule(confirmation=ConfirmationRequirement.REQUIRED)
    engine = _engine(rule)
    call = _call(rule)

    pending = engine.evaluate(call, _request(rule))
    approved = engine.evaluate(
        call,
        _request(
            rule,
            confirmation=ConfirmationState.APPROVED,
            confirmation_grant=ToolConfirmationGrant.for_call("approved-once", call),
        ),
    )

    assert pending.kind is PolicyDecisionKind.CONFIRMATION_REQUIRED
    assert pending.reason is PolicyReasonCode.CONFIRMATION_REQUIRED
    assert approved.kind is PolicyDecisionKind.ALLOW


@pytest.mark.parametrize("replay", ("call", "tool", "scope"))
def test_confirmation_grant_cannot_be_replayed_across_call_tool_or_scope(
    replay: str,
) -> None:
    rule = _rule(confirmation=ConfirmationRequirement.REQUIRED)
    source_call = _call(rule, correlation_id="source", actor_id="source")
    grant = ToolConfirmationGrant.for_call("approved-once", source_call)
    target_rule = (
        _rule("other.inspect", confirmation=ConfirmationRequirement.REQUIRED)
        if replay == "tool"
        else rule
    )
    target_call = (
        _call(target_rule, correlation_id="source", actor_id="source")
        if replay == "tool"
        else _call(
            rule,
            correlation_id="target" if replay == "call" else "source",
            actor_id="target" if replay == "scope" else "source",
        )
    )
    catalog = (
        ToolPolicyCatalog([rule, target_rule])
        if replay == "tool"
        else ToolPolicyCatalog([rule])
    )
    decision = ToolPolicyEngine(catalog).evaluate(
        target_call,
        _request(
            target_rule,
            confirmation=ConfirmationState.APPROVED,
            confirmation_grant=grant,
        ),
    )

    assert decision.kind is PolicyDecisionKind.DENY
    assert decision.reason is PolicyReasonCode.INVALID_CONFIRMATION_GRANT


def test_explicit_confirmation_rejection_is_a_typed_deny() -> None:
    rule = _rule(confirmation=ConfirmationRequirement.REQUIRED)
    decision = _engine(rule).evaluate(
        _call(rule), _request(rule, confirmation=ConfirmationState.REJECTED)
    )

    assert decision.kind is PolicyDecisionKind.DENY
    assert decision.reason is PolicyReasonCode.CONFIRMATION_REJECTED


def test_expired_confirmation_grant_is_denied() -> None:
    rule = _rule(confirmation=ConfirmationRequirement.REQUIRED)
    call = _call(rule)
    grant = ToolConfirmationGrant.for_call(
        "expired", call, expires_at=datetime(2020, 1, 1, tzinfo=timezone.utc)
    )
    decision = _engine(rule).evaluate(
        call,
        _request(
            rule,
            confirmation=ConfirmationState.APPROVED,
            confirmation_grant=grant,
            now=datetime(2021, 1, 1, tzinfo=timezone.utc),
        ),
    )

    assert decision.kind is PolicyDecisionKind.DENY
    assert decision.reason is PolicyReasonCode.INVALID_CONFIRMATION_GRANT


def test_rejected_migration_and_unknown_tool_are_denied() -> None:
    rejected = _rule(disposition=MigrationDisposition.REJECT)
    assert (
        _engine(rejected).evaluate(_call(rejected), _request(rejected)).reason
        is PolicyReasonCode.MIGRATION_REJECTED
    )

    known = _rule()
    unknown = _rule("other.inspect")
    assert (
        _engine(known).evaluate(_call(unknown), _request(unknown)).reason
        is PolicyReasonCode.UNKNOWN_TOOL
    )


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"requested_timeout": 0.0}, PolicyReasonCode.INVALID_TIMEOUT),
        ({"requested_timeout": 11.0}, PolicyReasonCode.TIMEOUT_EXCEEDED),
        ({"remaining_calls": 0}, PolicyReasonCode.CALL_BUDGET_EXHAUSTED),
        ({"estimated_tokens": 101}, PolicyReasonCode.TOKEN_BUDGET_EXHAUSTED),
    ],
)
def test_timeout_and_budget_requests_are_denied_without_clamping(
    overrides: dict[str, object], reason: PolicyReasonCode
) -> None:
    rule = _rule()
    decision = _engine(rule).evaluate(_call(rule), _request(rule, **overrides))

    assert decision.kind is PolicyDecisionKind.DENY
    assert decision.reason is reason


def test_retry_requires_authoritative_idempotency_retryable_error_and_attempt_budget() -> (
    None
):
    rule = _rule()
    engine = _engine(rule)
    call = _call(rule)
    error = ToolError(ToolErrorCode.INVALID_ARGUMENT, "transient failure")
    retryable = ToolResult(
        call.call_id, ToolResultStatus.ERROR, error=error, retryable=True
    )
    cancelled = ToolResult(
        call.call_id, ToolResultStatus.CANCELLED, error=error, retryable=True
    )

    assert engine.retry_eligible(
        call, retryable, _request(rule, retry_attempt=1, maximum_attempts=2)
    )
    assert not engine.retry_eligible(
        call, cancelled, _request(rule, retry_attempt=1, maximum_attempts=2)
    )
    assert not engine.retry_eligible(
        call, retryable, _request(rule, retry_attempt=2, maximum_attempts=2)
    )

    mutation = _rule(idempotency=IdempotencyClass.NON_IDEMPOTENT)
    assert not _engine(mutation).retry_eligible(
        _call(mutation),
        retryable,
        _request(mutation, retry_attempt=1, maximum_attempts=2),
    )


def test_retry_rechecks_result_identity_and_current_authorization() -> None:
    rule = _rule()
    call = _call(rule)
    error = ToolError(ToolErrorCode.INVALID_ARGUMENT, "transient failure")
    mismatched = ToolResult(
        "other-call", ToolResultStatus.ERROR, error=error, retryable=True
    )
    result = ToolResult(
        call.call_id, ToolResultStatus.ERROR, error=error, retryable=True
    )
    engine = _engine(rule)

    assert not engine.retry_eligible(
        call, mismatched, _request(rule, retry_attempt=1, maximum_attempts=2)
    )
    assert not engine.retry_eligible(
        call,
        result,
        _request(
            rule, granted_capabilities=frozenset(), retry_attempt=1, maximum_attempts=2
        ),
    )
    assert not engine.retry_eligible(
        call,
        result,
        _request(
            rule, enabled_tool_ids=frozenset(), retry_attempt=1, maximum_attempts=2
        ),
    )
    assert not engine.retry_eligible(
        call,
        result,
        _request(rule, remaining_calls=0, retry_attempt=1, maximum_attempts=2),
    )


def test_catalog_from_matrix_is_complete_and_canonical_only() -> None:
    from OpenAgentLib.ToolCompatibility import compatibility_matrix

    catalog = ToolPolicyCatalog.from_compatibility_matrix()
    assert set(catalog.rules) == {
        entry.canonical_id for entry in compatibility_matrix()
    }
    assert catalog.get("chat") is None


def test_concurrency_gate_serializes_one_scope_but_not_other_scopes() -> None:
    async def scenario() -> tuple[int, int]:
        rule = _rule(concurrency=ConcurrencyClass.SERIAL)
        gate = ToolConcurrencyGate(_engine(rule))
        active = 0
        maximum = 0

        async def run(call: ToolCall) -> None:
            nonlocal active, maximum
            async with gate.acquire(call):
                active += 1
                maximum = max(maximum, active)
                await asyncio.sleep(0.02)
                active -= 1

        same_scope = [
            _call(rule, correlation_id="same", actor_id="one") for _ in range(3)
        ]
        await asyncio.gather(*(run(call) for call in same_scope))
        same_maximum = maximum
        active = maximum = 0
        await asyncio.gather(
            run(_call(rule, correlation_id="one", actor_id="one")),
            run(_call(rule, correlation_id="two", actor_id="two")),
        )
        return same_maximum, maximum

    assert asyncio.run(scenario()) == (1, 2)


def test_concurrency_gate_allows_approved_parallel_reads_to_overlap() -> None:
    async def scenario() -> int:
        rule = _rule()
        gate = ToolConcurrencyGate(_engine(rule))
        active = 0
        maximum = 0

        async def run(index: int) -> None:
            nonlocal active, maximum
            async with gate.acquire(
                _call(rule, correlation_id=f"scope-{index}", actor_id=f"actor-{index}")
            ):
                active += 1
                maximum = max(maximum, active)
                await asyncio.sleep(0.02)
                active -= 1

        await asyncio.gather(*(run(index) for index in range(3)))
        return maximum

    assert asyncio.run(scenario()) == 3


def test_concurrency_gate_cancellation_does_not_leak_a_waiting_lock() -> None:
    async def scenario() -> None:
        rule = _rule(concurrency=ConcurrencyClass.SERIAL)
        gate = ToolConcurrencyGate(_engine(rule))
        call = _call(rule)
        entered = asyncio.Event()
        release = asyncio.Event()

        async def holder() -> None:
            async with gate.acquire(call):
                entered.set()
                await release.wait()

        task = asyncio.create_task(holder())
        await entered.wait()
        waiter = asyncio.create_task(gate.acquire(call).__aenter__())
        await asyncio.sleep(0)
        assert gate.active_scope_count == 1
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        release.set()
        await task
        assert gate.active_scope_count == 0
        async with gate.acquire(call):
            pass
        assert gate.active_scope_count == 0

    asyncio.run(scenario())


def test_concurrency_gate_cleans_up_transient_scopes_and_cancelled_holders() -> None:
    async def scenario() -> None:
        rule = _rule(concurrency=ConcurrencyClass.SERIAL)
        gate = ToolConcurrencyGate(_engine(rule))

        for index in range(20):
            async with gate.acquire(
                _call(rule, correlation_id=f"scope-{index}", actor_id=f"actor-{index}")
            ):
                assert gate.active_scope_count == 1
            assert gate.active_scope_count == 0

        entered = asyncio.Event()
        call = _call(rule, correlation_id="cancelled", actor_id="cancelled")

        async def holder() -> None:
            async with gate.acquire(call):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(holder())
        await entered.wait()
        assert gate.active_scope_count == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert gate.active_scope_count == 0
        async with gate.acquire(call):
            pass
        assert gate.active_scope_count == 0

    asyncio.run(scenario())
