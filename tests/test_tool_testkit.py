from __future__ import annotations

import asyncio
from dataclasses import replace
from types import MappingProxyType

import pytest

from tool_testkit import (
    FIXED_NOW,
    build_capability_exchange,
    exercise_compatibility_contract,
)

from OpenAgentLib.PluginHost import PluginHostRequest, PluginHostStatus
from OpenAgentLib.PluginSDK import CapabilityFamily
from OpenAgentLib.ToolCompatibility import TOOL_COMPATIBILITY_MATRIX
from OpenAgentLib.ToolKernel import ToolTraceState
from OpenAgentLib.ToolPolicy import (
    ConfirmationState,
    PolicyDecisionKind,
    ToolPolicyCatalog,
    ToolPolicyEngine,
)

pytestmark = pytest.mark.usefixtures("offline_network")


def test_native_call_builders_use_safe_authoritative_defaults(
    tool_spec_builder,
    tool_registry_builder,
) -> None:
    spec = tool_spec_builder()
    registry = tool_registry_builder((spec,))

    call = registry.create_call(
        call_id="call-native",
        requested_name="inspect",
        arguments={"value": "offline"},
    )

    assert call.spec is spec
    assert call.canonical_id == "sample.inspect"
    assert call.arguments == {"value": "offline"}


def test_confirmed_mutation_uses_real_policy_validation(
    tool_spec_builder,
    tool_call_builder,
    policy_rule_builder,
    policy_request_builder,
    confirmation_grant_builder,
) -> None:
    spec = tool_spec_builder(
        "sample.mutate",
        aliases=(),
        capabilities=frozenset({"write"}),
        confirmation="required",
        concurrency="serial",
        idempotency="non-idempotent",
    )
    call = tool_call_builder(spec)
    policy = ToolPolicyEngine(ToolPolicyCatalog((policy_rule_builder(spec),)))
    request = policy_request_builder(call)

    assert (
        policy.evaluate(call, request).kind is PolicyDecisionKind.CONFIRMATION_REQUIRED
    )
    confirmed = replace(
        request,
        confirmation=ConfirmationState.APPROVED,
        confirmation_grant=confirmation_grant_builder(call),
    )
    assert policy.evaluate(call, confirmed).kind is PolicyDecisionKind.ALLOW


def test_trace_collector_uses_fixed_timestamps(
    fixed_clock,
    tool_call_builder,
    trace_collector,
) -> None:
    call = tool_call_builder()
    trace_collector.start(call)
    fixed_clock.advance(seconds=1)
    trace = trace_collector.record(call, ToolTraceState.COMPLETED, {"count": 1})

    assert trace.created_at == FIXED_NOW
    assert trace.updated_at == FIXED_NOW.replace(second=1)
    assert trace.events[0].details == {"count": 1}


def test_fake_telegram_capability_dispatch_records_normalized_json(
    tool_spec_builder,
    tool_call_builder,
    policy_rule_builder,
    policy_request_builder,
    capability_broker_builder,
    fake_telegram_backend,
) -> None:
    spec = tool_spec_builder(
        "sample.telegram",
        aliases=(),
        capabilities=frozenset({"plugin-capability"}),
        concurrency="serial",
    )
    call = tool_call_builder(spec)
    policy = ToolPolicyEngine(ToolPolicyCatalog((policy_rule_builder(spec),)))
    policy_request = policy_request_builder(call)
    grant, request = build_capability_exchange(call)
    broker = capability_broker_builder(
        policy,
        {CapabilityFamily.TELEGRAM: fake_telegram_backend},
    )

    response = broker.dispatch(call, policy_request, grant, request)

    assert response.ok
    assert response.data == MappingProxyType(
        {"accepted": True, "message_id": "message-0001"}
    )
    assert fake_telegram_backend.calls == [
        {
            "grant_id": "capability-grant-0001",
            "operation": "send-message",
            "payload": {"data": {"peer_id": "peer-0001", "text": "hello"}},
        }
    ]


def test_sandboxed_host_uses_real_bwrap(sandboxed_host) -> None:
    outcome = asyncio.run(
        sandboxed_host.call(
            PluginHostRequest("host-request-real", "call-real", "ping", {})
        )
    )

    assert outcome.response.status is PluginHostStatus.SUCCESS
    assert strict_transport_marker(sandboxed_host) is False


def strict_transport_marker(host) -> bool:
    return bool(getattr(host, "transport_only", False))


def test_host_cancellation_reaps_strict_transport_child(
    strict_transport_host,
) -> None:
    error = asyncio.run(strict_transport_host.cancel_host())

    assert error.retryable is True
    assert strict_transport_host.process.transport_only is True
    assert strict_transport_host.process.stdin.closed
    assert strict_transport_host.process.stdout.closed
    assert strict_transport_host.process.stderr.closed


def test_gate_cancellation_releases_serial_scope(
    strict_transport_host,
    serial_cancellation_gate,
) -> None:
    gate, call = serial_cancellation_gate
    asyncio.run(strict_transport_host.cancel_gate(gate, call))

    assert gate.active_scope_count == 0


def test_offline_network_guard_reports_unexpected_access() -> None:
    import socket

    with pytest.raises(pytest.fail.Exception, match="unexpected network access"):
        socket.getaddrinfo("example.com", 443)


def test_compatibility_parametrization_uses_exact_canonical_id(
    compatibility_case,
    authoritative_policy_catalog,
    request: pytest.FixtureRequest,
) -> None:
    assert request.node.callspec.id == compatibility_case.canonical_id
    exercise_compatibility_contract(compatibility_case, authoritative_policy_catalog)


def test_contract_failure_names_exact_canonical_id(
    authoritative_policy_catalog,
) -> None:
    broken = replace(TOOL_COMPATIBILITY_MATRIX[0], capability_class="")

    with pytest.raises(AssertionError, match=rf"^{broken.canonical_id}:"):
        exercise_compatibility_contract(broken, authoritative_policy_catalog)
