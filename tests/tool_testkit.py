from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import sys
import threading
from typing import Any, Iterable, Iterator, Mapping

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def _runtime_import_path() -> Iterator[None]:
    source = str(ROOT / "Src")
    original = list(sys.path)
    try:
        if source not in sys.path:
            sys.path.insert(0, source)
        yield
    finally:
        sys.path[:] = original


with _runtime_import_path():
    from OpenAgentLib import PluginHost as plugin_host_module
    from OpenAgentLib.PluginCapabilities import CapabilityGrant, CapabilityRequest
    from OpenAgentLib.PluginHost import (
        PluginHostCallError,
        PluginHostErrorCode,
        PluginHostRequest,
    )
    from OpenAgentLib.PluginSDK import CapabilityFamily
    from OpenAgentLib.ToolCompatibility import ToolCompatibility
    from OpenAgentLib.ToolKernel import (
        TOOL_API_VERSION,
        TOOL_SCHEMA_VERSION,
        ConcurrencyClass,
        ConfirmationRequirement,
        IdempotencyClass,
        MigrationDisposition,
        ToolCall,
        ToolContext,
        ToolRegistry,
        ToolSpec,
        ToolTrace,
        ToolTraceEvent,
        ToolTraceState,
    )
    from OpenAgentLib.ToolPolicy import (
        ConfirmationState,
        PolicyDecisionKind,
        PolicyReasonCode,
        ToolConcurrencyGate,
        ToolConfirmationGrant,
        ToolPolicyCatalog,
        ToolPolicyEngine,
        ToolPolicyRequest,
        ToolPolicyRule,
    )


FIXED_NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


class FixedClock:
    def __init__(self, now: datetime = FIXED_NOW) -> None:
        self._now = now

    def __call__(self) -> datetime:
        return self._now

    def advance(self, **delta: float) -> datetime:
        self._now += timedelta(**delta)
        return self._now


def build_tool_spec(canonical_id: str = "sample.inspect", **overrides: Any) -> ToolSpec:
    values: dict[str, Any] = {
        "canonical_id": canonical_id,
        "aliases": ("inspect",),
        "input_schema": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "additionalProperties": False,
        },
        "output_schema": {"type": "object", "additionalProperties": True},
        "api_version": TOOL_API_VERSION,
        "schema_version": TOOL_SCHEMA_VERSION,
        "capabilities": frozenset({"read-only"}),
        "confirmation": ConfirmationRequirement.NONE,
        "concurrency": ConcurrencyClass.PARALLEL_READ,
        "idempotency": IdempotencyClass.IDEMPOTENT,
        "migration_disposition": MigrationDisposition.MIGRATE,
        "description": "Deterministic test tool",
        "source_family": "test",
        "source_module": "tests.tool_testkit",
    }
    values.update(overrides)
    return ToolSpec(**values)


def build_tool_call(spec: ToolSpec | None = None, **overrides: Any) -> ToolCall:
    selected = spec or build_tool_spec()
    values: dict[str, Any] = {
        "call_id": "call-0001",
        "spec": selected,
        "requested_name": selected.canonical_id,
        "arguments": {},
        "context": ToolContext("correlation-0001", "actor-0001"),
    }
    values.update(overrides)
    return ToolCall(**values)


def build_tool_registry(specs: Iterable[ToolSpec] | None = None) -> ToolRegistry:
    return ToolRegistry(tuple(specs) if specs is not None else (build_tool_spec(),))


def build_policy_rule(spec: ToolSpec | None = None, **overrides: Any) -> ToolPolicyRule:
    selected = spec or build_tool_spec()
    values: dict[str, Any] = {
        "canonical_id": selected.canonical_id,
        "capabilities": selected.capabilities,
        "confirmation": selected.confirmation,
        "concurrency": selected.concurrency,
        "idempotency": selected.idempotency,
        "migration_disposition": selected.migration_disposition,
    }
    values.update(overrides)
    return ToolPolicyRule(**values)


def build_policy_request(call: ToolCall, **overrides: Any) -> ToolPolicyRequest:
    values: dict[str, Any] = {
        "enabled_tool_ids": frozenset({call.spec.canonical_id}),
        "granted_capabilities": call.spec.capabilities,
        "requested_timeout": 5.0,
        "maximum_timeout": 5.0,
        "remaining_calls": 1,
        "remaining_token_budget": 100,
        "estimated_tokens": 1,
        "now": FIXED_NOW,
    }
    values.update(overrides)
    return ToolPolicyRequest(**values)


def build_confirmation_grant(call: ToolCall, **overrides: Any) -> ToolConfirmationGrant:
    values: dict[str, Any] = {
        "token": "confirmation-0001",
        "expires_at": FIXED_NOW + timedelta(minutes=5),
    }
    values.update(overrides)
    return ToolConfirmationGrant.for_call(call=call, **values)


class TraceCollector:
    def __init__(self, clock: FixedClock) -> None:
        self.clock = clock
        self._traces: dict[str, ToolTrace] = {}

    def start(self, call: ToolCall) -> ToolTrace:
        trace = ToolTrace.created(call, self.clock())
        self._traces[call.call_id] = trace
        return trace

    def record(
        self,
        call: ToolCall,
        state: ToolTraceState,
        details: Mapping[str, Any] | None = None,
    ) -> ToolTrace:
        previous = self._traces.get(call.call_id) or self.start(call)
        timestamp = self.clock()
        event = ToolTraceEvent(state, timestamp, details or {})
        trace = ToolTrace(
            previous.call_id,
            previous.correlation_id,
            state,
            previous.created_at,
            timestamp,
            (*previous.events, event),
        )
        self._traces[call.call_id] = trace
        return trace

    def get(self, call_id: str) -> ToolTrace:
        return self._traces[call_id]


def _json_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        value = {str(key): _json_copy(nested) for key, nested in value.items()}
    elif isinstance(value, (list, tuple)):
        value = [_json_copy(item) for item in value]
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


class JsonRecordingBackend:
    def __init__(self, response: Mapping[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = _json_copy(response or {"ok": True})

    def invoke(self, operation: str, payload: object, grant: object) -> dict[str, Any]:
        record = {
            "operation": str(operation),
            "payload": _json_copy(payload),
            "grant_id": str(getattr(grant, "grant_id")),
        }
        self.calls.append(record)
        return _json_copy(self.response)


class FakeTelegramBackend(JsonRecordingBackend):
    def __init__(self) -> None:
        super().__init__({"message_id": "message-0001", "accepted": True})


def build_capability_exchange(
    call: ToolCall,
    capability: CapabilityFamily = CapabilityFamily.TELEGRAM,
    operation: str = "send-message",
    payload: Mapping[str, Any] | None = None,
) -> tuple[CapabilityGrant, CapabilityRequest]:
    grant = CapabilityGrant.for_call(
        "capability-grant-0001",
        "host-request-0001",
        call,
        capability,
        frozenset({operation}),
    )
    request = CapabilityRequest(
        grant.host_request_id,
        call.call_id,
        call.spec.canonical_id,
        grant.actor_scope,
        grant.grant_id,
        capability,
        operation,
        "capability-request-0001",
        payload or {"data": {"peer_id": "peer-0001", "text": "hello"}},
    )
    return grant, request


def spec_from_compatibility(entry: ToolCompatibility) -> ToolSpec:
    return build_tool_spec(
        entry.canonical_id,
        aliases=entry.aliases,
        capabilities=frozenset({entry.capability_class}),
        confirmation=entry.confirmation_class,
        concurrency=entry.concurrency_class,
        idempotency=entry.idempotency_class,
        migration_disposition=entry.migration_disposition,
        source_family=entry.source_family,
        source_module=entry.source_module,
    )


def exercise_compatibility_contract(
    entry: ToolCompatibility,
    catalog: ToolPolicyCatalog,
) -> None:
    canonical_id = entry.canonical_id
    try:
        spec = spec_from_compatibility(entry)
        call = build_tool_call(spec, call_id=f"contract-{canonical_id}")
        engine = ToolPolicyEngine(catalog)
        request = build_policy_request(call)
        decision = engine.evaluate(call, request)

        if spec.migration_disposition is MigrationDisposition.REJECT:
            assert decision.reason is PolicyReasonCode.MIGRATION_REJECTED
            return
        if spec.confirmation is ConfirmationRequirement.REQUIRED:
            assert decision.kind is PolicyDecisionKind.CONFIRMATION_REQUIRED
            grant = build_confirmation_grant(call)
            request = replace(
                request,
                confirmation=ConfirmationState.APPROVED,
                confirmation_grant=grant,
            )
            decision = engine.evaluate(call, request)
        assert decision.kind is PolicyDecisionKind.ALLOW
    except Exception as exc:
        raise AssertionError(
            f"{canonical_id}: compatibility contract failed: {exc}"
        ) from exc


class _BlockingOutput(io.BytesIO):
    def __init__(self) -> None:
        super().__init__()
        self.read_started = threading.Event()
        self.released = threading.Event()

    def readline(self, size: int = -1) -> bytes:
        self.read_started.set()
        self.released.wait()
        return b""

    def close(self) -> None:
        self.released.set()
        super().close()


class StrictTransportProcess:
    """Minimal Popen double for PluginHost._exchange transport tests only."""

    transport_only = True
    pid = 987654

    def __init__(self) -> None:
        self.stdin = io.BytesIO()
        self.stdout = _BlockingOutput()
        self.stderr = io.BytesIO()
        self.returncode: int | None = None
        self.waited = threading.Event()

    def poll(self) -> int | None:
        return self.returncode

    def wait(self) -> int:
        self.stdout.released.wait()
        self.waited.set()
        return self.returncode or 0

    def terminate(self) -> None:
        self.returncode = -15
        self.stdout.released.set()

    def kill(self) -> None:
        self.returncode = -9
        self.stdout.released.set()


@dataclass
class CancellationHarness:
    host: Any
    process: StrictTransportProcess

    async def cancel_host(self) -> PluginHostCallError:
        request = PluginHostRequest(
            "host-request-cancel",
            "call-cancel",
            "ping",
            {},
            retryable=True,
        )
        task = asyncio.create_task(
            self.host._exchange(
                self.process, request, request.to_json_line(), 5.0, None
            )
        )
        read_started = await asyncio.to_thread(
            self.process.stdout.read_started.wait, 1.0
        )
        if not read_started:
            raise AssertionError("fake child did not reach the transport read barrier")
        task.cancel()
        cancellation_error: PluginHostCallError | None = None
        try:
            await task
        except PluginHostCallError as error:
            if error.code is not PluginHostErrorCode.CANCELLED:
                raise AssertionError(
                    f"unexpected cancellation code: {error.code}"
                ) from error
            cancellation_error = error
        else:
            raise AssertionError("cancelled host call unexpectedly completed")
        if not self.process.waited.is_set():
            raise AssertionError("cancelled fake child was not reaped")
        assert cancellation_error is not None
        return cancellation_error

    async def cancel_gate(self, gate: ToolConcurrencyGate, call: ToolCall) -> None:
        acquired = asyncio.Event()
        hold = asyncio.Event()

        async def child() -> None:
            async with gate.acquire(call):
                acquired.set()
                await hold.wait()

        task = asyncio.create_task(child())
        await acquired.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancelled gate holder unexpectedly completed")
        if gate.active_scope_count != 0:
            raise AssertionError("cancelled gate holder leaked its serial scope")


def install_fake_process_signals(
    monkeypatch: Any,
    process: StrictTransportProcess,
) -> None:
    def kill_process_group(pid: int, _signal: int) -> None:
        if pid != process.pid:
            raise AssertionError(f"unexpected fake process pid: {pid}")
        process.terminate()

    monkeypatch.setattr(plugin_host_module.os, "killpg", kill_process_group)


__all__ = [
    "CancellationHarness",
    "FIXED_NOW",
    "FakeTelegramBackend",
    "FixedClock",
    "JsonRecordingBackend",
    "StrictTransportProcess",
    "TraceCollector",
    "build_capability_exchange",
    "build_confirmation_grant",
    "build_policy_request",
    "build_policy_rule",
    "build_tool_call",
    "build_tool_registry",
    "build_tool_spec",
    "exercise_compatibility_contract",
    "install_fake_process_signals",
]
