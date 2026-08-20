from __future__ import annotations

import asyncio
import contextlib
import sys
import threading
import time
from typing import Any

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "Src"))

from OpenAgentLib.PluginHost import (  # noqa: E402
    PluginHostCallError,
    PluginHostErrorCode,
    PluginHostFailure,
    PluginHostOutcome,
    PluginHostRequest,
    PluginHostResponse,
    PluginHostStatus,
    PluginHostTrace,
    PluginHostTraceState,
)
from OpenAgentLib.ToolExecutor import (  # noqa: E402
    CooperativeSyncHandler,
    ToolExecutor,
    ToolExecutorError,
    ToolHookResult,
)
from OpenAgentLib.ToolKernel import (  # noqa: E402
    ToolErrorCode,
    ToolResultStatus,
    ToolTraceState,
    validate_schema_value,
)
from OpenAgentLib.ToolPolicy import (  # noqa: E402
    ToolPolicyCatalog,
    ToolPolicyEngine,
)


def _executor(
    tool_registry_builder: Any,
    policy_rule_builder: Any,
    specs: tuple[Any, ...],
    **kwargs: Any,
) -> ToolExecutor:
    registry = tool_registry_builder(specs)
    policy = ToolPolicyEngine(
        ToolPolicyCatalog(tuple(policy_rule_builder(spec) for spec in specs))
    )
    return ToolExecutor(registry, policy, **kwargs)


def test_close_cancels_and_reaps_timeout_operations(
    tool_registry_builder: Any, policy_rule_builder: Any, tool_spec_builder: Any
) -> None:
    async def scenario() -> None:
        executor = _executor(
            tool_registry_builder,
            policy_rule_builder,
            (tool_spec_builder("utility.wait"),),
        )
        started = asyncio.Event()
        release = asyncio.Event()

        async def operation() -> None:
            started.set()
            await release.wait()

        running = asyncio.create_task(executor._run_with_timeout(operation(), 30))
        await started.wait()
        await executor.close()
        with contextlib.suppress(asyncio.CancelledError):
            await running
        assert not executor._live_tasks

    asyncio.run(scenario())


class _Hooks:
    def __init__(self, before: Any = None, after: Any = None) -> None:
        self.before = before or ToolHookResult()
        self.after = after
        self.calls = 0

    async def before_execution(self, _call: Any) -> ToolHookResult:
        self.calls += 1
        if isinstance(self.before, BaseException):
            raise self.before
        return self.before

    async def after_execution(self, _call: Any, _result: Any) -> None:
        if isinstance(self.after, BaseException):
            raise self.after


class _TraceSink:
    def __init__(self) -> None:
        self.traces: list[Any] = []

    async def emit(self, trace: Any) -> None:
        self.traces.append(trace)


class _Host:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[str, bool]] = []

    async def invoke(
        self, call: Any, _request: Any, *, retryable: bool
    ) -> PluginHostOutcome:
        self.calls.append((call.call_id, retryable))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _Spill:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.results: list[Any] = []

    async def spill(self, _call: Any, result: Any) -> None:
        self.results.append(result)
        if self.error is not None:
            raise self.error


def _host_outcome(
    call: Any, output: Any = None, error: PluginHostFailure | None = None
) -> PluginHostOutcome:
    request = PluginHostRequest(
        f"request-{call.call_id}",
        call.call_id,
        "echo",
        {},
        retryable=error.retryable if error is not None else False,
    )
    status = PluginHostStatus.ERROR if error is not None else PluginHostStatus.SUCCESS
    state = (
        PluginHostTraceState.FAILED
        if error is not None
        else PluginHostTraceState.COMPLETED
    )
    trace = PluginHostTrace(request.request_id, call.call_id, state)
    response = PluginHostResponse(
        request.request_id, call.call_id, status, trace, output, error
    )
    return PluginHostOutcome(request, response)


def _request(policy_request_builder: Any, call: Any, **overrides: Any) -> Any:
    return policy_request_builder(call, **overrides)


def _cooperative(callback: Any) -> CooperativeSyncHandler:
    return CooperativeSyncHandler(callback)


def test_policy_denial_happens_before_hooks_or_handlers(
    tool_spec_builder: Any,
    tool_registry_builder: Any,
    tool_call_builder: Any,
    policy_rule_builder: Any,
    policy_request_builder: Any,
) -> None:
    spec = tool_spec_builder()
    call = tool_call_builder(spec)
    hooks = _Hooks()
    invoked = False

    def handler(_call: Any, _stop: threading.Event) -> dict[str, bool]:
        nonlocal invoked
        invoked = True
        return {"ok": True}

    executor = _executor(
        tool_registry_builder,
        policy_rule_builder,
        (spec,),
        native_handlers={spec.canonical_id: _cooperative(handler)},
        hooks=hooks,
    )
    result, trace = asyncio.run(
        executor.execute(
            call, _request(policy_request_builder, call, enabled_tool_ids=frozenset())
        )
    )

    assert result.error is not None and result.error.code is ToolErrorCode.POLICY_DENIED
    assert trace.state is ToolTraceState.FAILED
    assert hooks.calls == 0
    assert invoked is False


def test_registry_identity_and_policy_request_type_precede_hooks_or_handlers(
    tool_spec_builder: Any,
    tool_registry_builder: Any,
    tool_call_builder: Any,
    policy_rule_builder: Any,
    policy_request_builder: Any,
) -> None:
    registered_spec = tool_spec_builder()
    copied_spec = tool_spec_builder()
    copied_call = tool_call_builder(copied_spec)
    registered_call = tool_call_builder(registered_spec, call_id="call-request-type")
    hooks = _Hooks()
    invocations = 0

    def handler(_call: Any, _stop: threading.Event) -> dict[str, bool]:
        nonlocal invocations
        invocations += 1
        return {"ok": True}

    executor = _executor(
        tool_registry_builder,
        policy_rule_builder,
        (registered_spec,),
        native_handlers={registered_spec.canonical_id: _cooperative(handler)},
        hooks=hooks,
    )

    async def scenario() -> tuple[Any, Any]:
        return (
            await executor.execute(
                copied_call, _request(policy_request_builder, copied_call)
            ),
            await executor.execute(registered_call, object()),  # type: ignore[arg-type]
        )

    copied_outcome, invalid_request_outcome = asyncio.run(scenario())
    assert copied_outcome[0].error is not None
    assert copied_outcome[0].error.code is ToolErrorCode.INVALID_CALL
    assert invalid_request_outcome[0].error is not None
    assert invalid_request_outcome[0].error.code is ToolErrorCode.INVALID_CALL
    assert hooks.calls == 0
    assert invocations == 0


def test_alias_resolves_to_its_canonical_native_handler(
    tool_spec_builder: Any,
    tool_registry_builder: Any,
    policy_rule_builder: Any,
    policy_request_builder: Any,
) -> None:
    spec = tool_spec_builder(
        output_schema={"type": "object", "additionalProperties": True}
    )
    registry = tool_registry_builder((spec,))
    call = registry.create_call(
        call_id="call-alias", requested_name="inspect", arguments={}
    )
    seen: list[str] = []
    spill = _Spill()

    def handler(received: Any, _stop: threading.Event) -> dict[str, str]:
        seen.append(received.canonical_id)
        return {"handler": received.canonical_id}

    executor = _executor(
        tool_registry_builder,
        policy_rule_builder,
        (spec,),
        native_handlers={spec.canonical_id: _cooperative(handler)},
        context_spill=spill,
    )
    result, _trace = asyncio.run(
        executor.execute(call, _request(policy_request_builder, call))
    )

    assert result.status is ToolResultStatus.SUCCESS
    assert result.output == {"handler": spec.canonical_id}
    assert seen == [spec.canonical_id]
    assert spill.results == [result]
    with pytest.raises(TypeError):
        result.output["handler"] = "mutated"


def test_native_cooperative_sync_and_async_handlers_are_supported(
    tool_spec_builder: Any,
    tool_registry_builder: Any,
    tool_call_builder: Any,
    policy_rule_builder: Any,
    policy_request_builder: Any,
) -> None:
    sync_spec = tool_spec_builder("sample.sync", aliases=())
    async_spec = tool_spec_builder("sample.async", aliases=())
    sync_call = tool_call_builder(sync_spec, call_id="call-sync")
    async_call = tool_call_builder(async_spec, call_id="call-async")

    def sync_handler(_call: Any, _stop: threading.Event) -> dict[str, str]:
        return {"kind": "sync"}

    async def async_handler(_call: Any) -> dict[str, str]:
        return {"kind": "async"}

    executor = _executor(
        tool_registry_builder,
        policy_rule_builder,
        (sync_spec, async_spec),
        native_handlers={
            sync_spec.canonical_id: _cooperative(sync_handler),
            async_spec.canonical_id: async_handler,
        },
    )

    async def scenario() -> list[Any]:
        return [
            await executor.execute(
                sync_call, _request(policy_request_builder, sync_call)
            ),
            await executor.execute(
                async_call, _request(policy_request_builder, async_call)
            ),
        ]

    outcomes = asyncio.run(scenario())
    assert [outcome[0].output for outcome in outcomes] == [
        {"kind": "sync"},
        {"kind": "async"},
    ]


def test_arbitrary_sync_native_handler_is_rejected_without_invocation(
    tool_spec_builder: Any,
    tool_registry_builder: Any,
    policy_rule_builder: Any,
) -> None:
    spec = tool_spec_builder()
    invoked = False

    def arbitrary_sync_handler(_call: Any) -> dict[str, bool]:
        nonlocal invoked
        invoked = True
        return {"unexpected": True}

    with pytest.raises(ToolExecutorError) as error:
        _executor(
            tool_registry_builder,
            policy_rule_builder,
            (spec,),
            native_handlers={spec.canonical_id: arbitrary_sync_handler},
        )

    assert error.value.code is ToolErrorCode.INVALID_SPEC
    assert invoked is False


def test_host_success_and_failure_are_normalized_without_secret_messages(
    tool_spec_builder: Any,
    tool_registry_builder: Any,
    tool_call_builder: Any,
    policy_rule_builder: Any,
    policy_request_builder: Any,
) -> None:
    spec = tool_spec_builder()
    success_call = tool_call_builder(spec, call_id="call-host-success")
    failed_call = tool_call_builder(spec, call_id="call-host-failure")
    failure = PluginHostFailure(
        PluginHostErrorCode.CHILD_CRASHED,
        "secret host detail",
        call_id=failed_call.call_id,
        retryable=True,
    )
    host = _Host(
        [
            _host_outcome(success_call, {"host": "ok"}),
            _host_outcome(failed_call, error=failure),
        ]
    )
    executor = _executor(
        tool_registry_builder, policy_rule_builder, (spec,), host_invoker=host
    )

    async def scenario() -> list[Any]:
        return [
            await executor.execute(
                success_call, _request(policy_request_builder, success_call)
            ),
            await executor.execute(
                failed_call, _request(policy_request_builder, failed_call)
            ),
        ]

    outcomes = asyncio.run(scenario())
    assert outcomes[0][0].output == {"host": "ok"}
    assert outcomes[1][0].error is not None
    assert outcomes[1][0].error.code is ToolErrorCode.HOST_FAILED
    assert "secret" not in outcomes[1][0].error.message
    assert outcomes[1][0].retryable is True


def test_output_schema_validation_freezes_successful_output(
    tool_spec_builder: Any,
    tool_registry_builder: Any,
    tool_call_builder: Any,
    policy_rule_builder: Any,
    policy_request_builder: Any,
) -> None:
    spec = tool_spec_builder(
        output_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    )
    call = tool_call_builder(spec)
    executor = _executor(
        tool_registry_builder,
        policy_rule_builder,
        (spec,),
        native_handlers={
            spec.canonical_id: _cooperative(lambda _call, _stop: {"answer": 3})
        },
    )
    result, _trace = asyncio.run(
        executor.execute(call, _request(policy_request_builder, call))
    )
    frozen = validate_schema_value(
        {"type": "array", "items": {"type": "string"}}, ["ok"]
    )

    assert (
        result.error is not None
        and result.error.code is ToolErrorCode.OUTPUT_SCHEMA_INVALID
    )
    assert frozen == ("ok",)
    with pytest.raises(AttributeError):
        frozen.append("no")


def test_confirmation_and_hook_outcomes_are_typed(
    tool_spec_builder: Any,
    tool_registry_builder: Any,
    tool_call_builder: Any,
    policy_rule_builder: Any,
    policy_request_builder: Any,
) -> None:
    confirmation_spec = tool_spec_builder(
        "sample.confirm", aliases=(), confirmation="required", concurrency="serial"
    )
    hook_spec = tool_spec_builder("sample.hook", aliases=())
    confirmation_call = tool_call_builder(confirmation_spec, call_id="call-confirm")
    hook_call = tool_call_builder(hook_spec, call_id="call-hook")
    cancel_hooks = _Hooks(ToolHookResult.cancelled())
    bad_hooks = _Hooks(RuntimeError("secret hook failure"))
    confirmation_executor = _executor(
        tool_registry_builder,
        policy_rule_builder,
        (confirmation_spec,),
        native_handlers={
            confirmation_spec.canonical_id: _cooperative(lambda _call, _stop: {})
        },
    )
    cancel_executor = _executor(
        tool_registry_builder,
        policy_rule_builder,
        (hook_spec,),
        native_handlers={hook_spec.canonical_id: _cooperative(lambda _call, _stop: {})},
        hooks=cancel_hooks,
    )
    failed_hook_executor = _executor(
        tool_registry_builder,
        policy_rule_builder,
        (hook_spec,),
        native_handlers={hook_spec.canonical_id: _cooperative(lambda _call, _stop: {})},
        hooks=bad_hooks,
    )

    async def scenario() -> list[Any]:
        return [
            await confirmation_executor.execute(
                confirmation_call,
                _request(policy_request_builder, confirmation_call),
            ),
            await cancel_executor.execute(
                hook_call, _request(policy_request_builder, hook_call)
            ),
            await failed_hook_executor.execute(
                hook_call, _request(policy_request_builder, hook_call)
            ),
        ]

    outcomes = asyncio.run(scenario())
    assert outcomes[0][0].error is not None
    assert outcomes[0][0].error.code is ToolErrorCode.CONFIRMATION_REQUIRED
    assert outcomes[1][0].status is ToolResultStatus.CANCELLED
    assert (
        outcomes[1][0].error is not None
        and outcomes[1][0].error.code is ToolErrorCode.HOOK_CANCELLED
    )
    assert (
        outcomes[2][0].error is not None
        and outcomes[2][0].error.code is ToolErrorCode.HOOK_FAILED
    )
    assert "secret" not in outcomes[2][0].error.message


def test_serial_calls_wait_and_parallel_reads_overlap(
    tool_spec_builder: Any,
    tool_registry_builder: Any,
    tool_call_builder: Any,
    policy_rule_builder: Any,
    policy_request_builder: Any,
) -> None:
    serial_spec = tool_spec_builder("sample.serial", aliases=(), concurrency="serial")
    parallel_spec = tool_spec_builder("sample.parallel", aliases=())
    serial_one = tool_call_builder(serial_spec, call_id="call-serial-one")
    serial_two = tool_call_builder(serial_spec, call_id="call-serial-two")
    parallel_one = tool_call_builder(parallel_spec, call_id="call-parallel-one")
    parallel_two = tool_call_builder(parallel_spec, call_id="call-parallel-two")
    first_serial_started = asyncio.Event()
    second_serial_started = asyncio.Event()
    release_serial = asyncio.Event()
    parallel_started = 0
    parallel_ready = asyncio.Event()
    release_parallel = asyncio.Event()

    async def serial_handler(call: Any) -> dict[str, str]:
        if call.call_id == serial_one.call_id:
            first_serial_started.set()
            await release_serial.wait()
        else:
            second_serial_started.set()
        return {"call": call.call_id}

    async def parallel_handler(call: Any) -> dict[str, str]:
        nonlocal parallel_started
        parallel_started += 1
        if parallel_started == 2:
            parallel_ready.set()
        await release_parallel.wait()
        return {"call": call.call_id}

    executor = _executor(
        tool_registry_builder,
        policy_rule_builder,
        (serial_spec, parallel_spec),
        native_handlers={
            serial_spec.canonical_id: serial_handler,
            parallel_spec.canonical_id: parallel_handler,
        },
    )

    async def scenario() -> None:
        serial_task = asyncio.create_task(
            executor.execute_batch(
                (serial_one, serial_two),
                (
                    _request(policy_request_builder, serial_one),
                    _request(policy_request_builder, serial_two),
                ),
            )
        )
        await first_serial_started.wait()
        await asyncio.sleep(0)
        assert not second_serial_started.is_set()
        release_serial.set()
        await serial_task

        parallel_task = asyncio.create_task(
            executor.execute_batch(
                (parallel_one, parallel_two),
                (
                    _request(policy_request_builder, parallel_one),
                    _request(policy_request_builder, parallel_two),
                ),
            )
        )
        await asyncio.wait_for(parallel_ready.wait(), 1)
        release_parallel.set()
        await parallel_task

    asyncio.run(scenario())


def test_mixed_batch_preserves_results_and_traces_in_input_order(
    tool_spec_builder: Any,
    tool_registry_builder: Any,
    tool_call_builder: Any,
    policy_rule_builder: Any,
    policy_request_builder: Any,
) -> None:
    native_spec = tool_spec_builder("sample.native", aliases=())
    denied_spec = tool_spec_builder("sample.denied", aliases=())
    host_spec = tool_spec_builder("sample.host", aliases=())
    native_call = tool_call_builder(native_spec, call_id="call-native")
    denied_call = tool_call_builder(denied_spec, call_id="call-denied")
    host_call = tool_call_builder(host_spec, call_id="call-host")
    host = _Host([_host_outcome(host_call, {"source": "host"})])
    executor = _executor(
        tool_registry_builder,
        policy_rule_builder,
        (native_spec, denied_spec, host_spec),
        native_handlers={
            native_spec.canonical_id: _cooperative(
                lambda _call, _stop: {"source": "native"}
            )
        },
        host_invoker=host,
    )
    results, traces = asyncio.run(
        executor.execute_batch(
            (native_call, denied_call, host_call),
            (
                _request(policy_request_builder, native_call),
                _request(
                    policy_request_builder, denied_call, enabled_tool_ids=frozenset()
                ),
                _request(policy_request_builder, host_call),
            ),
        )
    )

    assert [result.call_id for result in results] == [
        call.call_id for call in (native_call, denied_call, host_call)
    ]
    assert [trace.call_id for trace in traces] == [
        call.call_id for call in (native_call, denied_call, host_call)
    ]
    assert results[0].output == {"source": "native"}
    assert (
        results[1].error is not None
        and results[1].error.code is ToolErrorCode.POLICY_DENIED
    )
    assert results[2].output == {"source": "host"}


def test_timeout_and_cancellation_reap_native_task(
    tool_spec_builder: Any,
    tool_registry_builder: Any,
    tool_call_builder: Any,
    policy_rule_builder: Any,
    policy_request_builder: Any,
) -> None:
    spec = tool_spec_builder()
    timeout_call = tool_call_builder(spec, call_id="call-timeout")
    cancel_call = tool_call_builder(spec, call_id="call-cancel")
    timeout_cleanup = asyncio.Event()
    cancel_started = asyncio.Event()
    cancel_cleanup = asyncio.Event()

    async def handler(call: Any) -> dict[str, bool]:
        try:
            if call.call_id == timeout_call.call_id:
                await asyncio.Event().wait()
            cancel_started.set()
            await asyncio.Event().wait()
        finally:
            if call.call_id == timeout_call.call_id:
                timeout_cleanup.set()
            else:
                cancel_cleanup.set()
        return {"unreachable": True}

    executor = _executor(
        tool_registry_builder,
        policy_rule_builder,
        (spec,),
        native_handlers={spec.canonical_id: handler},
    )

    async def scenario() -> tuple[Any, Any]:
        timeout_result = await executor.execute(
            timeout_call,
            _request(
                policy_request_builder,
                timeout_call,
                requested_timeout=0.01,
                maximum_timeout=1,
            ),
        )
        task = asyncio.create_task(
            executor.execute(cancel_call, _request(policy_request_builder, cancel_call))
        )
        await cancel_started.wait()
        task.cancel()
        cancelled_result = await task
        return timeout_result, cancelled_result

    timeout_outcome, cancelled_outcome = asyncio.run(scenario())
    assert timeout_outcome[0].status is ToolResultStatus.TIMED_OUT
    assert cancelled_outcome[0].status is ToolResultStatus.CANCELLED
    assert timeout_cleanup.is_set()
    assert cancel_cleanup.is_set()


def test_cooperative_sync_timeout_signals_cleanup_before_terminal_result(
    tool_spec_builder: Any,
    tool_registry_builder: Any,
    tool_call_builder: Any,
    policy_rule_builder: Any,
    policy_request_builder: Any,
) -> None:
    spec = tool_spec_builder()
    call = tool_call_builder(spec, call_id="call-cooperative-timeout")
    started = threading.Event()
    stop_observed = threading.Event()
    cleaned_up = threading.Event()
    mutations: list[str] = []

    def handler(_call: Any, stop: threading.Event) -> dict[str, bool]:
        started.set()
        if not stop.wait(1):
            return {"completed": True}
        stop_observed.set()
        time.sleep(0.02)
        mutations.append("cleanup-complete")
        cleaned_up.set()
        return {"completed": False}

    executor = _executor(
        tool_registry_builder,
        policy_rule_builder,
        (spec,),
        native_handlers={spec.canonical_id: _cooperative(handler)},
    )

    async def scenario() -> Any:
        task = asyncio.create_task(
            executor.execute(
                call,
                _request(
                    policy_request_builder,
                    call,
                    requested_timeout=0.02,
                    maximum_timeout=1,
                ),
            )
        )
        assert await asyncio.to_thread(started.wait, 1)
        return await task

    result, _trace = asyncio.run(scenario())
    assert result.status is ToolResultStatus.TIMED_OUT
    assert stop_observed.is_set()
    assert cleaned_up.is_set()
    mutation_snapshot = list(mutations)
    time.sleep(0.05)
    assert mutations == mutation_snapshot == ["cleanup-complete"]


def test_cooperative_sync_cancellation_signals_cleanup_before_terminal_result(
    tool_spec_builder: Any,
    tool_registry_builder: Any,
    tool_call_builder: Any,
    policy_rule_builder: Any,
    policy_request_builder: Any,
) -> None:
    spec = tool_spec_builder()
    call = tool_call_builder(spec, call_id="call-cooperative-cancel")
    started = threading.Event()
    stop_observed = threading.Event()
    cleaned_up = threading.Event()

    def handler(_call: Any, stop: threading.Event) -> dict[str, bool]:
        started.set()
        if not stop.wait(1):
            return {"completed": True}
        stop_observed.set()
        cleaned_up.set()
        return {"completed": False}

    executor = _executor(
        tool_registry_builder,
        policy_rule_builder,
        (spec,),
        native_handlers={spec.canonical_id: _cooperative(handler)},
    )

    async def scenario() -> Any:
        task = asyncio.create_task(
            executor.execute(call, _request(policy_request_builder, call))
        )
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        return await task

    result, _trace = asyncio.run(scenario())
    assert result.status is ToolResultStatus.CANCELLED
    assert stop_observed.is_set()
    assert cleaned_up.is_set()


def test_retry_is_limited_to_idempotent_host_failures(
    tool_spec_builder: Any,
    tool_registry_builder: Any,
    tool_call_builder: Any,
    policy_rule_builder: Any,
    policy_request_builder: Any,
) -> None:
    spec = tool_spec_builder()
    call = tool_call_builder(spec)
    crash = PluginHostCallError(
        PluginHostFailure(PluginHostErrorCode.CHILD_CRASHED, "secret", retryable=True)
    )
    host = _Host([crash, _host_outcome(call, {"attempt": "two"})])
    sink = _TraceSink()
    executor = _executor(
        tool_registry_builder,
        policy_rule_builder,
        (spec,),
        host_invoker=host,
        trace_sink=sink,
    )
    result, trace = asyncio.run(
        executor.execute(
            call, _request(policy_request_builder, call, maximum_attempts=2)
        )
    )

    assert result.output == {"attempt": "two"}
    assert len(host.calls) == 2
    assert [
        event.details.get("attempt")
        for event in trace.events
        if "attempt" in event.details
    ] == [1, 2]
    assert sink.traces[-1] == trace


def test_non_idempotent_failure_and_spill_failure_never_retry(
    tool_spec_builder: Any,
    tool_registry_builder: Any,
    tool_call_builder: Any,
    policy_rule_builder: Any,
    policy_request_builder: Any,
) -> None:
    mutation_spec = tool_spec_builder(
        "sample.mutation",
        aliases=(),
        idempotency="non-idempotent",
        concurrency="serial",
    )
    mutation_call = tool_call_builder(mutation_spec, call_id="call-mutation")
    host = _Host(
        [
            PluginHostCallError(
                PluginHostFailure(
                    PluginHostErrorCode.CHILD_CRASHED, "secret", retryable=True
                )
            )
        ]
    )
    mutation_executor = _executor(
        tool_registry_builder,
        policy_rule_builder,
        (mutation_spec,),
        host_invoker=host,
    )
    spill_spec = tool_spec_builder("sample.spill", aliases=())
    spill_call = tool_call_builder(spill_spec, call_id="call-spill")
    invocations = 0

    def handler(_call: Any, _stop: threading.Event) -> dict[str, bool]:
        nonlocal invocations
        invocations += 1
        return {"ok": True}

    spill = _Spill(RuntimeError("spill unavailable"))
    spill_executor = _executor(
        tool_registry_builder,
        policy_rule_builder,
        (spill_spec,),
        native_handlers={spill_spec.canonical_id: _cooperative(handler)},
        context_spill=spill,
    )

    async def scenario() -> tuple[Any, Any]:
        return (
            await mutation_executor.execute(
                mutation_call,
                _request(policy_request_builder, mutation_call, maximum_attempts=2),
            ),
            await spill_executor.execute(
                spill_call,
                _request(policy_request_builder, spill_call, maximum_attempts=2),
            ),
        )

    mutation_outcome, spill_outcome = asyncio.run(scenario())
    assert mutation_outcome[0].error is not None
    assert mutation_outcome[0].error.code is ToolErrorCode.HOST_FAILED
    assert len(host.calls) == 1
    assert spill_outcome[0].error is not None
    assert spill_outcome[0].error.code is ToolErrorCode.SPILL_FAILED
    assert invocations == 1
    assert spill.results[0].status is ToolResultStatus.SUCCESS


def test_duplicate_call_ids_are_rejected_before_batch_execution(
    tool_spec_builder: Any,
    tool_registry_builder: Any,
    tool_call_builder: Any,
    policy_rule_builder: Any,
    policy_request_builder: Any,
) -> None:
    spec = tool_spec_builder()
    one = tool_call_builder(spec, call_id="call-duplicate")
    two = tool_call_builder(spec, call_id="call-duplicate")
    executor = _executor(tool_registry_builder, policy_rule_builder, (spec,))

    with pytest.raises(ToolExecutorError) as error:
        asyncio.run(
            executor.execute_batch(
                (one, two),
                (
                    _request(policy_request_builder, one),
                    _request(policy_request_builder, two),
                ),
            )
        )

    assert error.value.code is ToolErrorCode.DUPLICATE_CALL_ID


def test_batch_length_mismatch_is_rejected_before_execution(
    tool_spec_builder: Any,
    tool_registry_builder: Any,
    tool_call_builder: Any,
    policy_rule_builder: Any,
    policy_request_builder: Any,
) -> None:
    spec = tool_spec_builder()
    call = tool_call_builder(spec)
    executor = _executor(tool_registry_builder, policy_rule_builder, (spec,))

    with pytest.raises(ToolExecutorError) as error:
        asyncio.run(executor.execute_batch((call,), ()))

    assert error.value.code is ToolErrorCode.BATCH_LENGTH_MISMATCH
