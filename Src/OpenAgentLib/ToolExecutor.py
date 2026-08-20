# SPDX-License-Identifier: MIT
"""Policy-first v2 tool execution with immutable terminal results and traces.

This module is deliberately standalone: callers inject native handlers, an
isolated host invoker, hooks, trace storage, and optional context spilling.
It never imports legacy dispatch or plugin modules.  Cancelling ``execute``
returns a ``CANCELLED`` result after cancelling and awaiting its native/host
task.  Synchronous native handlers must use ``CooperativeSyncHandler`` and
return after their stop event is set; the executor never returns a terminal
result until that handler has finished cleanup.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import inspect
import threading
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

from .PluginHost import (
    PluginHostCallError,
    PluginHostErrorCode,
    PluginHostOutcome,
    PluginHostStatus,
)
from .ToolKernel import (
    IdempotencyClass,
    ToolArgumentError,
    ToolCall,
    ToolError,
    ToolErrorCode,
    ToolKernelError,
    ToolRegistry,
    ToolResult,
    ToolResultStatus,
    ToolTrace,
    ToolTraceEvent,
    ToolTraceState,
    normalize_tool_name,
    validate_schema_value,
)
from .ToolPolicy import (
    PolicyDecisionKind,
    ToolConcurrencyGate,
    ToolPolicyEngine,
    ToolPolicyRequest,
)


class ToolHookAction(str, Enum):
    """The only control outcome a lifecycle hook may return."""

    CONTINUE = "continue"
    CANCEL = "cancel"


@dataclass(frozen=True)
class ToolHookResult:
    """An immutable lifecycle decision that cannot alter call metadata."""

    action: ToolHookAction = ToolHookAction.CONTINUE

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", ToolHookAction(self.action))

    @classmethod
    def cancelled(cls) -> "ToolHookResult":
        return cls(ToolHookAction.CANCEL)


def _is_async_callable(value: Any) -> bool:
    return inspect.iscoroutinefunction(value) or inspect.iscoroutinefunction(
        getattr(value, "__call__", None)
    )


class NativeToolHandler(Protocol):
    """An async native implementation selected only by canonical tool ID."""

    def __call__(self, call: ToolCall) -> Awaitable[Any]: ...


@dataclass(frozen=True)
class CooperativeSyncHandler:
    """Explicit lifecycle contract for trusted synchronous native handlers.

    ``callback`` runs off the event loop and receives a stop event.  It must
    observe that event, finish cleanup, and return promptly.  The executor
    waits for that return on timeout or cancellation, so a non-cooperative
    callback stalls the call instead of leaving a side effect running after a
    terminal result.
    """

    callback: Callable[[ToolCall, threading.Event], Any]

    def __post_init__(self) -> None:
        if not callable(self.callback) or _is_async_callable(self.callback):
            raise TypeError("cooperative sync handlers require a synchronous callable")


class ToolHostInvoker(Protocol):
    """A host boundary that returns one correlated isolated-host outcome."""

    async def invoke(self, call: ToolCall, *, retryable: bool) -> PluginHostOutcome: ...


class ToolLifecycleHooks(Protocol):
    """Optional immutable lifecycle notifications around authorized execution."""

    async def before_execution(self, call: ToolCall) -> ToolHookResult: ...

    async def after_execution(self, call: ToolCall, result: ToolResult) -> None: ...


class ToolTraceSink(Protocol):
    """Receives immutable traces containing only stable state and reason codes."""

    async def emit(self, trace: ToolTrace) -> None: ...


class ContextSpillAdapter(Protocol):
    """Persists a validated success result after execution has completed."""

    async def spill(self, call: ToolCall, result: ToolResult) -> None: ...


class ToolExecutorError(ValueError):
    """Typed misuse errors that cannot be associated with one valid call."""

    def __init__(self, code: ToolErrorCode, message: str) -> None:
        self.code = ToolErrorCode(code)
        super().__init__(message)


@dataclass
class _TracePipeline:
    trace: ToolTrace


class ToolExecutor:
    """Execute immutable calls only after registry validation and policy approval."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: ToolPolicyEngine,
        *,
        native_handlers: (
            Mapping[str, NativeToolHandler | CooperativeSyncHandler] | None
        ) = None,
        host_invoker: ToolHostInvoker | None = None,
        hooks: ToolLifecycleHooks | None = None,
        trace_sink: ToolTraceSink | None = None,
        context_spill: ContextSpillAdapter | None = None,
        gate: ToolConcurrencyGate | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(registry, ToolRegistry):
            raise TypeError("registry must be a ToolRegistry")
        if not isinstance(policy, ToolPolicyEngine):
            raise TypeError("policy must be a ToolPolicyEngine")
        self.registry = registry
        self.policy = policy
        self.gate = gate or ToolConcurrencyGate(policy)
        self.host_invoker = host_invoker
        self.hooks = hooks
        self.trace_sink = trace_sink
        self.context_spill = context_spill
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._native_handlers = self._validate_native_handlers(native_handlers or {})
        self._live_tasks: set[asyncio.Task[Any]] = set()

    async def close(self) -> None:
        """Cancel and await every timeout-wrapped operation still in flight."""

        tasks = tuple(self._live_tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._live_tasks.difference_update(tasks)

    async def on_unload(self) -> None:
        """Lifecycle hook used by the owning v2 runtime."""

        await self.close()

    async def execute(
        self,
        call: ToolCall,
        request: ToolPolicyRequest,
    ) -> tuple[ToolResult, ToolTrace]:
        """Return one terminal result/trace pair without propagating cancellation."""

        if not isinstance(call, ToolCall):
            raise ToolExecutorError(
                ToolErrorCode.INVALID_CALL, "executor requires a ToolCall"
            )
        pipeline = _TracePipeline(ToolTrace.created(call, self._now()))
        await self._emit(pipeline.trace)
        try:
            result = await self._execute(call, request, pipeline)
        except asyncio.CancelledError:
            result = self._result(
                call,
                ToolResultStatus.CANCELLED,
                ToolErrorCode.CANCELLED,
                "tool execution was cancelled",
            )
        except Exception:
            result = self._result(
                call,
                ToolResultStatus.ERROR,
                ToolErrorCode.EXECUTOR_FAILED,
                "tool executor failed",
            )
        await self._record_terminal(pipeline, result)
        return result, pipeline.trace

    async def execute_batch(
        self,
        calls: Sequence[ToolCall],
        requests: Sequence[ToolPolicyRequest],
    ) -> tuple[tuple[ToolResult, ...], tuple[ToolTrace, ...]]:
        """Execute independent calls concurrently and preserve input positions."""

        call_items = tuple(calls)
        request_items = tuple(requests)
        if len(call_items) != len(request_items):
            raise ToolExecutorError(
                ToolErrorCode.BATCH_LENGTH_MISMATCH,
                "calls and policy requests must have matching lengths",
            )
        if any(not isinstance(call, ToolCall) for call in call_items):
            raise ToolExecutorError(
                ToolErrorCode.INVALID_CALL, "executor requires ToolCall values"
            )
        call_ids = tuple(call.call_id for call in call_items)
        if len(call_ids) != len(set(call_ids)):
            raise ToolExecutorError(
                ToolErrorCode.DUPLICATE_CALL_ID,
                "batch call IDs must be unique",
            )
        completed = await asyncio.gather(
            *(
                self.execute(call, request)
                for call, request in zip(call_items, request_items)
            ),
            return_exceptions=False,
        )
        return (
            tuple(result for result, _trace in completed),
            tuple(trace for _result, trace in completed),
        )

    async def _execute(
        self,
        call: ToolCall,
        request: ToolPolicyRequest,
        pipeline: _TracePipeline,
    ) -> ToolResult:
        try:
            self.registry.validate_call(call)
        except ToolKernelError:
            return self._result(
                call,
                ToolResultStatus.ERROR,
                ToolErrorCode.INVALID_CALL,
                "tool call does not match the authoritative registry",
            )
        if type(request) is not ToolPolicyRequest:
            return self._result(
                call,
                ToolResultStatus.ERROR,
                ToolErrorCode.INVALID_CALL,
                "tool execution requires a ToolPolicyRequest",
            )

        await self._record(pipeline, ToolTraceState.VALIDATED, {"stage": "registry"})
        current_request = request
        while True:
            result = await self._attempt(call, current_request, pipeline)
            if not self.policy.retry_eligible(call, result, current_request):
                return result
            current_request = replace(
                current_request,
                retry_attempt=current_request.retry_attempt + 1,
            )

    async def _attempt(
        self,
        call: ToolCall,
        request: ToolPolicyRequest,
        pipeline: _TracePipeline,
    ) -> ToolResult:
        try:
            decision = self.policy.evaluate(call, request)
        except Exception:
            return self._result(
                call,
                ToolResultStatus.ERROR,
                ToolErrorCode.EXECUTOR_FAILED,
                "tool policy evaluation failed",
            )
        await self._record(
            pipeline,
            ToolTraceState.VALIDATED,
            {
                "attempt": request.retry_attempt,
                "policy": decision.kind.value,
                "reason": decision.reason.value,
            },
        )
        if decision.kind is PolicyDecisionKind.CONFIRMATION_REQUIRED:
            return self._result(
                call,
                ToolResultStatus.ERROR,
                ToolErrorCode.CONFIRMATION_REQUIRED,
                "tool confirmation is required",
            )
        if decision.kind is not PolicyDecisionKind.ALLOW:
            return self._result(
                call,
                ToolResultStatus.ERROR,
                ToolErrorCode.POLICY_DENIED,
                "tool execution was denied by policy",
            )

        if self.hooks is not None:
            try:
                hook_result = await self.hooks.before_execution(call)
                if not isinstance(hook_result, ToolHookResult):
                    raise TypeError("before_execution must return ToolHookResult")
            except asyncio.CancelledError:
                raise
            except Exception:
                return self._result(
                    call,
                    ToolResultStatus.ERROR,
                    ToolErrorCode.HOOK_FAILED,
                    "tool lifecycle hook failed",
                )
            if hook_result.action is ToolHookAction.CANCEL:
                return self._result(
                    call,
                    ToolResultStatus.CANCELLED,
                    ToolErrorCode.HOOK_CANCELLED,
                    "tool execution was cancelled by a lifecycle hook",
                )

        async with self.gate.acquire(call):
            result = await self._invoke(call, request)
        if result.status is not ToolResultStatus.SUCCESS:
            return result

        if self.context_spill is not None:
            try:
                await self.context_spill.spill(call, result)
            except asyncio.CancelledError:
                raise
            except Exception:
                return self._result(
                    call,
                    ToolResultStatus.ERROR,
                    ToolErrorCode.SPILL_FAILED,
                    "tool result context spill failed",
                )

        if self.hooks is not None:
            try:
                await self.hooks.after_execution(call, result)
            except asyncio.CancelledError:
                raise
            except Exception:
                return self._result(
                    call,
                    ToolResultStatus.ERROR,
                    ToolErrorCode.HOOK_FAILED,
                    "tool lifecycle hook failed",
                )
        return result

    async def _invoke(self, call: ToolCall, request: ToolPolicyRequest) -> ToolResult:
        try:
            output = await self._run_with_timeout(
                self._dispatch(call),
                request.requested_timeout,
            )
        except asyncio.TimeoutError:
            return self._result(
                call,
                ToolResultStatus.TIMED_OUT,
                ToolErrorCode.TIMED_OUT,
                "tool execution timed out",
            )
        except asyncio.CancelledError:
            raise
        except PluginHostCallError as error:
            return self._host_failure(call, error.code, error.retryable)
        except _HostFailure as failure:
            return self._host_failure(call, failure.code, failure.retryable)
        except Exception:
            return self._result(
                call,
                ToolResultStatus.ERROR,
                ToolErrorCode.HANDLER_FAILED,
                "tool handler failed",
            )

        try:
            frozen_output = validate_schema_value(
                call.spec.output_schema,
                output,
                canonical_id=call.canonical_id,
                requested_name=call.requested_name,
            )
        except ToolArgumentError as error:
            return self._result(
                call,
                ToolResultStatus.ERROR,
                ToolErrorCode.OUTPUT_SCHEMA_INVALID,
                "tool output violates the declared schema",
                field_path=error.field_path,
            )
        return ToolResult(call.call_id, ToolResultStatus.SUCCESS, output=frozen_output)

    async def _dispatch(self, call: ToolCall) -> Any:
        handler = self._native_handlers.get(call.canonical_id)
        if handler is not None:
            return await self._invoke_native(handler, call)
        if self.host_invoker is None:
            raise RuntimeError(
                "no native handler or isolated host invoker is registered"
            )
        outcome = await self.host_invoker.invoke(
            call,
            retryable=call.spec.idempotency is IdempotencyClass.IDEMPOTENT,
        )
        if not isinstance(outcome, PluginHostOutcome):
            raise _HostFailure(PluginHostErrorCode.WORKER_ERROR, False)
        if (
            outcome.request.call_id != call.call_id
            or outcome.response.call_id != call.call_id
        ):
            raise _HostFailure(PluginHostErrorCode.RESPONSE_MISMATCH, False)
        if outcome.response.status is PluginHostStatus.SUCCESS:
            return outcome.response.result
        assert outcome.response.error is not None
        raise _HostFailure(outcome.response.error.code, outcome.retryable)

    async def _invoke_native(
        self,
        handler: NativeToolHandler | CooperativeSyncHandler,
        call: ToolCall,
    ) -> Any:
        if isinstance(handler, CooperativeSyncHandler):
            return await self._invoke_cooperative_sync(handler, call)
        return await handler(call)

    async def _invoke_cooperative_sync(
        self,
        handler: CooperativeSyncHandler,
        call: ToolCall,
    ) -> Any:
        stop_event = threading.Event()
        loop = asyncio.get_running_loop()
        completion: asyncio.Future[Any] = loop.create_future()

        def complete(result: Any = None, error: BaseException | None = None) -> None:
            if completion.done():
                return
            if error is None:
                completion.set_result(result)
            else:
                completion.set_exception(error)

        def run() -> None:
            try:
                result = handler.callback(call, stop_event)
            except BaseException as error:
                loop.call_soon_threadsafe(complete, None, error)
            else:
                loop.call_soon_threadsafe(complete, result)

        worker = threading.Thread(
            target=run,
            name=f"openagent-tool-{call.call_id}",
            daemon=False,
        )
        worker.start()
        try:
            return await asyncio.shield(completion)
        except asyncio.CancelledError:
            stop_event.set()
            await self._await_sync_cleanup(completion)
            raise

    @staticmethod
    async def _await_sync_cleanup(completion: asyncio.Future[Any]) -> None:
        """Wait through repeated caller cancellation until the worker returns."""

        while not completion.done():
            try:
                await asyncio.shield(completion)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        with contextlib.suppress(BaseException):
            completion.result()

    async def _run_with_timeout(self, operation: Awaitable[Any], timeout: float) -> Any:
        task = asyncio.ensure_future(operation)
        self._live_tasks.add(task)
        task.add_done_callback(self._live_tasks.discard)
        try:
            return await asyncio.wait_for(task, timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            await self._cancel_and_reap(task)
            raise

    @staticmethod
    async def _cancel_and_reap(task: asyncio.Task[Any]) -> None:
        if not task.done():
            task.cancel()
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break

    def _host_failure(
        self,
        call: ToolCall,
        code: PluginHostErrorCode,
        retryable: bool,
    ) -> ToolResult:
        if code is PluginHostErrorCode.CANCELLED:
            return self._result(
                call,
                ToolResultStatus.CANCELLED,
                ToolErrorCode.CANCELLED,
                "isolated host execution was cancelled",
            )
        if code is PluginHostErrorCode.TIMED_OUT:
            return self._result(
                call,
                ToolResultStatus.TIMED_OUT,
                ToolErrorCode.TIMED_OUT,
                "isolated host execution timed out",
            )
        return self._result(
            call,
            ToolResultStatus.ERROR,
            ToolErrorCode.HOST_FAILED,
            "isolated host execution failed",
            retryable=retryable,
        )

    @staticmethod
    def _result(
        call: ToolCall,
        status: ToolResultStatus,
        code: ToolErrorCode,
        message: str,
        *,
        retryable: bool = False,
        field_path: tuple[str | int, ...] = (),
    ) -> ToolResult:
        return ToolResult(
            call.call_id,
            status,
            error=ToolError(
                code,
                message,
                canonical_id=call.canonical_id,
                requested_name=call.requested_name,
                field_path=field_path,
            ),
            retryable=retryable,
        )

    async def _record(
        self,
        pipeline: _TracePipeline,
        state: ToolTraceState,
        details: Mapping[str, Any],
    ) -> None:
        timestamp = self._now()
        previous = pipeline.trace
        pipeline.trace = ToolTrace(
            previous.call_id,
            previous.correlation_id,
            state,
            previous.created_at,
            timestamp,
            (*previous.events, ToolTraceEvent(state, timestamp, details)),
        )
        await self._emit(pipeline.trace)

    async def _record_terminal(
        self, pipeline: _TracePipeline, result: ToolResult
    ) -> None:
        state = {
            ToolResultStatus.SUCCESS: ToolTraceState.COMPLETED,
            ToolResultStatus.ERROR: ToolTraceState.FAILED,
            ToolResultStatus.CANCELLED: ToolTraceState.CANCELLED,
            ToolResultStatus.TIMED_OUT: ToolTraceState.TIMED_OUT,
        }[result.status]
        details = {"status": result.status.value}
        if result.error is not None:
            details["reason"] = result.error.code.value
        await self._record(pipeline, state, details)

    async def _emit(self, trace: ToolTrace) -> None:
        if self.trace_sink is None:
            return
        try:
            await self.trace_sink.emit(trace)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    def _validate_native_handlers(
        self,
        handlers: Mapping[str, NativeToolHandler | CooperativeSyncHandler],
    ) -> Mapping[str, NativeToolHandler | CooperativeSyncHandler]:
        indexed: dict[str, NativeToolHandler | CooperativeSyncHandler] = {}
        for canonical_id, handler in handlers.items():
            normalized = normalize_tool_name(canonical_id, canonical=True)
            if normalized != canonical_id:
                raise TypeError("native handlers require canonical callable keys")
            if not isinstance(
                handler, CooperativeSyncHandler
            ) and not _is_async_callable(handler):
                raise ToolExecutorError(
                    ToolErrorCode.INVALID_SPEC,
                    "native handlers must be async or CooperativeSyncHandler instances",
                )
            try:
                spec = self.registry.resolve(normalized)
            except ToolKernelError as error:
                raise ToolExecutorError(
                    ToolErrorCode.INVALID_CALL,
                    "native handler is not registered for a canonical tool ID",
                ) from error
            if spec.canonical_id != normalized:
                raise ToolExecutorError(
                    ToolErrorCode.INVALID_CALL,
                    "native handler keys cannot be aliases",
                )
            indexed[normalized] = handler
        return MappingProxyType(indexed)

    def _now(self) -> datetime:
        value = self.clock()
        if not isinstance(value, datetime):
            raise TypeError("executor clock must return a datetime")
        return value


@dataclass(frozen=True)
class _HostFailure(Exception):
    code: PluginHostErrorCode
    retryable: bool
