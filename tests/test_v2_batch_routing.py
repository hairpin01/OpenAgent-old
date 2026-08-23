from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from OpenAgentLib.Plugin.PluginsEngine import (
    _OpenAgentAgentLoopMixin,
    _OpenAgentStatusMixin,
)
from OpenAgentLib.ToolKernel import (
    ToolError,
    ToolErrorCode,
    ToolResult,
    ToolResultStatus,
)
from OpenAgentLib.ToolPolicy import ConfirmationState
from OpenAgentLib.V2Bootstrap import build_v2_tool_runtime
from tool_testkit import build_tool_call, build_tool_spec
from OpenAgentLib.ToolKernel import ConfirmationRequirement


@dataclass(frozen=True)
class _Event:
    sender_id: int = 91
    chat_id: int = 17


class _RuntimeApp:
    _v2_source_event = None


class _RecordingExecutor:
    def __init__(self) -> None:
        self.batches: list[tuple[tuple[object, ...], tuple[object, ...]]] = []
        self.executed_calls: list[object] = []
        self.statuses: tuple[ToolResultStatus, ...] = ()
        self.dispatch_error: Exception | None = None
        self.result_limit: int | None = None

    async def execute_batch(
        self, calls: tuple[object, ...], requests: tuple[object, ...]
    ):
        self.batches.append((calls, requests))
        self.executed_calls.extend(
            call
            for call, request in zip(calls, requests, strict=True)
            if request.confirmation is ConfirmationState.APPROVED
        )
        if self.dispatch_error is not None:
            raise self.dispatch_error
        results = []
        for index, call in enumerate(calls):
            status = (
                self.statuses[index]
                if index < len(self.statuses)
                else ToolResultStatus.SUCCESS
            )
            error = None
            output = {"result": call.canonical_id}
            if status is not ToolResultStatus.SUCCESS:
                output = None
                error = ToolError(ToolErrorCode.HANDLER_FAILED, "Tool execution failed")
            results.append(ToolResult(call.call_id, status, output, error))
        if self.result_limit is not None:
            results = results[: self.result_limit]
        return tuple(results), ()


class _StatusEvent:
    _openagent_status_buttons: tuple[()] = ()

    def __init__(self) -> None:
        self.edits: list[str] = []

    async def edit(self, text: str, **_kwargs: object) -> None:
        self.edits.append(text)


class _Logger:
    def __init__(self) -> None:
        self.debug_calls: list[tuple[object, ...]] = []

    def debug(self, *args: object) -> None:
        self.debug_calls.append(args)


class _BatchHarness(_OpenAgentAgentLoopMixin, _OpenAgentStatusMixin):
    def __init__(
        self, *, confirmation_enabled: object = True, mode: str = "medium"
    ) -> None:
        runtime = build_v2_tool_runtime(_RuntimeApp())
        self.executor = _RecordingExecutor()
        self._v2_runtime = SimpleNamespace(
            registry=runtime.registry,
            executor=self.executor,
        )
        self._v2_source_event = None
        self._cancelled_generations: set[str] = set()
        self.config = {
            "tool_confirmation_enabled": confirmation_enabled,
            "tool_confirmation_mode": mode,
        }
        self.log = _Logger()
        self.confirm_calls = 0
        self.confirmed = True

    @staticmethod
    def _parse_xml_attrs(attrs_raw: str) -> dict[str, str]:
        if attrs_raw == "malformed":
            raise ValueError("malformed attributes")
        return {}

    @staticmethod
    def _render_tool_display(*, title: str, value: str, **_kwargs: object) -> str:
        return f"{title}|{value}"

    @staticmethod
    def _event_chat_id(event: _Event) -> int:
        return event.chat_id

    @staticmethod
    def _tool_group(name: str) -> str:
        return name.partition(".")[0]

    @staticmethod
    def _get_system_tool(_name: str):
        return None

    async def _confirm_dangerous_tool(
        self, _event: object, _tool: str, _value: str, *, elapsed: float | None
    ) -> bool:
        del elapsed
        self.confirm_calls += 1
        return self.confirmed


def test_normal_model_batch_routes_once_through_executor_and_preserves_order() -> None:
    async def scenario() -> None:
        harness = _BatchHarness()
        outputs = await harness._dispatch_agent_tool_batch(
            [
                ("utility.list_tools", "", ""),
                ("utility.token_usage", "", ""),
            ],
            source_event=_Event(),
            status_event=None,
            agent_log=[],
            started_at=None,
            thinking_notes=[],
            cancel_token=None,
        )

        assert len(harness.executor.batches) == 1
        calls, requests = harness.executor.batches[0]
        assert [call.canonical_id for call in calls] == [
            "utility.list_tools",
            "utility.token_usage",
        ]
        assert len(requests) == 2
        assert [output for output in outputs if "success" in output] == outputs
        assert harness._v2_source_event is None

    asyncio.run(scenario())


def test_success_status_uses_one_stable_invocation_id() -> None:
    async def scenario() -> None:
        harness = _BatchHarness()
        status = _StatusEvent()

        await harness._dispatch_agent_tool_batch(
            [("utility.list_tools", "", "")],
            source_event=_Event(),
            status_event=status,
            agent_log=[],
            started_at=None,
            thinking_notes=[],
            cancel_token=None,
        )

        assert len(status.edits) == 2
        assert status.edits[0].startswith("Started utility.list_tools|")
        assert status.edits[1].startswith("Succeeded utility.list_tools|")
        invocation_ids = [
            edit.split("invocation=", 1)[1].split()[0] for edit in status.edits
        ]
        assert invocation_ids[0] == invocation_ids[1]
        assert len(invocation_ids[0]) == 32

    asyncio.run(scenario())


def test_failure_status_uses_one_stable_invocation_id() -> None:
    async def scenario() -> None:
        harness = _BatchHarness()
        harness.executor.statuses = (ToolResultStatus.ERROR,)
        status = _StatusEvent()

        await harness._dispatch_agent_tool_batch(
            [("utility.list_tools", "", "")],
            source_event=_Event(),
            status_event=status,
            agent_log=[],
            started_at=None,
            thinking_notes=[],
            cancel_token=None,
        )

        assert [edit.split(" ", 1)[0] for edit in status.edits] == [
            "Started",
            "Failed",
        ]
        assert (
            status.edits[0].split("invocation=", 1)[1].split()[0]
            == status.edits[1].split("invocation=", 1)[1].split()[0]
        )

    asyncio.run(scenario())


def test_cancelled_status_uses_one_stable_invocation_id() -> None:
    async def scenario() -> None:
        harness = _BatchHarness()
        harness.executor.statuses = (ToolResultStatus.CANCELLED,)
        status = _StatusEvent()

        await harness._dispatch_agent_tool_batch(
            [("utility.list_tools", "", "")],
            source_event=_Event(),
            status_event=status,
            agent_log=[],
            started_at=None,
            thinking_notes=[],
            cancel_token=None,
        )

        assert [edit.split(" ", 1)[0] for edit in status.edits] == [
            "Started",
            "Cancelled",
        ]
        assert (
            status.edits[0].split("invocation=", 1)[1].split()[0]
            == status.edits[1].split("invocation=", 1)[1].split()[0]
        )

    asyncio.run(scenario())


def test_malformed_call_gets_started_and_failed_status_without_execution() -> None:
    async def scenario() -> None:
        harness = _BatchHarness()
        status = _StatusEvent()

        outputs = await harness._dispatch_agent_tool_batch(
            [("utility.list_tools", "malformed", "")],
            source_event=_Event(),
            status_event=status,
            agent_log=[],
            started_at=None,
            thinking_notes=[],
            cancel_token=None,
        )

        assert [edit.split(" ", 1)[0] for edit in status.edits] == [
            "Started",
            "Failed",
        ]
        assert harness.executor.executed_calls == []
        assert '"call_id"' in outputs[0]

    asyncio.run(scenario())


def test_multi_tool_status_order_preserves_call_and_terminal_order() -> None:
    async def scenario() -> None:
        harness = _BatchHarness()
        harness.executor.statuses = (
            ToolResultStatus.SUCCESS,
            ToolResultStatus.ERROR,
        )
        status = _StatusEvent()

        await harness._dispatch_agent_tool_batch(
            [
                ("utility.list_tools", "", ""),
                ("utility.token_usage", "", ""),
            ],
            source_event=_Event(),
            status_event=status,
            agent_log=[],
            started_at=None,
            thinking_notes=[],
            cancel_token=None,
        )

        assert [edit.split("|", 1)[0] for edit in status.edits] == [
            "Started utility.list_tools",
            "Started utility.token_usage",
            "Succeeded utility.list_tools",
            "Failed utility.token_usage",
        ]
        ids = [edit.split("invocation=", 1)[1].split()[0] for edit in status.edits]
        assert ids[0] == ids[2]
        assert ids[1] == ids[3]
        assert ids[0] != ids[1]

    asyncio.run(scenario())


@pytest.mark.parametrize("result_limit", [None, 1])
def test_dispatch_errors_fail_every_started_invocation(
    result_limit: int | None,
) -> None:
    async def scenario() -> None:
        harness = _BatchHarness()
        if result_limit is None:
            harness.executor.dispatch_error = RuntimeError("dispatch failed")
        else:
            harness.executor.result_limit = result_limit
        status = _StatusEvent()

        with pytest.raises(RuntimeError):
            await harness._dispatch_agent_tool_batch(
                [
                    ("utility.list_tools", "", ""),
                    ("utility.token_usage", "", ""),
                ],
                source_event=_Event(),
                status_event=status,
                agent_log=[],
                started_at=None,
                thinking_notes=[],
                cancel_token=None,
            )

        assert [edit.split("|", 1)[0] for edit in status.edits] == [
            "Started utility.list_tools",
            "Started utility.token_usage",
            "Failed utility.list_tools",
            "Failed utility.token_usage",
        ]

    asyncio.run(scenario())


def test_confirmation_grant_is_bound_to_the_exact_mutating_call() -> None:
    async def scenario() -> None:
        harness = _BatchHarness()
        await harness._dispatch_agent_tool_batch(
            [("todo.add", "", "native task")],
            source_event=_Event(),
            status_event=object(),
            agent_log=[],
            started_at=None,
            thinking_notes=[],
            cancel_token=None,
        )

        calls, requests = harness.executor.batches[0]
        grant = requests[0].confirmation_grant
        assert grant is not None
        assert grant.call_id == calls[0].call_id
        assert grant.canonical_id == calls[0].canonical_id

    asyncio.run(scenario())


def test_disabled_or_off_confirmation_auto_approves_required_v2_call() -> None:
    async def scenario() -> None:
        spec = build_tool_spec(
            "terminal.run", confirmation=ConfirmationRequirement.REQUIRED
        )
        call = build_tool_call(spec, call_id="terminal-call")
        for enabled, mode in ((False, "medium"), ("off", "medium"), (True, "off")):
            harness = _BatchHarness(confirmation_enabled=enabled, mode=mode)
            request = await harness._v2_policy_request(
                call, status_event=None, started_at=None
            )

            assert request.confirmation is ConfirmationState.APPROVED
            assert request.confirmation_grant is not None
            assert request.confirmation_grant.call_id == call.call_id
            assert request.confirmation_grant.canonical_id == "terminal.run"
            assert request.confirmation_grant.token.startswith("config-auto-")
            assert harness.confirm_calls == 0
            assert harness.log.debug_calls

    asyncio.run(scenario())


def test_disabled_confirmation_executes_terminal_with_exact_auto_grant() -> None:
    async def scenario() -> None:
        harness = _BatchHarness(confirmation_enabled=False)
        await harness._dispatch_agent_tool_batch(
            [("terminal.run", "", "")],
            source_event=_Event(),
            status_event=None,
            agent_log=[],
            started_at=None,
            thinking_notes=[],
            cancel_token=None,
        )

        calls, requests = harness.executor.batches[0]
        grant = requests[0].confirmation_grant
        assert [call.canonical_id for call in harness.executor.executed_calls] == [
            "terminal.run"
        ]
        assert harness.confirm_calls == 0
        assert grant is not None
        assert grant.call_id == calls[0].call_id
        assert grant.canonical_id == calls[0].canonical_id

    asyncio.run(scenario())


def test_required_confirmation_prompts_once_and_fails_closed_without_ui() -> None:
    async def scenario() -> None:
        spec = build_tool_spec(
            "terminal.run", confirmation=ConfirmationRequirement.REQUIRED
        )
        call = build_tool_call(spec, call_id="terminal-call")
        harness = _BatchHarness(mode="medium")

        missing = await harness._v2_policy_request(
            call, status_event=None, started_at=None
        )
        assert missing.confirmation is ConfirmationState.MISSING
        assert missing.confirmation_grant is None
        assert harness.confirm_calls == 0

        harness.confirmed = False
        rejected = await harness._v2_policy_request(
            call, status_event=object(), started_at=None
        )
        assert rejected.confirmation is ConfirmationState.REJECTED
        assert rejected.confirmation_grant is None
        assert harness.confirm_calls == 1

        await harness._dispatch_agent_tool_batch(
            [("terminal.run", "", "")],
            source_event=_Event(),
            status_event=object(),
            agent_log=[],
            started_at=None,
            thinking_notes=[],
            cancel_token=None,
        )
        assert harness.executor.executed_calls == []

    asyncio.run(scenario())
