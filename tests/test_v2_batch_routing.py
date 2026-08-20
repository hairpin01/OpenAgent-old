from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

from OpenAgentLib.Plugin.PluginsEngine import _OpenAgentAgentLoopMixin
from OpenAgentLib.ToolKernel import ToolResult, ToolResultStatus
from OpenAgentLib.V2Bootstrap import build_v2_tool_runtime


@dataclass(frozen=True)
class _Event:
    sender_id: int = 91
    chat_id: int = 17


class _RuntimeApp:
    _v2_source_event = None


class _RecordingExecutor:
    def __init__(self) -> None:
        self.batches: list[tuple[tuple[object, ...], tuple[object, ...]]] = []

    async def execute_batch(
        self, calls: tuple[object, ...], requests: tuple[object, ...]
    ):
        self.batches.append((calls, requests))
        return (
            tuple(
                ToolResult(
                    call.call_id,
                    ToolResultStatus.SUCCESS,
                    {"result": call.canonical_id},
                )
                for call in calls
            ),
            (),
        )


class _BatchHarness(_OpenAgentAgentLoopMixin):
    def __init__(self) -> None:
        runtime = build_v2_tool_runtime(_RuntimeApp())
        self.executor = _RecordingExecutor()
        self._v2_runtime = SimpleNamespace(
            registry=runtime.registry,
            executor=self.executor,
        )
        self._v2_source_event = None
        self._cancelled_generations: set[str] = set()

    @staticmethod
    def _parse_xml_attrs(_attrs_raw: str) -> dict[str, str]:
        return {}

    @staticmethod
    def _event_chat_id(event: _Event) -> int:
        return event.chat_id

    async def _confirm_dangerous_tool(
        self, _event: object, _tool: str, _value: str, *, elapsed: float | None
    ) -> bool:
        del elapsed
        return True


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
