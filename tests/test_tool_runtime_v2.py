from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from OpenAgentLib.ToolKernel import ToolContext, ToolResultStatus
from OpenAgentLib.ToolModelBoundary import ModelTurnKind
from OpenAgentLib.ToolPolicy import ToolPolicyRequest
from OpenAgentLib.ToolRuntimeV2 import ToolRuntimeV2


class RecordingServices:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def __getattr__(self, operation: str) -> Any:
        async def execute(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
            self.calls.append((operation, arguments))
            return {"result": operation}

        return execute


def _runtime(services: RecordingServices) -> ToolRuntimeV2:
    call_ids = iter(("native-call-1", "native-call-2"))
    return ToolRuntimeV2.from_native(
        services,
        context=ToolContext("correlation-1", "actor-1"),
        call_id_factory=lambda: next(call_ids),
    )


def _policy(runtime: ToolRuntimeV2) -> ToolPolicyRequest:
    return ToolPolicyRequest(
        enabled_tool_ids=frozenset(
            spec.canonical_id for spec in runtime.registry.specs()
        ),
        granted_capabilities=frozenset({"read-only"}),
    )


def test_native_factory_executes_and_renders_model_tool_call() -> None:
    async def scenario() -> None:
        services = RecordingServices()
        runtime = _runtime(services)

        execution = await runtime.execute_model_output(
            '{"name":"utility.token_usage","arguments":{}}',
            _policy(runtime),
        )

        assert execution.boundary_output.kind is ModelTurnKind.TOOLS
        assert execution.boundary_output.calls[0].call_id == "native-call-1"
        assert execution.results[0].status is ToolResultStatus.SUCCESS
        assert execution.traces[0].call_id == "native-call-1"
        assert json.loads(execution.rendered_results[0]) == {
            "call_id": "native-call-1",
            "output": {"result": "utility_token_usage"},
            "status": "success",
        }
        assert services.calls == [("utility_token_usage", {})]

    asyncio.run(scenario())


def test_non_tool_turn_returns_boundary_output_without_execution() -> None:
    async def scenario() -> None:
        services = RecordingServices()
        execution = await _runtime(services).execute_model_output(
            "```final\nfinal answer\n```"
        )

        assert execution.boundary_output.kind is ModelTurnKind.FINAL
        assert execution.boundary_output.final_answer == "final answer"
        assert execution.results == ()
        assert execution.traces == ()
        assert execution.rendered_results == ()
        assert services.calls == []

    asyncio.run(scenario())


def test_tool_turn_requires_explicit_policy_environment() -> None:
    async def scenario() -> None:
        runtime = _runtime(RecordingServices())
        try:
            await runtime.execute_model_output(
                '{"name":"utility.token_usage","arguments":{}}'
            )
        except ValueError as error:
            assert str(error) == "policy_request is required for tool calls"
        else:
            raise AssertionError("tool calls must fail closed without a policy request")

    asyncio.run(scenario())
