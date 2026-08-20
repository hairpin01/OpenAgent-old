from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from OpenAgentLib.SystemPlugins.native import NativeSystemToolServices
from OpenAgentLib.ToolCompatibility import TOOL_COMPATIBILITY_MATRIX
from OpenAgentLib.ToolKernel import ToolContext, ToolResultStatus
from OpenAgentLib.ToolPolicy import ConfirmationState, ToolPolicyRequest
from OpenAgentLib.V2Bootstrap import build_v2_tool_runtime


class _Services(NativeSystemToolServices):
    def __getattr__(self, name: str) -> Any:
        async def result(_arguments: Mapping[str, Any]) -> Mapping[str, Any]:
            return {"result": name}

        return result

    async def utility_list_tools(
        self, _arguments: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return {"result": "utility.list_tools"}


def test_bootstrap_builds_native_and_sibling_registry_and_executes_model_call() -> None:
    runtime = build_v2_tool_runtime(_Services())
    expected = sum(
        entry.migration_disposition == "migrate" for entry in TOOL_COMPATIBILITY_MATRIX
    )
    assert len(runtime.registry.specs()) == expected
    assert runtime.registry.resolve("terminal").canonical_id == "terminal.run"

    async def scenario() -> None:
        output, results, traces = await runtime.execute_model_output(
            '```tool_call\n{"tool":"utility.list_tools","args":{}}\n```',
            context=ToolContext("test"),
            request_for=lambda call: ToolPolicyRequest(
                enabled_tool_ids=frozenset({call.canonical_id}),
                granted_capabilities=call.spec.capabilities,
                confirmation=ConfirmationState.APPROVED,
                remaining_calls=1,
            ),
        )
        assert output.calls[0].canonical_id == "utility.list_tools"
        assert results[0].status is ToolResultStatus.SUCCESS
        assert traces[0].call_id == results[0].call_id

    asyncio.run(scenario())
