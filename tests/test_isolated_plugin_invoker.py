from __future__ import annotations

import asyncio

from conftest import ROOT

from OpenAgentLib.IsolatedPluginInvoker import IsolatedPluginInvoker
from OpenAgentLib.PluginDiscovery import inspect_v2_plugin_source
from OpenAgentLib.PluginHost import PluginHost
from OpenAgentLib.ToolKernel import ToolContext, ToolResultStatus
from OpenAgentLib.ToolPolicy import (
    ConfirmationState,
    ToolConfirmationGrant,
    ToolPolicyRequest,
)
from OpenAgentLib.V2Bootstrap import build_v2_tool_runtime


class _RuntimeApp:
    """No native service is exercised by this isolated-plugin routing test."""

    _v2_source_event = None


def test_eval_plugin_executes_only_through_isolated_host() -> None:
    plugin_root = ROOT.parent / "repo-MCUB-fork" / "OpenAgent"
    source = inspect_v2_plugin_source(plugin_root / "plugins" / "eval.py")
    invoker = IsolatedPluginInvoker(
        PluginHost(),
        {"eval": source},
        plugin_root=plugin_root,
        openagent_source=ROOT / "Src",
        capability_handler=lambda _call, _policy, request: {"ok": False},
    )
    runtime = build_v2_tool_runtime(_RuntimeApp(), host_invoker=invoker)
    call = runtime.registry.create_call(
        call_id="plugin-call",
        requested_name="eval.python",
        arguments={"code": "1 + 1"},
        context=ToolContext("isolated"),
    )

    result, _trace = asyncio.run(
        runtime.executor.execute(
            call,
            ToolPolicyRequest(
                enabled_tool_ids=frozenset({call.canonical_id}),
                granted_capabilities=call.spec.capabilities,
                confirmation=ConfirmationState.APPROVED,
                confirmation_grant=ToolConfirmationGrant.for_call("test", call),
            ),
        )
    )

    assert result.status is ToolResultStatus.SUCCESS
    assert result.output["ok"] is True
