from __future__ import annotations

import asyncio
from pathlib import Path

from conftest import ROOT

from OpenAgentLib.IsolatedPluginInvoker import IsolatedPluginInvoker
from OpenAgentLib.PluginDiscovery import inspect_v2_plugin_source
from OpenAgentLib.PluginHost import PluginHost
from OpenAgentLib.ToolKernel import ToolContext
from OpenAgentLib.V2Bootstrap import build_v2_tool_runtime


class _Services:
    def __getattr__(self, name: str):
        async def result(_arguments):
            return {"result": name}

        return result


def test_eval_plugin_executes_only_through_isolated_host() -> None:
    plugin_root = ROOT.parent / "repo-MCUB-fork" / "OpenAgent"
    source = inspect_v2_plugin_source(plugin_root / "plugins" / "eval.py")
    invoker = IsolatedPluginInvoker(
        PluginHost(),
        {"eval": source},
        plugin_root=plugin_root,
        openagent_source=ROOT / "Src",
        capability_handler=lambda request: {"ok": False},
    )
    runtime = build_v2_tool_runtime(_Services(), host_invoker=invoker)
    call = runtime.registry.create_call(
        call_id="plugin-call",
        requested_name="eval.python",
        arguments={"code": "1 + 1"},
        context=ToolContext("isolated"),
    )

    outcome = asyncio.run(invoker.invoke(call, retryable=True))

    assert outcome.response.error is None
    assert outcome.response.result["ok"] is True
