# SPDX-License-Identifier: MIT
from __future__ import annotations

from openagent_system_tool_api import SystemTool

SYSTEM_TOOL = SystemTool(
    tool_class="utility",
    name="list_tools",
    handler="handle",
    parallel_safe=True,
    docs={
        "desc": "List all available core and plugin tools by category with short descriptions."
    },
)


async def handle(
    agent,
    tool_name: str,
    attrs_raw: str = "",
    body: str = "",
    source_event=None,
) -> str:
    return await agent._utility_registry_tool(tool_name, attrs_raw, body)
