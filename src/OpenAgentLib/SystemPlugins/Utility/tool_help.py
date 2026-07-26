# SPDX-License-Identifier: MIT
from __future__ import annotations

from openagent_system_tool_api import SystemTool

SYSTEM_TOOL = SystemTool(
    tool_class='utility',
    name='tool_help',
    handler='handle',
    docs={'desc': 'Show normalized documentation for one core/plugin tool.', 'args': 'tool (str) — exact tool name', 'body': 'optional tool name'},
)


async def handle(
    agent,
    tool_name: str,
    attrs_raw: str = "",
    body: str = "",
    source_event=None,
) -> str:
    return await agent._utility_registry_tool(tool_name, attrs_raw, body)
