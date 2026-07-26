# SPDX-License-Identifier: MIT
from __future__ import annotations

from openagent_system_tool_api import SystemTool

SYSTEM_TOOL = SystemTool(
    tool_class='utility',
    name='placeholders',
    handler='handle',
    docs={'desc': 'Show available OpenAgent template placeholders.'},
)


async def handle(
    agent,
    tool_name: str,
    attrs_raw: str = "",
    body: str = "",
    source_event=None,
) -> str:
    return await agent._utility_registry_tool(tool_name, attrs_raw, body)
