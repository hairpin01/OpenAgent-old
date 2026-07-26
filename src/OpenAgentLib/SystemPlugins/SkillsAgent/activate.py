# SPDX-License-Identifier: MIT
from __future__ import annotations

from openagent_system_tool_api import SystemTool

SYSTEM_TOOL = SystemTool(
    tool_class='skills',
    name='activate',
    handler='handle',
    docs={'desc': 'Activate/load the best matching installed skill for the current task.', 'args': 'query/name', 'body': 'optional query'},
)


async def handle(
    agent,
    tool_name: str,
    attrs_raw: str = "",
    body: str = "",
    source_event=None,
) -> str:
    return await agent._skills_registry_tool(tool_name, attrs_raw, body)
