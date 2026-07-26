# SPDX-License-Identifier: MIT
from __future__ import annotations

from openagent_system_tool_api import SystemTool

SYSTEM_TOOL = SystemTool(
    tool_class='context',
    name='discard',
    handler='handle',
    docs={'desc': 'Alias for context.prune.', 'args': 'target/all; keep', 'body': 'optional target list'},
)


async def handle(
    agent,
    tool_name: str,
    attrs_raw: str = "",
    body: str = "",
    source_event=None,
) -> str:
    return await agent._context_registry_tool(tool_name, attrs_raw, body, source_event)
