# SPDX-License-Identifier: MIT
from __future__ import annotations

from openagent_system_tool_api import SystemTool

SYSTEM_TOOL = SystemTool(
    tool_class='code',
    name='choose_filename',
    handler='handle',
    docs={'desc': 'Choose/sanitize a filename for generated code.', 'args': 'name/path', 'body': 'optional filename'},
)


async def handle(
    agent,
    tool_name: str,
    attrs_raw: str = "",
    body: str = "",
    source_event=None,
) -> str:
    return await agent._code_registry_tool(tool_name, attrs_raw, body, source_event=source_event)
