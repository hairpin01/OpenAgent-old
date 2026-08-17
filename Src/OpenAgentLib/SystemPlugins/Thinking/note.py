# SPDX-License-Identifier: MIT
from __future__ import annotations

from openagent_system_tool_api import SystemTool

SYSTEM_TOOL = SystemTool(
    tool_class="thinking",
    name="note",
    handler="handle",
    docs={
        "desc": "Record a concise progress/thinking note for the user.",
        "args": "note/text",
        "body": "optional note text",
    },
)


async def handle(
    agent,
    tool_name: str,
    attrs_raw: str = "",
    body: str = "",
    source_event=None,
) -> str:
    return await agent._thinking_note_tool(attrs_raw, body)
