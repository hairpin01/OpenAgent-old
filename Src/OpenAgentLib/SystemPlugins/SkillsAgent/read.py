# SPDX-License-Identifier: MIT
from __future__ import annotations

from openagent_system_tool_api import SystemTool

SYSTEM_TOOL = SystemTool(
    tool_class="skills",
    name="read",
    handler="handle",
    docs={
        "desc": "Read an installed OpenAgent skill.",
        "args": "name",
        "body": "optional skill name",
    },
)


async def handle(
    agent,
    tool_name: str,
    attrs_raw: str = "",
    body: str = "",
    source_event=None,
) -> str:
    return await agent._skills_registry_tool(tool_name, attrs_raw, body)
