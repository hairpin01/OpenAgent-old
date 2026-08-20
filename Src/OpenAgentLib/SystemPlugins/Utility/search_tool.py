# SPDX-License-Identifier: MIT
from __future__ import annotations

from openagent_system_tool_api import SystemTool

SYSTEM_TOOL = SystemTool(
    tool_class="utility",
    name="search_tool",
    handler="handle",
    parallel_safe=True,
    docs={
        "desc": "Search core and plugin tools by name, description, arguments, body usage, and normalized docs.",
        "args": "query (str) — capability or tool name to search for",
        "body": "optional natural-language search query",
        "returns": "ranked matching tool names with their documentation",
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
