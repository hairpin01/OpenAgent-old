# SPDX-License-Identifier: MIT
from __future__ import annotations

from openagent_system_tool_api import SystemTool

SYSTEM_TOOL = SystemTool(
    tool_class="skills",
    name="repo_list",
    handler="handle",
    parallel_safe=True,
    docs={"desc": "List skills available in the configured skill repository."},
)


async def handle(
    agent,
    tool_name: str,
    attrs_raw: str = "",
    body: str = "",
    source_event=None,
) -> str:
    return await agent._skills_registry_tool(tool_name, attrs_raw, body)
