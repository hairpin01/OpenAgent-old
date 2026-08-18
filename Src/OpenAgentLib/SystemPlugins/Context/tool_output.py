# SPDX-License-Identifier: MIT
from __future__ import annotations

from openagent_system_tool_api import SystemTool

SYSTEM_TOOL = SystemTool(
    tool_class="context",
    name="tool_output",
    handler="handle",
    parallel_safe=True,
    aliases=("context.read_tool_output", "tool_output.read"),
    docs={
        "desc": "Read full tool outputs saved by OpenAgent when a tool trace was too large for inline context.",
        "args": "path/file/id; latest=true; mode=head|tail|all; limit; offset",
        "body": "optional saved output path or filename from the tool trace",
    },
)


async def handle(
    agent,
    attrs: dict[str, str] | None = None,
    body: str = "",
    source_event=None,
) -> str:
    return await agent._read_tool_output_registry_tool(attrs or {}, body, source_event)
