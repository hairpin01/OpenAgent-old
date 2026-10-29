from __future__ import annotations

from datetime import datetime, timezone

from ..core import Tool, ToolContext


async def handle(_ctx: ToolContext, _args: dict) -> str:
    return datetime.now(timezone.utc).isoformat()


TOOL = Tool(
    name="time_now",
    description="Return the current UTC date and time in ISO 8601 format.",
    parameters={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    handler=handle,
)
