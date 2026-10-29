"""Copy this file to ``echo.py`` to enable a custom tool."""

from __future__ import annotations

from ..core import Tool, ToolContext


async def handle(_ctx: ToolContext, args: dict) -> str:
    return str(args.get("text", ""))


TOOL = Tool(
    name="echo",
    description="Return the provided text unchanged.",
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string"},
        },
        "required": ["text"],
        "additionalProperties": False,
    },
    handler=handle,
)
