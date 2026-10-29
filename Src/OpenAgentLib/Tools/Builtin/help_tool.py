from __future__ import annotations

import json

from ..core import Tool, ToolContext


async def handle(ctx: ToolContext, args: dict) -> str:
    tools = ctx.tools or []
    name = str(args.get("name") or "").strip()

    if not name:
        return json.dumps(
            {
                "ok": True,
                "tools": [
                    {"name": item.name, "description": item.description}
                    for item in tools
                ],
            },
            ensure_ascii=False,
        )

    tool = next((item for item in tools if item.name == name), None)
    if tool is None:
        return json.dumps(
            {
                "ok": False,
                "error": f"unknown tool: {name}",
                "available": [item.name for item in tools],
            },
            ensure_ascii=False,
        )

    properties = tool.parameters.get("properties") or {}
    required = set(tool.parameters.get("required") or [])
    arguments = {}
    for key, spec in properties.items():
        spec = spec if isinstance(spec, dict) else {}
        arguments[key] = {
            "type": spec.get("type"),
            "required": key in required,
            "description": spec.get("description", ""),
        }

    return json.dumps(
        {
            "ok": True,
            "name": tool.name,
            "description": tool.description,
            "schema": tool.parameters,
            "arguments": arguments,
        },
        ensure_ascii=False,
    )


TOOL = Tool(
    name="help_tool",
    description=(
        "Show the schema and argument descriptions of a tool. "
        "Call without arguments to list all available tools."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Tool name. Omit to list all tools.",
            },
        },
        "additionalProperties": False,
    },
    handler=handle,
)
