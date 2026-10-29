from __future__ import annotations

from ..core import Tool, ToolContext
from ..workspace import read_text, resolve_path


async def handle(_ctx: ToolContext, args: dict) -> str:
    path, name = resolve_path(args.get("path", ""))
    lines = read_text(path, name).splitlines()

    start = max(int(args.get("start_line") or 1), 1)
    end = int(args.get("end_line") or len(lines))
    return "\n".join(lines[start - 1 : end])


TOOL = Tool(
    name="read_file",
    description="Read a text file from the workspace, optionally a range of lines.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path inside the workspace."},
            "start_line": {"type": "integer", "description": "First line, 1-based."},
            "end_line": {"type": "integer", "description": "Last line, inclusive."},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    handler=handle,
)
