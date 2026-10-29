from __future__ import annotations

from ..core import Tool, ToolContext
from ..workspace import make_diff, resolve_path


async def handle(ctx: ToolContext, args: dict) -> str:
    changes = ctx.state.get("changes") or {}
    if args.get("path"):
        _, name = resolve_path(args["path"])
        if name not in changes:
            raise ValueError(f"no changes recorded for: {name}")
        changes = {name: changes[name]}

    diffs = [make_diff(name, item["old"], item["new"]) for name, item in changes.items()]
    return "\n\n".join(diff for diff in diffs if diff) or "no changes"


TOOL = Tool(
    name="diff",
    description=(
        "Show the unified diff of files changed by the edit and write tools "
        "in this request. Without a path, shows all changed files."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Limit the diff to one file."},
        },
        "additionalProperties": False,
    },
    handler=handle,
)
