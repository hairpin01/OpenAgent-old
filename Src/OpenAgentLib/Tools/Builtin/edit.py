from __future__ import annotations

import json

from ..core import Tool, ToolContext
from ..workspace import read_text, record_change, resolve_path, write_text


async def handle(ctx: ToolContext, args: dict) -> str:
    path, name = resolve_path(args.get("path", ""))
    old_strings = args.get("old_strings")
    new_strings = args.get("new_strings")
    if not isinstance(old_strings, list) or not isinstance(new_strings, list):
        raise ValueError("old_strings and new_strings must be arrays")
    if len(old_strings) != len(new_strings):
        raise ValueError("old_strings and new_strings must have the same length")

    old = read_text(path, name)
    text = old
    for index, (before, after) in enumerate(zip(old_strings, new_strings)):
        before, after = str(before), str(after)
        if not before:
            raise ValueError(f"old_strings[{index}] is empty")
        count = text.count(before)
        if count != 1:
            raise ValueError(
                f"old_strings[{index}] must match exactly once, found {count}"
            )
        text = text.replace(before, after, 1)

    # Nothing is written unless every replacement succeeded
    write_text(path, name, text)
    record_change(ctx, name, old, text)
    return json.dumps(
        {"ok": True, "path": name, "replaced": len(old_strings)}, ensure_ascii=False
    )


TOOL = Tool(
    name="edit",
    description=(
        "Edit a file in place: replace each old string with the new string "
        "at the same index. Every old string must occur exactly once. "
        "Replacements are applied in order and either all succeed or none."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path (relative to the bot working dir or absolute)."},
            "old_strings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exact fragments to find.",
            },
            "new_strings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Replacements, same order and length as old_strings.",
            },
        },
        "required": ["path", "old_strings", "new_strings"],
        "additionalProperties": False,
    },
    handler=handle,
)
