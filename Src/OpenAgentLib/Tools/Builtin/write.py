from __future__ import annotations

import json

from ..checks import LANG_PARAM, apply_result, verify
from ..core import Tool, ToolContext
from ..workspace import read_text, record_change, resolve_path, write_text


async def handle(ctx: ToolContext, args: dict) -> str:
    path, name = resolve_path(args.get("path", ""))
    content = str(args.get("content", ""))
    old = read_text(path, name) if path.is_file() else None

    payload = {"ok": True, "path": name, "created": old is None}
    apply_result(verify(args.get("lang"), name, old, content), payload)
    write_text(path, name, content)
    record_change(ctx, name, old, content)
    return json.dumps(payload, ensure_ascii=False)


TOOL = Tool(
    name="write",
    description=(
        "Create a file or overwrite it with the given content. The content is "
        "syntax-checked by lang before saving (python, json, yaml, toml) and "
        "rejected if it does not parse."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path (relative to the bot working dir or absolute)."},
            "content": {"type": "string", "description": "Full new file content."},
            "lang": LANG_PARAM,
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
    handler=handle,
)
