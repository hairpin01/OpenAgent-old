"""Long-term memory write: append one note that ``recall`` can find later."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from ..core import AgentPlugin, Tool, ToolContext
from .recall import MAX_TEXT_CHARS, MEMORY_PATH, load_entries

MAX_TAGS = 8
MAX_TAG_CHARS = 32


async def handle(_ctx: ToolContext, args: dict) -> str:
    # One flat line: recall prints a note on one line
    text = " ".join(str(args.get("text") or "").split())
    if not text:
        raise ValueError("text is empty")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"note is too long ({len(text)} > {MAX_TEXT_CHARS}); shorten it")

    tags: list[str] = []
    for raw in args.get("tags") or []:
        tag = " ".join(str(raw).split()).lower()[:MAX_TAG_CHARS]
        if tag and tag not in tags:
            tags.append(tag)
    tags = tags[:MAX_TAGS]

    entries = load_entries()
    if any(" ".join(str(e.get("text", "")).split()).lower() == text.lower() for e in entries):
        return json.dumps(
            {"ok": True, "saved": False, "reason": "same note already exists"},
            ensure_ascii=False,
        )

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "text": text,
        "tags": tags,
    }
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    # A file that lost its final newline would glue two notes into one line
    glue = ""
    if MEMORY_PATH.is_file() and MEMORY_PATH.stat().st_size:
        with MEMORY_PATH.open("rb") as fh:
            fh.seek(-1, 2)
            glue = "" if fh.read(1) == b"\n" else "\n"
    with MEMORY_PATH.open("a", encoding="utf-8") as fh:
        fh.write(glue + json.dumps(entry, ensure_ascii=False) + "\n")

    return json.dumps(
        {"ok": True, "saved": True, "entries": len(entries) + 1}, ensure_ascii=False
    )


PLUGIN = AgentPlugin(
    name="remember",
    description=(
        "Save ONE short note to long-term memory (found later by recall). "
        "Call it only after the user agreed to save it. A note says what "
        "the project/topic is, where the relevant files are, and decisions "
        "or conventions that mattered; no secrets, no long logs."
    ),
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": f"The note, up to {MAX_TEXT_CHARS} chars.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Lowercase keywords, e.g. the repo name. Up to 8.",
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    },
    handler=handle,
)

TOOL = PLUGIN
