"""Long-term memory search: keyword/tag recall over stored notes."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from ..core import Tool, ToolContext

if TYPE_CHECKING:
    pass

MEMORY_PATH = Path(os.path.expanduser("~/.openagent/memory.jsonl"))
MAX_ENTRIES = 100
MAX_TEXT_CHARS = 2000


def load_entries() -> list[dict]:
    if not MEMORY_PATH.is_file():
        return []
    entries: list[dict] = []
    for line in MEMORY_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            entries.append(item)
    return entries


def _fmt(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return ts


async def handle(_ctx: ToolContext, args: dict) -> str:
    query = str(args.get("query", "")).strip().lower()
    tags = [str(t).strip().lower() for t in (args.get("tags") or []) if str(t).strip()]
    limit = int(args.get("limit") or 10)
    limit = max(1, min(limit, MAX_ENTRIES))

    if not args.get("all") and not query and not tags:
        raise ValueError("provide query, tags, or use all=true")

    entries = load_entries()
    if not entries:
        return "Memory is empty. Notes are stored with the remember tool."

    if not args.get("all"):
        keywords = [k for k in query.split() if k]
        kept: list[dict] = []
        for entry in entries:
            text = str(entry.get("text", "")).lower()
            entry_tags = [str(t).lower() for t in entry.get("tags") or []]
            if keywords and not all(k in text or k in " ".join(entry_tags) for k in keywords):
                continue
            if tags and not set(tags) <= set(entry_tags):
                continue
            kept.append(entry)
        entries = kept
    else:
        entries = entries[-limit:]

    entries.sort(key=lambda e: str(e.get("ts", "")), reverse=True)
    if not entries:
        return f"No memory entries match (query={query!r}, tags={tags})."

    shown = entries[:limit]
    lines = [f"Found {len(entries)} entries, showing {len(shown)}:"]
    for entry in shown:
        text = str(entry.get("text", "")).strip()[:MAX_TEXT_CHARS]
        entry_tags = entry.get("tags") or []
        tag_part = f" [{', '.join(str(t) for t in entry_tags)}]" if entry_tags else ""
        lines.append(f"- {_fmt(str(entry.get('ts', '?')))}{tag_part}: {text}")
    return "\n".join(lines)


TOOL = Tool(
    name="recall",
    description=(
        "Search the agent's long-term memory (JSONL notes stored at "
        "~/.openagent/memory.jsonl). Each entry has: ts (ISO timestamp), "
        "text, tags (list). All query keywords must match (case-insensitive, "
        "text or tags). New notes are stored with the remember tool. Use "
        "all=true to list the most recent entries."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Space-separated keywords; all must match. Optional.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Only entries having all of these tags. Optional.",
            },
            "all": {
                "type": "boolean",
                "description": "Ignore filters and list the most recent entries.",
            },
            "limit": {
                "type": "integer",
                "description": "Max entries to return (1-100). Optional, default 10.",
            },
        },
        "required": [],
        "additionalProperties": False,
    },
    handler=handle,
)
