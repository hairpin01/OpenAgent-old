"""Todo tool: the agent's visible plan for a multi-step task."""

from __future__ import annotations

import json

from Settings import TODO_ITEM_CHARS, TODO_MAX_ITEMS

from ..core import Tool, ToolContext

STATUSES = ("pending", "in_progress", "completed")


async def handle(ctx: ToolContext, args: dict) -> str:
    raw = args.get("todos")
    if not isinstance(raw, list):
        raise ValueError("todos must be an array")
    if len(raw) > TODO_MAX_ITEMS:
        raise ValueError(f"at most {TODO_MAX_ITEMS} items")

    items: list[dict[str, str]] = []
    for index, entry in enumerate(raw, 1):
        if isinstance(entry, str):
            entry = {"content": entry}
        if not isinstance(entry, dict):
            raise ValueError(f"item {index}: must be an object")
        content = " ".join(str(entry.get("content") or "").split())
        if not content:
            raise ValueError(f"item {index}: content is empty")
        status = str(entry.get("status") or "pending").strip().lower()
        if status not in STATUSES:
            raise ValueError(f"item {index}: status must be one of {', '.join(STATUSES)}")
        items.append({"content": content[:TODO_ITEM_CHARS], "status": status})

    # Rendered by the "todo" text module in the running and the final message
    ctx.state["todos"] = items

    current = next((i for i in items if i["status"] == "in_progress"), None)
    current = current or next((i for i in items if i["status"] == "pending"), None)
    return json.dumps(
        {
            "ok": True,
            "done": sum(1 for i in items if i["status"] == "completed"),
            "total": len(items),
            "current": current["content"] if current else None,
        },
        ensure_ascii=False,
    )


TOOL = Tool(
    name="todo",
    description=(
        "Write or update your plan for a task of 3 or more steps; the user "
        "sees it as a checklist. Always pass the FULL list: it replaces the "
        "previous one. Statuses: pending, in_progress (at most one), "
        "completed. Call it together with the next real tool call, never "
        "alone. An empty list clears the plan."
    ),
    parameters={
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "The whole plan, in execution order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "One short step.",
                        },
                        "status": {
                            "type": "string",
                            "enum": list(STATUSES),
                            "description": "Default pending.",
                        },
                    },
                    "required": ["content"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["todos"],
        "additionalProperties": False,
    },
    handler=handle,
)
