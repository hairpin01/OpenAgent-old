"""Thinking tool: append short progress notes to the running message."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

from ..core import Tool, ToolContext

if TYPE_CHECKING:
    from OpenAgentMain import OpenAgent
    from ...Format.TextRender import RenderContext

MAX_LINES = 3  # only the newest notes are shown, older ones are trimmed


def _notes_module(agent: "OpenAgent", ctx: "RenderContext") -> str | None:
    """The last thinking notes as an italic blockquote with a cite."""
    lines = (ctx.state or {}).get("thinking_lines")
    if not lines:
        return None
    body = "\n\n".join(f"<i>{html.escape(str(line))}</i>" for line in lines)
    # Single newline after the cite; a blank line only *between* the notes.
    return (
        f"<blockquote><cite>{html.escape(agent.strings('thinking_title'))}</cite>\n"
        f"{body}</blockquote>"
    )


async def handle(ctx: ToolContext, args: dict) -> str:
    # One short note; flatten any newlines so it stays a single line
    text = " ".join(str(args.get("text", "")).split())
    if not text:
        raise ValueError("text is empty")

    if bool(args.get("reset")):
        lines = ctx.state.get("thinking_lines") or []
        lines.clear()
    else:
        lines = ctx.state.setdefault("thinking_lines", [])
    lines.append(text)
    del lines[:-MAX_LINES]  # keep the newest notes only

    # The running message already renders ``thinking_lines`` through the
    # built-in "thinking" module, so adding the notes there again would show
    # the block twice. Only the final message needs an extra module.
    ctx.add_module("result", "thinking_notes", _notes_module, after="body")

    return "\n".join(lines)


TOOL = Tool(
    name="thinking",
    description=(
        "Add a short note to the 'Thinking' block shown to the user while you "
        "work (e.g. what you are doing, a hiccup, a plan). "
        f"Only the last {MAX_LINES} notes are visible, older ones are trimmed. "
        "Call it between tool calls. Use reset=true to start a fresh block."
    ),
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The note text (one short sentence).",
            },
            "reset": {
                "type": "boolean",
                "description": "Clear previous notes first. Optional, default false.",
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    },
    handler=handle,
)
