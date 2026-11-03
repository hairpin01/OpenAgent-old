"""Tool that downloads and gets info about media in the replied-to message."""

from __future__ import annotations

import os
import tempfile
from typing import Any

from ..core import AgentPlugin, Tool, ToolContext
from ..workspace import resolve_path


async def handle(ctx: ToolContext, args: dict[str, Any]) -> str:
    event = ctx.event
    client = ctx.client if ctx.client is not None else getattr(event, "client", None)
    if client is None and event is None:
        raise ValueError("no Telegram client or event in this context")

    target_msg_id = args.get("message_id")
    target_msg = None

    if target_msg_id is not None:
        chat = args.get("chat_id") or ctx.chat_id or getattr(event, "chat_id", None)
        if chat is None:
            raise ValueError("no chat: pass chat_id or run in a chat context")
        msg = await client.get_messages(chat, ids=int(target_msg_id))
        if msg is None:
            return f"Message {target_msg_id} not found."
        if getattr(msg, "media", None):
            target_msg = msg
        else:
            target_msg = await msg.get_reply_message()
    else:
        if event is None:
            raise ValueError("no Telegram event in this context")
        msg = getattr(event, "message", None) or event
        if hasattr(msg, "get_reply_message"):
            target_msg = await msg.get_reply_message()

    if target_msg is None:
        return "No reply message found."

    if not getattr(target_msg, "media", None):
        return f"Message {target_msg.id} has no media attached."

    output_arg = args.get("path")
    if output_arg:
        path_obj, _ = resolve_path(output_arg)
        if str(output_arg).endswith(("/", "\\")) or path_obj.is_dir():
            path_obj.mkdir(parents=True, exist_ok=True)
            download_dest = str(path_obj)
        else:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            download_dest = str(path_obj)
    else:
        temp_dir = os.path.join(tempfile.gettempdir(), "openagent_media")
        os.makedirs(temp_dir, exist_ok=True)
        download_dest = temp_dir

    saved_path = await client.download_media(target_msg, file=download_dest)
    if not saved_path or not os.path.exists(saved_path):
        return "Failed to download media from message."

    saved_path = os.path.abspath(saved_path)
    file_size = os.path.getsize(saved_path)
    file_name = os.path.basename(saved_path)

    if file_size < 1024:
        size_str = f"{file_size} B"
    elif file_size < 1024 * 1024:
        size_str = f"{file_size / 1024:.1f} KB"
    else:
        size_str = f"{file_size / (1024 * 1024):.2f} MB"

    mime = getattr(getattr(target_msg, "document", None), "mime_type", "unknown")

    lines = [
        "Media downloaded successfully.",
        f"Message ID: {target_msg.id}",
        f"File name: {file_name}",
        f"Saved path: {saved_path}",
        f"File size: {size_str} ({file_size} bytes)",
        f"MIME type: {mime}",
    ]

    if file_size <= 32768:
        try:
            with open(saved_path, "r", encoding="utf-8") as f:
                content = f.read()
            lines.append("Content preview:\n```\n" + content[:2000] + ("\n..." if len(content) > 2000 else "") + "\n```")
        except Exception:
            pass

    return "\n".join(lines)


PLUGIN = AgentPlugin(
    name="get_reply_media",
    description=(
        "Download and retrieve information about media attached to the replied-to "
        "Telegram message (or specified message_id). Saves the file to disk and "
        "returns the file path, size, MIME type, and optional text preview."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Optional destination file path or directory to save the media into. Defaults to temporary media directory.",
            },
            "message_id": {
                "type": "integer",
                "description": "Optional message ID containing or replying to media. Defaults to current reply.",
            },
            "chat_id": {
                "type": ["integer", "string"],
                "description": "Chat id if message_id is provided. Optional, defaults to current chat.",
            },
        },
        "additionalProperties": False,
    },
    handler=handle,
    timeout=60,
)

TOOL = PLUGIN
