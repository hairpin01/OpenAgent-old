"""Tool that inspects the replied-to message in the Telegram chat."""

from __future__ import annotations

from typing import Any

from ..core import Tool, ToolContext


async def handle(ctx: ToolContext, args: dict[str, Any]) -> str:
    event = ctx.event
    client = ctx.client if ctx.client is not None else getattr(event, "client", None)
    if client is None and event is None:
        raise ValueError("no Telegram client or event in this context")

    reply = None

    if event is None:
        raise ValueError("no Telegram event in this context")
    msg = getattr(event, "message", None) or event
    if hasattr(msg, "get_reply_message"):
        reply = await msg.get_reply_message()

    if reply is None:
        return "No reply message found (current message is not replying to any message)."

    date = reply.date.strftime("%Y-%m-%d %H:%M:%S") if getattr(reply, "date", None) else "unknown"
    sender_name = ""
    sender_id = getattr(reply, "sender_id", None)
    if client is not None and sender_id is not None:
        try:
            user = await client.get_entity(sender_id)
            if user is not None:
                sender_name = getattr(user, "title", None) or (
                    f"{user.first_name or ''} {user.last_name or ''}".strip()
                    if hasattr(user, "first_name")
                    else str(getattr(user, "username", "unknown"))
                )
                if getattr(user, "username", None):
                    sender_name += f" (@{user.username})"
        except Exception:
            sender_name = str(sender_id)
    else:
        sender_name = str(sender_id or "unknown")

    media_info = []
    if getattr(reply, "photo", None):
        media_info.append("Photo")
    if getattr(reply, "video", None):
        media_info.append("Video")
    if getattr(reply, "video_note", None):
        media_info.append("Round video note")
    if getattr(reply, "voice", None):
        media_info.append("Voice message")
    if getattr(reply, "audio", None):
        media_info.append("Audio")
    if getattr(reply, "document", None):
        doc = reply.document
        doc_name = getattr(doc, "file_name", None) or "document"
        mime = getattr(doc, "mime_type", "unknown")
        size = getattr(doc, "size", 0)
        media_info.append(f"Document ({doc_name}, {mime}, {size} bytes)")
    if getattr(reply, "sticker", None):
        sticker = reply.sticker
        alt = getattr(sticker, "alt", "")
        media_info.append(f"Sticker ({alt})" if alt else "Sticker")
    if getattr(reply, "poll", None):
        media_info.append("Poll")
    if getattr(reply, "contact", None):
        media_info.append("Contact")
    if getattr(reply, "geo", None):
        media_info.append("Location")

    text = (getattr(reply, "message", None) or getattr(reply, "text", None) or "").strip()

    lines = [
        f"Reply message ID: {reply.id}",
        f"Chat ID: {getattr(reply, 'chat_id', 'unknown')}",
        f"Date: {date}",
        f"Sender: {sender_name} (ID: {sender_id})",
    ]
    if getattr(reply, "reply_to_msg_id", None):
        lines.append(f"Replying to msg ID: {reply.reply_to_msg_id}")
    if media_info:
        lines.append(f"Media: {', '.join(media_info)}")
    lines.append("Text:")
    lines.append(text if text else "(empty text / media only)")

    return "\n".join(lines)


TOOL = Tool(
    name="get_reply_message",
    description=(
        "Get information and text content of the message that the current command "
        "is replying to (or the reply to a specific message_id). Returns sender info, "
        "date, media types, and text."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    handler=handle,
    timeout=30,
)
