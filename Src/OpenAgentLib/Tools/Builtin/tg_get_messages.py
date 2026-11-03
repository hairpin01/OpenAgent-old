"""Tool that reads recent Telegram messages via the userbot Telethon client."""

from __future__ import annotations

from ..core import AgentPlugin, Tool, ToolContext


async def handle(ctx: ToolContext, args: dict) -> str:
    client = ctx.client
    if client is None:
        raise ValueError("no Telegram client in this context")

    # Default to the current chat; explicit chat_id overrides it
    chat = args.get("chat_id") or ctx.chat_id
    if chat is None:
        raise ValueError("no chat: pass chat_id or use a chat event")

    limit = int(args.get("limit") or 10)
    limit = max(1, min(limit, 50))
    from_me = bool(args.get("from_me"))

    results = []
    count = 0
    async for message in client.iter_messages(chat, limit=limit * 3):
        if from_me and message.out is False:
            continue
        count += 1
        if count > limit:
            break
        date = message.date.strftime("%Y-%m-%d %H:%M")
        sender_name = ""
        try:
            user = (
                await client.get_entity(message.sender_id)
                if message.sender_id
                else None
            )
            if user is not None:
                sender_name = getattr(user, "title", None) or (
                    f"{user.first_name} {user.last_name}".strip()
                    if hasattr(user, "first_name")
                    else str(getattr(user, "username", "unknown"))
                )
        except Exception:
            sender_name = str(message.sender_id or "")
        text = (message.message or "").strip()
        if not text:
            text = f"[media: {message.media or 'unknown'}]"
        results.append(f"--- msg {message.id} | {date} | {sender_name}\n{text}")

    if not results:
        return "No messages found."
    return "\n\n".join(results)


PLUGIN = AgentPlugin(
    name="tg_get_messages",
    description=(
        "Get the latest messages from a Telegram chat (oldest first). "
        "Optional: chat_id (default = current chat), from_me (only messages "
        "from the userbot), limit (1-50, default 10). Returns each message "
        "with id, date, sender and text; media-only messages show a placeholder."
    ),
    parameters={
        "type": "object",
        "properties": {
            "chat_id": {
                "type": ["integer", "string"],
                "description": "Chat id to read from. Optional, defaults to current chat.",
            },
            "from_me": {
                "type": "boolean",
                "description": "Only messages sent by the userbot itself.",
            },
            "limit": {
                "type": "integer",
                "description": "Max messages to return (1-50). Optional, default 10.",
            },
        },
        "additionalProperties": False,
    },
    handler=handle,
    timeout=30,
)

TOOL = PLUGIN
