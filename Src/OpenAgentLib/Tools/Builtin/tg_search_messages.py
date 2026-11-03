"""Tool that searches Telegram messages via the userbot Telethon client."""

from __future__ import annotations

from ..core import AgentPlugin, Tool, ToolContext


async def handle(ctx: ToolContext, args: dict) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        raise ValueError("query is empty")

    client = ctx.client
    if client is None:
        raise ValueError("no Telegram client in this context")

    # Default to the current chat; explicit chat_id overrides it
    chat = args.get("chat_id") or ctx.chat_id
    if chat is None:
        raise ValueError("no chat: pass chat_id or use a chat event")

    if args.get("from_me"):
        query = f"from:me {query}"
    if args.get("from_sender") is not None:
        sender = args["from_sender"]
        if isinstance(sender, int) and not isinstance(sender, bool):
            query = f"from:{sender} {query}"
        else:
            query = f"from:{sender} {query}"

    limit = int(args.get("limit") or 10)
    limit = max(1, min(limit, 50))

    results = []
    async for message in client.iter_messages(
        chat, search=query, limit=limit,
    ):
        date = message.date.strftime("%Y-%m-%d %H:%M")
        sender_name = ""
        try:
            user = await client.get_entity(message.sender_id) if message.sender_id else None
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
        return f"No messages found for: {query}"
    return "\n\n".join(results)


PLUGIN = AgentPlugin(
    name="tg_search_messages",
    description=(
        "Search Telegram messages in a chat using Telethon full-text search. "
        "query is required. Optional: chat_id (default = current chat), "
        "from_me (only messages from the userbot), from_sender (user id or "
        "username to filter by), limit (1-50, default 10). Returns up to "
        "limit messages with id, date, sender and text."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Text to search for in message bodies.",
            },
            "chat_id": {
                "type": ["integer", "string"],
                "description": "Chat id to search in. Optional, defaults to current chat.",
            },
            "from_me": {
                "type": "boolean",
                "description": "Only search messages sent by the userbot itself.",
            },
            "from_sender": {
                "type": ["integer", "string"],
                "description": "Filter by sender: user id or @username. Optional.",
            },
            "limit": {
                "type": "integer",
                "description": "Max messages to return (1-50). Optional, default 10.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    handler=handle,
    timeout=30,
)

TOOL = PLUGIN
