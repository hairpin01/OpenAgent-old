"""Web page fetch tool: downloads a URL and returns readable text."""

from __future__ import annotations

import re
from html import unescape

import aiohttp

from ..core import AgentPlugin, Tool, ToolContext

_HEADERS = {"User-Agent": "MCUB-agent/1.0 (web_fetch tool)"}
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)
_BLOCK_TAGS = re.compile(
    r"</?(p|div|br|li|tr|h[1-6]|section|article|header|footer|blockquote)[^>]*>",
    re.I,
)
_DROP_RE = re.compile(r"<(script|style|head|noscript)[^>]*>.*?</\1>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")


def _html_to_text(text: str) -> str:
    text = _DROP_RE.sub(" ", text)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = _BLOCK_TAGS.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    text = unescape(text)
    # Collapse runs of spaces on the same line, then blank line runs
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()


async def handle(ctx: ToolContext, args: dict) -> str:
    url = str(args.get("url", "")).strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError("url must start with http:// or https://")
    limit = int(args.get("max_chars") or 20000)
    limit = max(500, min(limit, 60000))

    if ctx.http is None:
        raise ValueError("http session is not available")

    async with ctx.http.get(
        url,
        headers=_HEADERS,
        timeout=aiohttp.ClientTimeout(total=30),
    ) as response:
        content_type = response.headers.get("Content-Type", "")
        if content_type.startswith("image/"):
            return f"[image: {content_type}]"
        if not any(
            kind in content_type
            for kind in ("text/", "json", "xml", "javascript")
        ):
            body = await response.read()
            return f"[binary content: {content_type or 'unknown'}; {len(body)} bytes]"
        page = await response.text(errors="replace")

    title = ""
    match = _TITLE_RE.search(page)
    if match:
        title = unescape(_TAG_RE.sub(" ", match.group(1))).strip()

    if "html" in content_type:
        body = _html_to_text(page)
    else:
        body = page.strip()
    if len(body) > limit:
        body = body[:limit].rstrip() + "…"

    header = f"[{response.status}] {title}\n" if title else f"[{response.status}]\n"
    return header + body


PLUGIN = AgentPlugin(
    name="web_fetch",
    description=(
        "Fetch a URL and return its content as readable text "
        "(HTML is converted to plain text, JSON/plain text is returned as is). "
        "Binary content is only reported by type and size."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "http(s) URL to fetch."},
            "max_chars": {
                "type": "integer",
                "description": "Max characters of returned text. "
                "Optional, default 20000, max 60000.",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    },
    handler=handle,
    timeout=40,
)

TOOL = PLUGIN
