"""Web search tool: DuckDuckGo (HTML endpoint, no API key needed)."""

from __future__ import annotations

import html
import re
from html import unescape
from typing import TYPE_CHECKING

from ..core import Tool, ToolContext

if TYPE_CHECKING:
    from OpenAgentMain import OpenAgent
    from ...Format.TextRender import RenderContext

_SEARCH_URL = "https://html.duckduckgo.com/html/"
_HEADERS = {"User-Agent": "MCUB-agent/1.0 (web_search tool)"}

_RESULT_RE = re.compile(
    r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S
)
_SNIPPET_RE = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_UDDG_RE = re.compile(r"[?&]uddg=([^&]+)")


def _clean(text: str) -> str:
    return unescape(_TAG_RE.sub(" ", text)).strip()


def _fix_url(href: str) -> str:
    match = _UDDG_RE.search(href)
    if match:
        from urllib.parse import unquote

        return unquote(match.group(1))
    if href.startswith("//"):
        return "https:" + href
    return href


def _parse(page: str, limit: int) -> list[tuple[str, str, str]]:
    titles = _RESULT_RE.findall(page)
    snippets = [_clean(snippet) for snippet in _SNIPPET_RE.findall(page)]
    results: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for index, (href, title) in enumerate(titles):
        url = _fix_url(href)
        if not url.startswith("http") or url in seen:
            continue
        seen.add(url)
        results.append((_clean(title), url, snippets[index] if index < len(snippets) else ""))
        if len(results) >= limit:
            break
    return results


def _results_module(agent: "OpenAgent", ctx: "RenderContext") -> str | None:
    """Details block with the search results, shown in the result format."""
    data = (ctx.state or {}).get("web_search_results")
    if not data:
        return None
    query = html.escape(str(data.get("query") or ""))
    rows: list[str] = []
    for title, url, snippet in data.get("items") or []:
        label = url if len(url) <= 20 else url[:20] + "..."
        button = (
            f"<tg-button type='url' url='{html.escape(url, quote=True)}'>"
            f"{html.escape(label)}</tg-button>"
        )
        description = f"{title} — {snippet}" if snippet else title
        rows.append(
            f"<p>{button}</p>\n<blockquote>{html.escape(description)}</blockquote>"
        )
    if not rows:
        return None
    return (
        f"<details><summary>Query: {query}</summary><blockquote>\n"
        + "\n".join(rows)
        + "\n</blockquote></details>"
    )


async def handle(ctx: ToolContext, args: dict) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        raise ValueError("query is empty")
    count = int(args.get("count") or 5)
    count = max(1, min(count, 10))

    if ctx.http is None:
        raise ValueError("http session is not available")

    async with ctx.http.get(
        _SEARCH_URL, params={"q": query}, headers=_HEADERS
    ) as response:
        page = await response.text()

    results = _parse(page, count)
    if not results:
        return f"No results for: {query}"

    # Show the results in the final message (after the body module)
    if ctx.state is not None:
        ctx.state["web_search_results"] = {"query": query, "items": results}
        ctx.add_module("result", "web_search_results", _results_module, after="body")

    lines = []
    for index, (title, url, snippet) in enumerate(results, 1):
        lines.append(f"{index}. {title}\n{url}")
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines)


TOOL = Tool(
    name="web_search",
    description=(
        "Search the web and return titles, URLs and short snippets. "
        "DuckDuckGo based, no API key required."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "count": {
                "type": "integer",
                "description": "Number of results (1-10). Optional, default 5.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    handler=handle,
    timeout=30,
)
