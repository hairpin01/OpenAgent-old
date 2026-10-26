from __future__ import annotations

import html
from html.parser import HTMLParser

from ....Settings import (
    ALLOWED_TAGS,
    VOID_TAGS,
    BLOCK_TAGS_FORBIDDEN_IN_CELLS,
)


class _RichParser(HTMLParser):

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.stack: list[str] = []
        self.out: list[str] = []
        self.in_cell: int = 0

    def handle_starttag(self, tag, attrs):
        if tag not in ALLOWED_TAGS:
            self.out.append(html.escape(self.get_starttag_text() or ""))
            return

        if tag in BLOCK_TAGS_FORBIDDEN_IN_CELLS and self.in_cell > 0:
            self.out.append(html.escape(self.get_starttag_text() or ""))
            return

        self.stack.append(tag)
        self.out.append(self.get_starttag_text() or f"<{tag}>")

        if tag in {"td", "th"}:
            self.in_cell += 1

    def handle_startendtag(self, tag, attrs):
        if tag in ALLOWED_TAGS:
            self.out.append(self.get_starttag_text() or f"<{tag}/>")
        else:
            self.out.append(html.escape(self.get_starttag_text() or ""))

    def handle_endtag(self, tag):
        if tag == "td" or tag == "th":
            self.in_cell = max(0, self.in_cell - 1)

        if tag in ALLOWED_TAGS and tag in self.stack:
            while self.stack:
                top = self.stack.pop()
                self.out.append(f"</{top}>")
                if top == tag:
                    break
        elif tag not in ALLOWED_TAGS:
            self.out.append(html.escape(f"</{tag}>"))

    def handle_data(self, data):
        self.out.append(data)

    def close_all(self) -> str:
        while self.stack:
            self.out.append(f"</{self.stack.pop()}>")
        return "".join(self.out)


def _trim_html(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text

    cut = text[:limit]
    if "<" in cut and cut.rfind("<") > cut.rfind(">"):
        cut = cut[: cut.rfind("<")]

    parser = _RichParser()
    parser.feed(cut)
    balanced = parser.close_all()

    if len(balanced) > limit:
        return _trim_html(balanced, limit - 20) + "…"
    return balanced.rstrip() + "…"


def render_body(text: str, limit: int) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    if "<" not in text:
        text = f"<p>{html.escape(text)}</p>"

    parser = _RichParser()
    parser.feed(text)
    balanced = parser.close_all()
    return _trim_html(balanced, limit)
