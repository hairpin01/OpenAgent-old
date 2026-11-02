from __future__ import annotations

import html
import re
from html.parser import HTMLParser

from Settings import (
    ALLOWED_TAGS,
    VOID_TAGS,
    BLOCK_TAGS_FORBIDDEN_IN_CELLS,
)


class _RichParser(HTMLParser):

    def __init__(self):
        # convert_charrefs: "&lt;" arrives as "<" in handle_data and is
        # re-escaped there (convert_charrefs=False silently dropped entities)
        super().__init__(convert_charrefs=True)
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

        if tag in VOID_TAGS:
            # <hr> / <img> without a slash have no end tag: never open them
            self.out.append(self.get_starttag_text() or f"<{tag}>")
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
        if tag in ALLOWED_TAGS and tag in self.stack:
            while self.stack:
                top = self.stack.pop()
                # A closing tag may implicitly close cells opened inside it
                if top in {"td", "th"}:
                    self.in_cell = max(0, self.in_cell - 1)
                self.out.append(f"</{top}>")
                if top == tag:
                    break
        elif tag not in ALLOWED_TAGS:
            self.out.append(html.escape(f"</{tag}>"))

    def handle_data(self, data):
        self.out.append(html.escape(data, quote=False))

    def close_all(self) -> str:
        # Flush text the parser still holds back (e.g. a trailing "R&D")
        self.close()
        while self.stack:
            self.out.append(f"</{self.stack.pop()}>")
        return "".join(self.out)


def _trim_html(text: str, limit: int) -> str:
    """Cut balanced ``text`` to ``limit`` chars and close what the cut left open."""
    if len(text) <= limit:
        return text

    budget = limit - 1  # room for the ellipsis
    while budget > 0:
        cut = text[:budget]
        if "<" in cut and cut.rfind("<") > cut.rfind(">"):
            cut = cut[: cut.rfind("<")]

        parser = _RichParser()
        parser.feed(cut)
        balanced = parser.close_all().rstrip()

        overshoot = len(balanced) + 1 - limit
        if overshoot <= 0:
            return balanced + "…"
        # The closing tags added by balancing did not fit: cut earlier by
        # exactly that much (normally settles on the next pass)
        budget -= overshoot
    return "…"


# A tag or entity cut in half at the end of a streamed text: "<", "<b",
# "</bl", "<a href=\"ht", "&", "&am". A "<" followed by a space or digit is
# plain text and is not matched.
_TAIL_INCOMPLETE = re.compile(r"(?:<(?:[A-Za-z/!][^<>]*)?|&#?[A-Za-z0-9]{0,31})$")

# Element without content, e.g. a just-opened <ul><li></li></ul>. Table cells
# stay: an empty <td> is a valid cell.
_EMPTY_ELEMENT = re.compile(
    r"<(?!(?:td|th)[\s>])([A-Za-z][\w-]*)(?:\s[^<>]*)?>\s*</\1>"
)


def _prune_empty(markup: str) -> str:
    """Drop empty elements, repeatedly (nested empties collapse level by level)."""
    while True:
        markup, count = _EMPTY_ELEMENT.subn("", markup)
        if not count:
            return markup


def render_body(text: str, limit: int, *, partial: bool = False) -> str:
    """Render model HTML as a balanced rich body.

    ``partial=True`` is for a text that is still being generated: a tag or
    entity cut in half at the end is held back until it is complete, tags that
    are not closed yet are closed in the output (nested ones too), and
    elements left empty by that are dropped.
    """
    text = (text or "").strip()
    if not text:
        return ""

    # Decided on the raw text, so the layout does not jump when the first
    # "<" is still incomplete
    has_markup = "<" in text
    if partial:
        text = _TAIL_INCOMPLETE.sub("", text).rstrip()
        if not text:
            return ""

    if not has_markup:
        text = f"<p>{html.escape(text)}</p>"

    parser = _RichParser()
    parser.feed(text)
    balanced = parser.close_all()
    if partial:
        balanced = _prune_empty(balanced)
    return _trim_html(balanced, limit)
