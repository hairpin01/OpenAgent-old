"""Modular rich-text rendering.

A text format is a declarative list of small modules. Every module receives
``(agent, ctx)`` and returns an HTML chunk or ``None``. Formats are assembled
with newline separators, so adding a block or changing its order does not
require touching the message-building methods on the agent.
"""

from __future__ import annotations

import html
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

from Settings import CUSTOM_EMOJI, MAX_REPLY_CHARS

if TYPE_CHECKING:
    from OpenAgentMain import OpenAgent

from .RichBoby import render_body
from ..Tools.workspace import make_diff
from ...Settings import (
    HISTORY_NAV_TTL,
    MAX_DETAIL_CHARS,
    MODELS_NAV_TTL,
    TOOL_BAR_MAX_PILLS,
    TOOL_BAR_TTL,
    TOOL_DETAIL_CHARS,
)

@dataclass
class RenderContext:
    """Everything modules need to build a message body."""

    kind: str = ""  # "running" | "done" | "error" | "cancelled"
    text: str = ""  # main body text
    state: dict[str, Any] | None = None  # active request state
    extra: dict[str, Any] = field(default_factory=dict)  # models, labels, ...
    copy_text: str = "" # copy text
    label: str = ""


TEXT_MODULES: dict[str, Callable[["OpenAgent", RenderContext], str | None]] = {}
TEXT_FORMATS: dict[str, tuple[str, ...]] = {}

def text_module(name: str):
    """Register a rendering module under ``name``."""

    def decorator(fn):
        TEXT_MODULES[name] = fn
        return fn

    return decorator


def text_format(name: str, *modules: str) -> None:
    """Register a format as an ordered list of module names."""

    TEXT_FORMATS[name] = modules


def render(agent: "OpenAgent", format_name: str, ctx: RenderContext) -> str:
    """Render ``format_name`` by joining non-empty module chunks."""

    chain = [(name, None) for name in TEXT_FORMATS.get(format_name, ())]
    # Modules added by tools for this request only (see ToolContext.add_module)
    extra = ((ctx.state or {}).get("ui") or {}).get(format_name, ())
    for after, name, fn in extra:
        names = [item[0] for item in chain]
        index = names.index(after) + 1 if after in names else len(chain)
        chain.insert(index, (name, fn))

    parts: list[str] = []
    for module_name, fn in chain:
        module = fn or TEXT_MODULES.get(module_name)
        if module is None:
            continue
        chunk = module(agent, ctx)
        if chunk:
            parts.append(chunk)
    return "\n".join(parts)


@text_module("status_header")
def _header(agent: "OpenAgent", ctx: RenderContext) -> str | None:
    state = ctx.state
    if state is None:
        return None

    elapsed = max(time.perf_counter() - state["start"], 0.0)
    model = html.escape(str(state["model"]))
    provider = html.escape(str(state["provider"]))

    if ctx.kind == "running":
        badges = [
            f"<tg-pill style='success'>{model}</tg-pill>",
            f"<tg-pill style='primary'>{provider}</tg-pill>",
        ]
    else:
        if ctx.kind == "done":
            style, label = "success", agent.strings("status_done")
        elif ctx.kind == "error":
            style, label = "danger", agent.strings("status_error")
        else:
            style, label = "danger", agent.strings("status_cancelled")
        badges = [
            f"<tg-pill style='{style}'>{html.escape(label)}</tg-pill>",
            f"<tg-pill style='primary'>{model}</tg-pill>",
        ]

    tool = str(state.get("tool") or "")
    tool_calls = int(state.get("tool_calls") or 0)
    if tool:
        badges.append(f"<tg-pill style='primary'>🔧 {html.escape(tool)}</tg-pill>")
    elif tool_calls:
        badges.append(f"<tg-pill style='primary'>🔧 {tool_calls}</tg-pill>")

    ctx_count = int(state.get("context_count") or 0)
    ctx_size = int(agent.config.get("context_size") or 0)
    ctx_label = "∞" if ctx_size == 0 else ctx_size
    badges.append(f"<tg-pill style='primary'>🧠 {ctx_count}/{ctx_label}</tg-pill>")

    badges.append(f"<tg-pill style='primary'>⏱ {elapsed:.2f}s</tg-pill>")
    return f"<p>{' '.join(badges)}</p>"

@text_module("standalone_header")
def _standalone_header(agent: "OpenAgent", ctx: RenderContext) -> str:
    """Simple header without an active request (errors, service screens)."""
    label = ctx.extra.get("label", agent.strings("status_error"))
    style = ctx.extra.get("style", "danger")
    model = html.escape(str(agent.config["model"]))
    return (
        f"<p><tg-pill style='{style}'>{html.escape(label)}</tg-pill> "
        f"<tg-pill style='primary'>{model}</tg-pill></p>"
    )


@text_module("usage_header")
def _usage_header(agent: "OpenAgent", ctx: RenderContext) -> str:
    model = html.escape(str(agent.config["model"]))
    return (
        f"<p><tg-pill style='success'>OpenAgent</tg-pill> "
        f"<tg-pill style='primary'>{model}</tg-pill></p>"
    )


STICK_FRAMES = ('⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏')
STICK_FRAME_SECONDS = 0.15


def stick_frame(elapsed: float) -> str:
    """Return a stick-spinner frame for ``elapsed`` seconds."""

    index = int(max(elapsed, 0.0) / STICK_FRAME_SECONDS) % len(STICK_FRAMES)
    return STICK_FRAMES[index]


@text_module("stick_animation")
def _stick_animation(agent: "OpenAgent", ctx: RenderContext) -> str:
    """Render one frame of the rotating stick animation."""

    start = (ctx.state or {}).get("start")
    now = time.perf_counter()
    elapsed = now - float(start) if start is not None else now
    return f"<tg-pill style='primary'>{html.escape(stick_frame(elapsed))}</tg-pill>"


@text_module("thinking")
def _thinking(agent: "OpenAgent", ctx: RenderContext) -> str:
    return (
        f"<blockquote><b>{_stick_animation(agent, ctx)}</b> "
        f"<i>{html.escape(agent.strings('thinking'))}</i></blockquote>"
    )


@text_module("cancel_button")
def _cancel_button(agent: "OpenAgent", ctx: RenderContext) -> str | None:
    return (ctx.state or {}).get("cancel_tag") or ctx.extra.get("cancel_tag")



@text_module("body")
def _body(agent: "OpenAgent", ctx: RenderContext) -> str:
    if not ctx.text:
        return f"<blockquote>{html.escape(agent.strings('err_bad_reply'))}</blockquote>"
    return render_body(ctx.text, MAX_REPLY_CHARS)


@text_module("error_detail")
def _error_detail(agent: "OpenAgent", ctx: RenderContext) -> str | None:
    """Raw provider error text, in a collapsible block under the body."""
    if ctx.kind != "error":
        return None
    detail = str((ctx.state or {}).get("error_detail") or "").strip()
    if not detail:
        return None
    return (
        f"<details><summary>{html.escape(agent.strings('error_detail_title'))}</summary>"
        f"<blockquote><cite>{html.escape(agent.strings('error_detail_title'))}</cite>\n"
        f"<code>{html.escape(detail)}</code></blockquote></details>"
    )


@text_module("cancelled_body")
def _cancelled_body(agent: "OpenAgent", ctx: RenderContext) -> str:
    return f"<blockquote>{html.escape(agent.strings('cancelled_body'))}</blockquote>"


@text_module("usage_body")
def _usage_body(agent: "OpenAgent", ctx: RenderContext) -> str:
    mention = agent._guest_mention()
    return (
        f"<blockquote>{html.escape(agent.strings('usage', mention=mention))}"
        "</blockquote>"
    )


@text_module("copy_button")
def _copy_button(agent: "OpenAgent", ctx: RenderContext) -> str:
    return (
        "<tg-button-row>"
        f"<tg-button type='copy_text' text='{html.escape(ctx.copy_text)}' style='primary'>"
        f"{ctx.label}</tg-button></tg-button-row>"
    )


@text_module("models_header")
def _models_header(agent: "OpenAgent", ctx: RenderContext) -> str:
    page = int(ctx.extra.get("page") or 0)
    pages = int(ctx.extra.get("pages") or 1)
    pager = (
        f" <tg-pill style='primary'>{page + 1}/{pages}</tg-pill>"
        if pages > 1
        else ""
    )
    return (
        f"<p>{CUSTOM_EMOJI['bals']} "
        f"<b>{html.escape(agent.strings('models_title'))}</b>{pager}</p>"
    )


@text_module("models_list")
def _models_list(agent: "OpenAgent", ctx: RenderContext) -> str:
    models = ctx.extra.get("models") or []
    if not models:
        return f"<blockquote>{html.escape(agent.strings('models_empty'))}</blockquote>"

    current = str(agent.config["model"])
    state = ctx.state or {}
    rows = []
    for model in models:
        selected = model == current
        style = "success" if selected else "link"
        marker = '  ' + CUSTOM_EMOJI["ok"] if selected else "  "
        button = agent.Button.rich.inline(
            model,
            agent.on_change_model,
            data=(state, model),
            style=style,
            html_tag=True,
        )
        rows.append(f"{marker} {button}")

    current = str(agent.config["model"])
    provider = agent._provider_label()
    footer = (
        f"<p><tg-pill style='success'>{html.escape(current)}</tg-pill> "
        f"<tg-pill style='primary'>{html.escape(provider)}</tg-pill></p>"
    )

    return "\n".join(
        ["\n".join(rows), f'<blockquote>{footer}</blockquote']
    )

@text_module("models_nav")
def _models_nav(agent: "OpenAgent", ctx: RenderContext) -> str | None:
    """Back / Next row for the paginated model list."""
    state = ctx.state or {}
    page = int(ctx.extra.get("page") or 0)
    pages = int(ctx.extra.get("pages") or 1)
    if pages <= 1:
        return None

    tags = state.setdefault("models_nav_tags", {})

    def tag(key: str, label: str, step: int) -> str:
        # Built once: every render would otherwise register a new callback
        if key not in tags:
            tags[key] = agent.Button.rich.inline(
                agent.strings(label),
                agent.on_models_scroll,
                data=(state, step),
                style="primary",
                ttl=MODELS_NAV_TTL,
                html_tag=True,
            )
        return tags[key]

    buttons: list[str] = []
    if page > 0:
        buttons.append(tag("back", "btn_back", -1))
    if page < pages - 1:
        buttons.append(tag("next", "btn_next", 1))
    if not buttons:
        return None
    return f"<tg-button-row>  {' '.join(buttons)}  </tg-button-row>"


@text_module("line_break")
def _line_break(agent: "OpenAgent", ctx: RenderContext) -> str:
    return "<hr/>\n"

@text_module("prompt")
def _prompt(agent: "OpenAgent", ctx: RenderContext) -> str:
    # A history view passes the stored prompt through ctx.extra
    prompt = ctx.extra.get("prompt", (ctx.state or {}).get("prompt", ""))
    return f"""<details><summary>{agent.strings('prompt')}</summary>⬅️ {html.escape(agent._clip(prompt))}</details>"""

def _bar_clip(text: object, limit: int) -> str:
    """Flatten ``text`` to one line and cut it to ``limit`` chars."""
    flat = str(text).replace("\n", " ")
    return flat[:limit] + "…" if len(flat) > limit else flat


def _fmt_tokens(value: object) -> str:
    """Short human count for the token widget: 15.5k, 980, 0."""
    n = int(value or 0)
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


@text_module("token_stats")
def _token_stats(agent: "OpenAgent", ctx: RenderContext) -> str | None:
    """Token usage of the latest provider step: prompt in / completion out / total."""
    usage = (ctx.state or {}).get("tokens") or {}
    if not usage:
        return None
    return (
        "<blockquote>"
        f"↑ {_fmt_tokens(usage.get('prompt_tokens'))} | "
        f"↓ {_fmt_tokens(usage.get('completion_tokens'))} | "
        f"~{_fmt_tokens(usage.get('total_tokens'))}"
        "</blockquote>"
    )


@text_module("tool_bar")
def _tool_bar(agent: "OpenAgent", ctx: RenderContext) -> str | None:
    """Feed of up to TOOL_BAR_MAX_PILLS pills, newest first; a finished pill
    toggles its details."""
    pills: list[str] = []
    for item in reversed(
        ((ctx.state or {}).get("tool_bar") or [])[-TOOL_BAR_MAX_PILLS:]
    ):
        name = item["name"]
        if item.get("status", "run") == "run":
            elapsed = max(time.perf_counter() - item.get("start", 0.0), 0.0)
            pills.append(
                f"<tg-pill style='primary'>{stick_frame(elapsed)} {elapsed:.1f}s "
                f"{html.escape(name)}</tg-pill>"
            )
            continue
        if "button" not in item:
            # Built once: every render would otherwise register a new callback.
            # data carries (state, item) so the handler can toggle and re-render.
            # An opened details block is highlighted as primary.
            if item.get("open"):
                style = "primary"
            else:
                style = "success" if item["status"] == "ok" else "danger"
            item["button"] = agent.Button.rich.inline(
                name,
                agent.on_tool_args,
                data=(ctx.state, item),
                style=style,
                ttl=TOOL_BAR_TTL,
                html_tag=True,
            )
        pills.append(item["button"])
    if not pills:
        return None
    return f"<blockquote><p>{' &gt; '.join(pills)}</p></blockquote>"


def _detail_clip(text: object) -> str:
    """Escaped one-line preview of ``text`` for the details block."""
    return html.escape(_bar_clip(text, TOOL_DETAIL_CHARS))


@text_module("tool_details")
def _tool_details(agent: "OpenAgent", ctx: RenderContext) -> str | None:
    """One details block per finished tool call that the user opened."""
    blocks: list[str] = []
    for item in (ctx.state or {}).get("tool_bar") or []:
        if not item.get("open") or item.get("status", "run") == "run":
            continue

        name = str(item["name"])
        ok = item["status"] == "ok"
        style = "success" if ok else "danger"
        label = agent.strings("tool_ok" if ok else "tool_failed")
        tool = next((t for t in agent._tools if t.name == name), None)
        description = tool.description if tool else ""
        elapsed = float(item.get("elapsed") or 0.0)

        # The tool-name pill is a button that collapses this details block.
        # Built once: every render would otherwise register a new callback.
        if "details_button" not in item:
            item["details_button"] = agent.Button.rich.inline(
                name,
                agent.on_tool_args,
                data=(ctx.state, item),
                style=style,
                ttl=TOOL_BAR_TTL,
                html_tag=True,
            )
        head = (
            f"<p>{item['details_button']} "
            f"<tg-pill style='{style}'>{html.escape(label)}</tg-pill></p>"
        )
        if description:
            head += f"\n<b>📝 {_detail_clip(description)}</b>"

        rows = [
            f"<b>{html.escape(agent.strings('tool_in'))}</b> "
            f"<code>{_detail_clip(item.get('args', ''))}</code>",
            f"<b>{html.escape(agent.strings('tool_out'))}</b> "
            f"<code>{_detail_clip(item.get('result', ''))}</code>",
            f"<b>{html.escape(agent.strings('tool_time'))}</b> "
            f"<p><tg-pill style='primary'>⏱️ {elapsed:.1f}s</tg-pill></p>",
        ]
        blocks.append(
            f"<blockquote><cite>{html.escape(agent.strings('tool_info_title'))}</cite>\n"
            f"{head}\n<hr/>\n"
            f"<blockquote>\n{'<br>'.join(rows)}\n</blockquote></blockquote>"
        )
    return "\n".join(blocks) or None


@text_module("history_header")
def _history_header(agent: "OpenAgent", ctx: RenderContext) -> str:
    """Header of a stored (older) exchange shown by the history pager."""
    pos = int(ctx.extra.get("pos") or 0)
    total = int(ctx.extra.get("total") or 0)
    return (
        f"<p><tg-pill style='primary'>🕘 {html.escape(agent.strings('history_title'))}</tg-pill> "
        f"<tg-pill style='primary'>{pos + 1}/{total}</tg-pill></p>"
    )


@text_module("history_nav")
def _history_nav(agent: "OpenAgent", ctx: RenderContext) -> str | None:
    """Back / Again / Next row that scrolls through the saved conversation history.

    Edge buttons are rendered only when there is somewhere to scroll, so the
    latest exchange shows just Back/Again and the oldest one Back/Again/Next.
    "Again" repeats the prompt of the currently displayed exchange.
    """
    state = ctx.state or {}
    history = state.get("history") or []
    total = len(history) // 2
    in_history = "pos" in ctx.extra
    pos = int(
        ctx.extra.get("pos", state.get("history_pos", total - 1 if total else 0))
    )
    if total:
        pos = max(0, min(pos, total - 1))
    prompt = ctx.extra.get("prompt") or state.get("prompt")
    if not total and not prompt:
        return None

    tags = state.setdefault("history_nav_tags", {})

    def tag(key: str, label: str, handler, data, style: str) -> str:
        # Built once per role: every render would otherwise register a new callback
        if key not in tags:
            tags[key] = agent.Button.rich.inline(
                agent.strings(label),
                handler,
                data=data,
                style=style,
                ttl=HISTORY_NAV_TTL,
                html_tag=True,
            )
        return tags[key]

    buttons: list[str] = []
    if total and pos > 0:
        buttons.append(
            tag("back", "btn_back", agent.on_history_scroll, (state, -1), "link")
        )
    if prompt:
        buttons.append(
            tag(
                "again_history" if in_history else "again",
                "btn_again",
                agent.on_repeat_request,
                (state, in_history),
                "primary",
            )
        )
    if total and pos < total - 1:
        buttons.append(
            tag("next", "btn_next", agent.on_history_scroll, (state, 1), "link")
        )
    if not buttons:
        return None
    return f"<tg-button-row align='center'>{''.join(buttons)}</tg-button-row>"


@text_module("diffs")
def _diffs(agent: "OpenAgent", ctx: RenderContext) -> str | None:
    """One collapsible diff per file changed by the file tools."""
    blocks = []
    for name, change in ((ctx.state or {}).get("changes") or {}).items():
        diff = make_diff(name, change["old"], change["new"])
        if diff:
            blocks.append(
                f"<details><summary>Diff: {html.escape(name)} file</summary>"
                f"<pre><code class=\"language-diff\">{html.escape(diff)}</code></pre>"
                "</details>"
            )
    return "\n".join(blocks) or None


text_format("notice", "standalone_header", "body")
text_format("progress", "status_header", "prompt", "token_stats", "tool_bar", "tool_details", "thinking", "cancel_button")
text_format("result", "status_header", "prompt", "body", "error_detail", "token_stats", "tool_bar", "tool_details", "history_nav")
text_format("history", "history_header", "prompt", "body", "history_nav")
text_format("cancelled", "status_header", "cancelled_body")
text_format("error_standalone", "standalone_header", "body", "error_detail")
text_format("usage", "usage_header", "usage_body", "copy_button")
text_format("models", "models_header", "models_list", "models_nav")
