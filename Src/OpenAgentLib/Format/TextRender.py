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

from core.lib.types import Message, InlineMessage

from Settings import CUSTOM_EMOJI, MAX_REPLY_CHARS

if TYPE_CHECKING:
    from OpenAgentMain import OpenAgent

from .RichBoby import render_body
from ..Tools.workspace import make_diff
from ...Settings import TOOL_LOG_JSON_CHARS

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
    badges.append(f"<tg-pill style='primary'>🧠 {ctx_count}/{agent.config.get('context_size')}</tg-pill>")

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
    current = str(agent.config["model"])
    return (
        f"<p>{CUSTOM_EMOJI['bals']} "
        f"<b>{html.escape(agent.strings('models_title'))}</b> "
        f"<tg-pill style='success'>{html.escape(current)}</tg-pill></p>"
    )


@text_module("models_list")
def _models_list(agent: "OpenAgent", ctx: RenderContext) -> str:
    models = ctx.extra.get("models") or []
    if not models:
        return f"<blockquote>{html.escape(agent.strings('models_empty'))}</blockquote>"

    current = str(agent.config["model"])
    rows = []
    for model in models:
        selected = model == current
        style = "success" if selected else "primary"
        marker = CUSTOM_EMOJI["ok"] if selected else "•"
        button = agent.Button.rich.inline(
            model,
            agent.on_change_model,
            data=model,
            style=style,
            html_tag=True,
        )
        rows.append(f"{marker} {button}")
    return "\n".join(["<blockquote expandable>", "<br>".join(rows), "</blockquote>"])

@text_module("line_break")
def _line_break(agent: "OpenAgent", ctx: RenderContext) -> str:
    return "<hr/>\n"

@text_module("prompt")
def _prompt(agent: "OpenAgent", ctx: RenderContext) -> str:
    return f"""<details><summary>{agent.strings('prompt')}</summary>⬅️ {html.escape(agent._clip(ctx.state["prompt"]))}</details>"""

@text_module("tool_log")
def _tool_log(agent: "OpenAgent", ctx: RenderContext) -> str | None:
    """Collapsible list of the tool calls made during the request."""
    rows = []
    for item in (ctx.state or {}).get("tool_log") or []:
        data = str(item["args"]).replace("\n", " ")
        if len(data) > TOOL_LOG_JSON_CHARS:
            data = data[:TOOL_LOG_JSON_CHARS] + "…"
        rows.append(f"<b>{html.escape(item['name'])}:</b> {html.escape(data)}")
    if not rows:
        return None
    title = html.escape(agent.strings("tool_calls_title"))
    return f"<details><summary>{title}</summary>{'<br>'.join(rows)}</details>"


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


@text_module("ask")
def _ask(agent: "OpenAgent", ctx: RenderContext) -> str:

    async def _input_ask(call: InlineMessage, args: str, data: Message | None = None):
        if not data:
            return

        data.guest_args = args
        data.reply_rich = data.edit_rich
        data.reply = data.edit
        try:
            await agent.guest_cmd_agent(data)
        except Exception as e:
            await agent.kernel.handler_error(
                e,
                message="guest_cmd_agent call failed"
            )

    input_button = agent.Button.rich.input(
        "Ask",
        _input_ask,
        style='success',
        data=ctx.extra['message'],
        html_tag=True,
    )
    row = "<tg-button-row>{button}</tg-button-row>".format(button=input_button)
    return row


text_format("notice", "standalone_header", "body")
text_format("progress", "status_header", "prompt", "tool_log", "thinking", "cancel_button")
text_format("result", "status_header", "prompt", "tool_log", "body")
text_format("cancelled", "status_header", "cancelled_body")
text_format("error_standalone", "standalone_header", "body")
text_format("usage", "usage_header", "usage_body", "copy_button")
text_format("models", "models_header", "models_list")
