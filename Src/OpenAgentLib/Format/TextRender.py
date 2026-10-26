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

    parts: list[str] = []
    for module_name in TEXT_FORMATS.get(format_name, ()):
        module = TEXT_MODULES.get(module_name)
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


@text_module("thinking")
def _thinking(agent: "OpenAgent", ctx: RenderContext) -> str:
    return (
        f"<blockquote>{CUSTOM_EMOJI['think']} "
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
text_format("progress", "status_header", "prompt", "thinking", "cancel_button")
text_format("result", "status_header", "prompt", "body")
text_format("cancelled", "status_header", "cancelled_body")
text_format("error_standalone", "standalone_header", "body")
text_format("usage", "usage_header", "usage_body", "copy_button")
text_format("models", "models_header", "models_list")
