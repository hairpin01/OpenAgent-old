"""Tool that runs Python/Telethon code inside the agent's Telegram context."""

from __future__ import annotations

import ast
import asyncio
import json
import re
import textwrap
from types import CodeType

import aiohttp

from ..core import AgentPlugin, Tool, ToolContext


def _wrap(code: str) -> tuple[CodeType, str]:
    """Compile user code, transparently supporting top-level ``await``.

    Returns ``(code_object, module_name)``. If the code's last statement is an
    expression, its value is captured into ``__result__`` (or becomes the
    wrapper function's return value).
    """
    # Try compiling the code as-is first
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None

    needs_async = tree is None or any(
        isinstance(node, (ast.Await, ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith))
        for node in ast.walk(tree)
    )

    if not needs_async:
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            value = tree.body.pop().value
            tree.body.append(
                ast.Assign(
                    targets=[ast.Name(id="__result__", ctx=ast.Store())],
                    value=value,
                )
            )
            ast.fix_missing_locations(tree)
        return compile(tree, "<python_telegram_exec>", "exec"), "exec"

    # Wrap into async function; also patch its return value
    indented = textwrap.indent(code.rstrip(), "    ")
    wrapper = f"async def __stack2__():\n{indented}\n"
    tree = ast.parse(wrapper)
    function = tree.body[0]
    if function.body and isinstance(function.body[-1], ast.Expr):
        value = function.body.pop().value
        function.body.append(
            ast.Assign(
                targets=[ast.Name(id="__result__", ctx=ast.Store())],
                value=value,
            )
        )
        function.body.append(ast.Return(value=ast.Name(id="__result__", ctx=ast.Load())))
        ast.fix_missing_locations(tree)
    return compile(tree, "<python_telegram_exec>", "exec"), "func"


async def handle(ctx: ToolContext, args: dict) -> str:
    code = str(args.get("code", "")).strip()
    if not code:
        raise ValueError("code is empty")
    event = ctx.event
    if event is None:
        raise ValueError("no Telegram event in this context")

    # Prefer the userbot client from the module/kernel over the inline bot
    client = ctx.client if ctx.client is not None else event.client
    chat_id = ctx.chat_id if ctx.chat_id is not None else getattr(event, "chat_id", None)
    me = await client.get_me()
    sender = await event.get_sender()

    printed: list[str] = []

    def _capture_print(*values: object, sep: str = " ") -> None:
        printed.append(sep.join(str(value) for value in values))

    environment: dict = {
        "client": client,
        "event": event,
        "chat_id": chat_id,
        "me": me,
        "sender": sender,
        "print": _capture_print,
        "asyncio": asyncio,
        "aiohttp": aiohttp,
        "json": json,
        "re": re,
        "__name__": "python_easy_setup",
    }

    code_object, mode = _wrap(code)

    if mode == "exec":
        exec(code_object, environment)
        result = environment.get("__result__")
    else:
        exec(code_object, environment)
        result = await environment["__stack2__"]()

    output = "\n".join(printed)
    if result is not None:
        if output:
            output += "\n"
        output += f"result: {result!r}"
    return output or "OK (no output)"


PLUGIN = AgentPlugin(
    name="python_telegram_exec",
    description=(
        "Execute Python code inside this Telegram session. Available globals: "
        "client (userbot Telethon client), event (current Telethon event), "
        "chat_id, me (current userbot user object), sender (current sender), "
        "print, asyncio, aiohttp, json, re. Top-level await is supported. The "
        "last expression, printed output, or 'OK' is returned. Example code: "
        "\"await client.send_message(event.chat_id, 'hi')\"."
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute."},
        },
        "required": ["code"],
        "additionalProperties": False,
    },
    handler=handle,
    timeout=120,
)

TOOL = PLUGIN
