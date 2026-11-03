"""Tool that runs Python code in a subprocess (no Telegram context)."""

from __future__ import annotations

import asyncio
import json
import sys

from ..core import AgentPlugin, Tool, ToolContext


async def handle(_ctx: ToolContext, args: dict) -> str:
    code = str(args.get("code", "")).strip()
    if not code:
        raise ValueError("code is empty")

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-c",
        code,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await proc.communicate()
    except (asyncio.CancelledError, TimeoutError):
        if proc.returncode is None:
            proc.kill()
        raise

    text = out.decode("utf-8", errors="replace").strip()
    error = err.decode("utf-8", errors="replace").strip()
    parts = []
    if text:
        parts.append(text)
    if error:
        parts.append(f"[stderr]\n{error}")
    if proc.returncode != 0:
        parts.append(f"[exit code: {proc.returncode}]")
    return "\n".join(parts) or "OK (no output)"


PLUGIN = AgentPlugin(
    name="run_python",
    description=(
        "Run Python code in a separate subprocess (isolated, no Telegram "
        "client available) and return its stdout/stderr. Use for pure "
        "computation, data processing, and quick scripts. Top-level await "
        "is NOT supported — wrap async code in asyncio.run()."
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
    timeout=60,
)

TOOL = PLUGIN
