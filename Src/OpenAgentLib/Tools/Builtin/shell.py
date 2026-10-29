from __future__ import annotations

import asyncio
import os
import signal

from ..core import Tool, ToolContext


async def handle(_ctx: ToolContext, args: dict) -> str:
    command = str(args.get("command", "")).strip()
    if not command:
        raise ValueError("command is empty")

    proc = await asyncio.create_subprocess_shell(
        command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=args.get("cwd") or None,
        start_new_session=True,
    )
    try:
        output, _ = await proc.communicate()
    finally:
        # Timeout or cancel: kill the whole process group, not only the shell
        if proc.returncode is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()

    text = output.decode("utf-8", errors="replace")
    return f"exit code: {proc.returncode}\n{text}"


TOOL = Tool(
    name="shell",
    description=(
        "Run a shell command and return its exit code and combined "
        "stdout/stderr. Commands that run too long are killed."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run."},
            "cwd": {"type": "string", "description": "Working directory. Optional."},
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    handler=handle,
    timeout=120,
)
