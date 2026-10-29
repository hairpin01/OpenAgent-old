"""Shared helpers for the file tools: paths, change tracking, diffs."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import ToolContext


def resolve_path(path: str) -> tuple[Path, str]:
    """Return the absolute path and its display name (relative to cwd if inside)."""
    target = Path(str(path).strip()).expanduser().resolve()
    try:
        return target, target.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return target, target.as_posix()


def read_text(path: Path, name: str) -> str:
    if not path.is_file():
        raise ValueError(f"file not found: {name}")
    return path.read_text(encoding="utf-8")


def write_text(path: Path, name: str, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def record_change(ctx: ToolContext, name: str, old: str | None, new: str) -> None:
    """Remember the first original and the latest content of a file."""
    changes = ctx.state.setdefault("changes", {})
    if name in changes:
        changes[name]["new"] = new
    else:
        changes[name] = {"old": old, "new": new}
    ctx.add_module("result", "diffs", after="body")


def make_diff(name: str, old: str | None, new: str) -> str:
    lines = difflib.unified_diff(
        (old or "").splitlines(),
        new.splitlines(),
        fromfile="/dev/null" if old is None else f"a/{name}",
        tofile=f"b/{name}",
        lineterm="",
    )
    return "\n".join(lines)
