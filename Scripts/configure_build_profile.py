from __future__ import annotations

import contextlib
import os
from pathlib import Path
import re
import sys
import tempfile

_MARKER_RE = re.compile(r"^DEBUG = (?:True|False)$", re.MULTILINE)


class BuildProfileError(RuntimeError):
    pass


def _project_dir() -> Path:
    configured = os.environ.get("CUBKIT_PROJECT_DIR")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[1]


def configure_profile(project_dir: Path, profile: str) -> Path:
    normalized = str(profile or "").strip().lower()
    if normalized not in {"debug", "release"}:
        raise BuildProfileError("profile must be debug or release")
    path = project_dir.resolve() / "Src" / "Settings.py"
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BuildProfileError(f"cannot read build marker: {path}") from exc
    matches = tuple(_MARKER_RE.finditer(source))
    if len(matches) != 1:
        raise BuildProfileError(
            f"expected exactly one DEBUG build marker in {path}, found {len(matches)}"
        )
    enabled = normalized == "debug"
    updated = _MARKER_RE.sub(f"DEBUG = {enabled}", source, count=1)
    if updated == source:
        return path
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise BuildProfileError(f"cannot update build marker: {path}") from exc
    finally:
        if temporary is not None:
            with contextlib.suppress(OSError):
                Path(temporary).unlink()
    return path


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if len(values) != 1:
        raise BuildProfileError("usage: configure_build_profile.py debug|release")
    configure_profile(_project_dir(), values[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
