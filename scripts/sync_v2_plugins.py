from __future__ import annotations

import ast
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import stat
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "Src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from OpenAgentLib.PluginDiscovery import (  # noqa: E402
    discover_v2_plugin_sources,
    inspect_v2_plugin_source,
)

EXPECTED_PUBLIC_MODULES = frozenset(
    {
        "ast_grep",
        "chat",
        "contacts",
        "creation",
        "dialog",
        "eval",
        "file",
        "mcub",
        "message",
        "moderation",
        "profile",
        "task",
        "terminal",
        "web",
    }
)
REQUIRED_SUPPORT_FILES = frozenset(
    {"__init__.py", "_resource_v2.py", "_telegram_v2.py"}
)
MAX_PLUGIN_BYTES = 200_000


@dataclass(frozen=True)
class Snapshot:
    content: bytes | None
    mode: int | None


def _absolute_without_resolving(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _has_symlink_component(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def _validated_directory(value: str, *, destination: bool) -> Path:
    unresolved = _absolute_without_resolving(Path(value))
    label = "destination" if destination else "source"
    if _has_symlink_component(unresolved):
        raise ValueError(f"{label} directory must not contain symlinks: {unresolved}")
    path = unresolved.resolve()
    if not path.is_dir():
        raise ValueError(f"{label} directory does not exist: {path}")
    if destination and path == Path(path.anchor):
        raise ValueError(f"refusing dangerous destination directory: {path}")
    return path


def _validate_source(source: Path) -> tuple[Path, ...]:
    files = tuple(sorted(source.glob("*.py"), key=lambda path: path.name))
    if not files:
        raise ValueError(f"source contains no Python files: {source}")
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source plugin must be a regular file: {path}")
        content = path.read_bytes()
        if len(content) > MAX_PLUGIN_BYTES:
            raise ValueError(f"source plugin exceeds {MAX_PLUGIN_BYTES} bytes: {path}")
        try:
            ast.parse(content, filename=str(path))
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"invalid Python source {path}: {exc}") from exc

    names = {path.name for path in files}
    missing_support = REQUIRED_SUPPORT_FILES - names
    if missing_support:
        missing = ", ".join(sorted(missing_support))
        raise ValueError(f"canonical support files are missing: {missing}")

    admitted = discover_v2_plugin_sources(source)
    public_names = {path.stem for path in files if not path.name.startswith("_")}
    if public_names != EXPECTED_PUBLIC_MODULES:
        raise ValueError(
            "canonical public plugin set mismatch: "
            f"expected {sorted(EXPECTED_PUBLIC_MODULES)}, got {sorted(public_names)}"
        )
    if set(admitted) != EXPECTED_PUBLIC_MODULES:
        raise ValueError("not every canonical public plugin passed static admission")
    return files


def _atomic_write(destination: Path, content: bytes, mode: int) -> None:
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as target:
            descriptor = -1
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, destination)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _restore(destination: Path, snapshots: dict[str, Snapshot]) -> None:
    errors: list[str] = []
    invariant_errors: list[str] = []
    for name, snapshot in snapshots.items():
        target = destination / name
        try:
            if snapshot.content is None:
                target.unlink(missing_ok=True)
            else:
                if snapshot.mode is None:
                    invariant_errors.append(f"{name}: content snapshot has no mode")
                    continue
                _atomic_write(target, snapshot.content, snapshot.mode)
        except OSError as exc:
            errors.append(f"{name}: {exc}")
    if invariant_errors:
        details = "; ".join(invariant_errors + errors)
        raise RuntimeError(f"rollback snapshot invariant violated: {details}")
    if errors:
        raise OSError("rollback failed for " + "; ".join(errors))


def sync_plugins(source: Path, destination: Path) -> int:
    files = _validate_source(source)
    snapshots: dict[str, Snapshot] = {}
    for source_file in files:
        target = destination / source_file.name
        if target.is_symlink():
            raise ValueError(
                f"canonical destination target must not be a symlink: {target}"
            )
        if target.exists() and not target.is_file():
            raise ValueError(f"canonical destination target is not a file: {target}")
        snapshots[source_file.name] = Snapshot(
            target.read_bytes() if target.exists() else None,
            stat.S_IMODE(target.stat().st_mode) if target.exists() else None,
        )

    try:
        for source_file in files:
            mode = stat.S_IMODE(source_file.stat().st_mode)
            _atomic_write(
                destination / source_file.name, source_file.read_bytes(), mode
            )

        for module_name in sorted(EXPECTED_PUBLIC_MODULES):
            source_admission = inspect_v2_plugin_source(source / f"{module_name}.py")
            destination_admission = inspect_v2_plugin_source(
                destination / f"{module_name}.py"
            )
            if destination_admission.digest != source_admission.digest:
                raise ValueError(f"digest mismatch after sync: {module_name}.py")
        for source_file in files:
            target_digest = sha256(
                (destination / source_file.name).read_bytes()
            ).hexdigest()
            source_digest = sha256(source_file.read_bytes()).hexdigest()
            if target_digest != source_digest:
                raise ValueError(f"digest mismatch after sync: {source_file.name}")
    except (OSError, ValueError, RuntimeError):
        _restore(destination, snapshots)
        raise
    return len(files)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(
            "usage: sync_v2_plugins.py <source-plugin-dir> <installed-plugin-dir>",
            file=sys.stderr,
        )
        return 2
    try:
        source = _validated_directory(args[0], destination=False)
        destination = _validated_directory(args[1], destination=True)
        count = sync_plugins(source, destination)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"failed to sync v2 plugins: {exc}", file=sys.stderr)
        return 1
    print(f"synced {count} canonical plugin files to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
