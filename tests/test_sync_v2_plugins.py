from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat

import pytest

from conftest import load_source_module

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "repo-MCUB-fork" / "OpenAgent" / "plugins"
sync = load_source_module("sync_v2_plugins", "scripts/sync_v2_plugins.py")


@pytest.fixture
def source(tmp_path: Path) -> Path:
    copied = tmp_path / "canonical"
    shutil.copytree(CANONICAL, copied)
    return copied


def _snapshot(directory: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.name: (path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
        for path in directory.iterdir()
        if path.is_file()
    }


def test_syncs_complete_bundle_and_preserves_runtime_extras(
    source: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "installed"
    destination.mkdir()
    legacy = destination / "ast_grep.py"
    legacy.write_bytes(b"legacy implementation\n")
    extra = destination / "extra.py"
    extra.write_bytes(b"user plugin\n")
    disabled = destination / "disabled_plugins.json"
    disabled.write_bytes(b'["openagent.ast_grep"]\n')
    cache = destination / "__pycache__"
    cache.mkdir()
    (cache / "cached.pyc").write_bytes(b"cache")

    count = sync.sync_plugins(source, destination)

    assert count == 17
    assert {
        path.stem
        for path in destination.glob("*.py")
        if not path.name.startswith("_") and path.name != "extra.py"
    } == sync.EXPECTED_PUBLIC_MODULES
    for source_file in source.glob("*.py"):
        assert (destination / source_file.name).read_bytes() == source_file.read_bytes()
    assert legacy.read_bytes() == (source / "ast_grep.py").read_bytes()
    assert extra.read_bytes() == b"user plugin\n"
    assert disabled.read_bytes() == b'["openagent.ast_grep"]\n'
    assert (cache / "cached.pyc").read_bytes() == b"cache"


@pytest.mark.parametrize("failure", ["syntax", "admission"])
def test_source_validation_failure_does_not_change_destination(
    source: Path, tmp_path: Path, failure: str
) -> None:
    destination = tmp_path / "installed"
    destination.mkdir()
    existing = destination / "ast_grep.py"
    existing.write_bytes(b"legacy\n")
    existing.chmod(0o640)
    extra = destination / "extra.py"
    extra.write_bytes(b"keep\n")
    before = _snapshot(destination)
    if failure == "syntax":
        (source / "_resource_v2.py").write_text("def broken(:\n")
    else:
        (source / "ast_grep.py").write_text("VALUE = 1\n")

    with pytest.raises(ValueError):
        sync.sync_plugins(source, destination)

    assert _snapshot(destination) == before


def test_mid_write_failure_rolls_back_bytes_modes_and_new_files(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "installed"
    destination.mkdir()
    first_names = [path.name for path in sorted(source.glob("*.py"))[:2]]
    for index, name in enumerate(first_names):
        target = destination / name
        target.write_bytes(f"legacy {name}\n".encode())
        target.chmod(0o600 + index * 0o40)
    extra = destination / "extra.py"
    extra.write_bytes(b"keep\n")
    before = _snapshot(destination)
    original_atomic_write = sync._atomic_write
    calls = 0

    def fail_third_write(target: Path, content: bytes, mode: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected write failure")
        original_atomic_write(target, content, mode)

    monkeypatch.setattr(sync, "_atomic_write", fail_third_write)

    with pytest.raises(OSError, match="injected write failure"):
        sync.sync_plugins(source, destination)

    assert _snapshot(destination) == before
    assert set(path.name for path in destination.iterdir()) == set(before)


def test_restore_reports_invalid_snapshot_after_restoring_valid_targets(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "invalid.py"
    invalid.write_bytes(b"current invalid\n")
    valid = tmp_path / "valid.py"
    valid.write_bytes(b"current valid\n")

    with pytest.raises(RuntimeError, match="content snapshot has no mode"):
        sync._restore(
            tmp_path,
            {
                "invalid.py": sync.Snapshot(b"original invalid\n", None),
                "valid.py": sync.Snapshot(b"original valid\n", 0o640),
            },
        )

    assert invalid.read_bytes() == b"current invalid\n"
    assert valid.read_bytes() == b"original valid\n"
    assert stat.S_IMODE(valid.stat().st_mode) == 0o640


def test_cli_rejects_symlink_destination(source: Path, tmp_path: Path) -> None:
    real_destination = tmp_path / "installed"
    real_destination.mkdir()
    linked_destination = tmp_path / "linked"
    os.symlink(real_destination, linked_destination)

    assert sync.main([str(source), str(linked_destination)]) == 1
    assert not tuple(real_destination.iterdir())
