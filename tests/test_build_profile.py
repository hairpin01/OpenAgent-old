from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from scripts.configure_build_profile import BuildProfileError, configure_profile

ROOT = Path(__file__).resolve().parents[1]


def _project(tmp_path: Path, marker: str = "DEBUG = False") -> Path:
    source = tmp_path / "Src"
    source.mkdir()
    (source / "Settings.py").write_text(
        f"# generated test source\n{marker}\nVALUE = 1\n", encoding="utf-8"
    )
    return tmp_path


def test_profile_marker_embeds_debug_then_restores_release(tmp_path: Path) -> None:
    project = _project(tmp_path)
    settings = project / "Src" / "Settings.py"

    configure_profile(project, "debug")
    assert "DEBUG = True" in settings.read_text(encoding="utf-8")

    configure_profile(project, "release")
    assert "DEBUG = False" in settings.read_text(encoding="utf-8")


def test_profile_marker_rejects_ambiguous_source(tmp_path: Path) -> None:
    project = _project(tmp_path, "DEBUG = False\nDEBUG = True")

    with pytest.raises(BuildProfileError, match="exactly one DEBUG"):
        configure_profile(project, "debug")


def test_manifest_wraps_debug_build_and_forces_release_false() -> None:
    manifest = tomllib.loads((ROOT / "cubkit.toml").read_text(encoding="utf-8"))
    debug = manifest["hooks"]["debug"]
    release = manifest["hooks"]["release"]

    assert debug["pre_build"][-1] == "debug"
    assert debug["post_build"][0][-1] == "release"
    assert release["pre_build"][-1] == "release"


def test_checked_in_marker_is_release_safe() -> None:
    settings = (ROOT / "Src" / "Settings.py").read_text(encoding="utf-8")

    assert "DEBUG = False" in settings
    assert "DEBUG = True" not in settings
