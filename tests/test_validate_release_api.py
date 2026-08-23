from __future__ import annotations

import importlib.util
import hashlib
import re
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_release_api", ROOT / "scripts" / "validate_release_api.py"
)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def _write_fixture(tmp_path: Path, source: str) -> tuple[Path, Path]:
    core = tmp_path / "mcub"
    loader = core / "core" / "lib" / "loader"
    loader.mkdir(parents=True)
    for package in (core / "core", core / "core" / "lib", loader):
        (package / "__init__.py").write_text("")
    (loader / "module_base.py").write_text("class ModuleBase: pass\n")
    artifact = tmp_path / "artifact.py"
    artifact.write_text(source)
    return artifact, core


def _write_openagent_mixin(core: Path) -> None:
    package = core / "OpenAgentLib"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "OpenAgentMixins.py").write_text("class Marker: pass\n")


VALID = """
import importlib
import sys
for name in tuple(sys.modules):
    if name in {"OpenAgentLib", "Settings", "MCUBEvent", "openagent_system_tool_api"} or name.startswith("OpenAgentLib."):
        sys.modules.pop(name, None)
importlib.invalidate_caches()
from core.lib.loader.module_base import ModuleBase
class OpenAgent(ModuleBase):
    def _registry_catalog_snapshot(self): pass
    def _final_buttons(self, *, agent_log=None): pass
"""

NON_EVICTING_VALID = """
from core.lib.loader.module_base import ModuleBase
class OpenAgent(ModuleBase):
    def _registry_catalog_snapshot(self): pass
    def _final_buttons(self, *, agent_log=None): pass
"""


def _parse_openagent_main_sha256(source_map: str) -> str | None:
    source_map_header = source_map.split("# CubKit source map:", 1)
    if len(source_map_header) != 2:
        return None
    match = re.search(
        r"^#\s+-\s+.*\bOpenAgentMain\.py\b.*\bsha256:\s*([0-9a-fA-F]{64})\b",
        source_map_header[1],
        re.MULTILINE,
    )
    return match.group(1).lower() if match else None


def _current_source_sha256() -> str:
    return hashlib.sha256((ROOT / "Src" / "OpenAgentMain.py").read_bytes()).hexdigest()


def test_valid_api_and_import_cleanup(tmp_path: Path) -> None:
    artifact, core = _write_fixture(tmp_path, VALID)
    marker = "_openagent_release_"
    before_path = list(sys.path)
    validator.validate_release_api(artifact, core)
    assert sys.path == before_path
    assert not any(name.startswith(marker) for name in sys.modules)


def test_rejects_artifact_that_reuses_stale_dependencies(tmp_path: Path) -> None:
    artifact, core = _write_fixture(tmp_path, NON_EVICTING_VALID)

    with pytest.raises(validator.ReleaseAPIError, match="reused stale dependency modules"):
        validator.validate_release_api(artifact, core)


def test_accepts_artifact_that_evicts_and_imports_stale_dependencies(tmp_path: Path) -> None:
    artifact, core = _write_fixture(
        tmp_path,
        """
# - OpenAgentLib/OpenAgentMixins.py -> OpenAgentMixins.py:1
import importlib
import sys
for name in tuple(sys.modules):
    if name in {"OpenAgentLib", "Settings", "MCUBEvent", "openagent_system_tool_api"} or name.startswith("OpenAgentLib."):
        sys.modules.pop(name, None)
importlib.invalidate_caches()
from OpenAgentLib.OpenAgentMixins import Marker
from core.lib.loader.module_base import ModuleBase
class OpenAgent(Marker, ModuleBase):
    def _registry_catalog_snapshot(self): pass
    def _final_buttons(self, *, agent_log=None): pass
""",
    )
    _write_openagent_mixin(core)

    validator.validate_release_api(artifact, core)


def test_import_failure_restores_stale_modules_and_path(tmp_path: Path, monkeypatch) -> None:
    artifact, core = _write_fixture(tmp_path, "raise RuntimeError('boom')\n")
    previous_path = list(sys.path)
    previous_modules = {
        name: types.ModuleType(f"existing_{name}")
        for name in validator._ARTIFACT_MODULE_ROOTS
    }
    for name, module in previous_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    with pytest.raises(validator.ReleaseAPIError, match="import failed"):
        validator.validate_release_api(artifact, core)
    assert sys.path == previous_path
    assert {name: sys.modules[name] for name in previous_modules} == previous_modules


@pytest.mark.parametrize(
    "source, message",
    [
        ("class Other: pass\n", "OpenAgent"),
        ("class OpenAgent: pass\n", "MRO"),
        (
            """
from core.lib.loader.module_base import ModuleBase
class OpenAgent(ModuleBase):
    def _final_buttons(self, *, agent_log=None): pass
""",
            "_registry_catalog_snapshot",
        ),
        (
            """
from core.lib.loader.module_base import ModuleBase
class OpenAgent(ModuleBase):
    def _registry_catalog_snapshot(self): pass
    def _final_buttons(self, value): pass
""",
            "agent_log",
        ),
        ("raise RuntimeError('boom')\n", "import failed"),
    ],
)
def test_invalid_api(tmp_path: Path, source: str, message: str) -> None:
    artifact, core = _write_fixture(
        tmp_path,
        source if source.startswith("raise ") else VALID.split("from core", 1)[0] + source,
    )
    with pytest.raises(validator.ReleaseAPIError, match=message):
        validator.validate_release_api(artifact, core)
    assert str(core) not in sys.path
    assert not any(name.startswith("_openagent_release_") for name in sys.modules)


def test_rejects_missing_and_symlink_paths(tmp_path: Path) -> None:
    artifact, core = _write_fixture(tmp_path, VALID)
    with pytest.raises(validator.ReleaseAPIError, match="does not exist"):
        validator.validate_release_api(tmp_path / "missing.py", core)
    symlink = tmp_path / "artifact-link.py"
    symlink.symlink_to(artifact)
    with pytest.raises(validator.ReleaseAPIError, match="symlink"):
        validator.validate_release_api(symlink, core)
    with pytest.raises(validator.ReleaseAPIError, match="unsafe"):
        validator.validate_release_api(artifact, Path(artifact.anchor))


def test_parse_matching_openagent_main_source_map_header() -> None:
    source_hash = _current_source_sha256()
    header = f"# CubKit source map:\n# - generated line 1 -> OpenAgentMain.py:1 (lines: 1, sha256: {source_hash})\n"

    assert _parse_openagent_main_sha256(header) == source_hash


def test_parse_missing_openagent_main_source_map_header() -> None:
    assert _parse_openagent_main_sha256("# CubKit source map:\n# - generated line 1 -> OpenAgentMain.py:1\n") is None


def test_parse_mismatched_openagent_main_source_map_header() -> None:
    source_hash = _current_source_sha256()
    mismatched_hash = "0" * 64
    assert mismatched_hash != source_hash
    header = f"# CubKit source map:\n# - generated line 1 -> OpenAgentMain.py:1 (lines: 1, sha256: {mismatched_hash})\n"

    assert _parse_openagent_main_sha256(header) != source_hash


def test_current_artifact_when_present_and_current() -> None:
    artifact = ROOT / "dist" / "OpenAgent-MCUB-repo.py"
    core = ROOT.parent / "MCUB-fork"
    if not artifact.is_file():
        pytest.skip(f"stale release artifact absent: {artifact}")
    embedded_hash = _parse_openagent_main_sha256(artifact.read_text(encoding="utf-8"))
    if embedded_hash is None:
        pytest.skip("stale release artifact has no OpenAgentMain.py SHA256 source-map entry")
    current_hash = _current_source_sha256()
    if embedded_hash != current_hash:
        pytest.skip(
            "stale release artifact has outdated OpenAgentMain.py: "
            f"embedded SHA256 {embedded_hash}, current SHA256 {current_hash}"
        )
    if not core.is_dir():
        pytest.skip(f"MCUB core unavailable for current release artifact validation: {core}")
    validator.validate_release_api(artifact, core)
