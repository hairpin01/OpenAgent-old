from __future__ import annotations

import ast
import importlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_EVICTED_EXACT_ROOTS = {
    "OpenAgentLib",
    "Settings",
    "MCUBEvent",
    "openagent_system_tool_api",
}


def _snapshot_evictable_modules() -> dict[str, types.ModuleType]:
    return {
        name: module
        for name, module in sys.modules.items()
        if name in _EVICTED_EXACT_ROOTS or name.startswith("OpenAgentLib.")
    }


def _restore_evictable_modules(snapshot: dict[str, types.ModuleType]) -> None:
    for name in tuple(sys.modules):
        if name in _EVICTED_EXACT_ROOTS or name.startswith("OpenAgentLib."):
            sys.modules.pop(name, None)
    sys.modules.update(snapshot)


def _load_bootstrap(filename: Path, *, cubkit_artifact: bool = False):
    source = (ROOT / "Src" / "OpenAgentMain.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    body = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Import)
            and any(alias.name in {"importlib", "sys"} for alias in node.names)
        )
        or (isinstance(node, ast.ImportFrom) and node.module == "pathlib")
        or (
            isinstance(node, ast.FunctionDef)
            and node.name
            in {"_evict_stale_openagent_bundle_modules", "_runs_from_cubkit_artifact"}
        )
        or (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Name)
            and node.test.func.id == "_runs_from_cubkit_artifact"
        )
    ]
    namespace: dict[str, object] = {"__file__": str(filename)}
    if cubkit_artifact:
        namespace["__cubkit_module_id__"] = "openagent"
        namespace["__cubkit_bundle_sha256__"] = "digest"
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(filename), "exec"),
        namespace,
    )
    return namespace


def _load_evictor():
    return _load_bootstrap(ROOT / "Src" / "OpenAgentMain.py")[
        "_evict_stale_openagent_bundle_modules"
    ]


def test_evictor_removes_only_artifact_owned_modules(monkeypatch) -> None:
    evict = _load_evictor()
    snapshot = _snapshot_evictable_modules()
    stale_names = (
        "OpenAgentLib",
        "OpenAgentLib.OpenAgentMixins",
        "OpenAgentLib.ResponseAgent",
        "Settings",
        "MCUBEvent",
        "openagent_system_tool_api",
    )
    preserved_names = ("OpenAgentSomething", "OpenAgentLibSomething", "cubkit", "core")
    for name in stale_names + preserved_names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    invalidations: list[None] = []
    monkeypatch.setattr(
        importlib, "invalidate_caches", lambda: invalidations.append(None)
    )

    try:
        evict()

        assert not any(name in sys.modules for name in stale_names)
        assert all(name in sys.modules for name in preserved_names)
        assert invalidations == [None]
    finally:
        _restore_evictable_modules(snapshot)


def test_evictor_is_idempotent(monkeypatch) -> None:
    evict = _load_evictor()
    snapshot = _snapshot_evictable_modules()
    monkeypatch.setitem(
        sys.modules, "OpenAgentLib.OpenAgentMixins", types.ModuleType("stale")
    )

    invalidations: list[None] = []
    monkeypatch.setattr(
        importlib, "invalidate_caches", lambda: invalidations.append(None)
    )

    try:
        evict()
        evict()

        assert "OpenAgentLib.OpenAgentMixins" not in sys.modules
        assert invalidations == [None, None]
    finally:
        _restore_evictable_modules(snapshot)


def test_source_bootstrap_preserves_existing_openagentlib_module_identities(
    monkeypatch,
) -> None:
    package = types.ModuleType("OpenAgentLib")
    mixins = types.ModuleType("OpenAgentLib.OpenAgentMixins")
    monkeypatch.setitem(sys.modules, "OpenAgentLib", package)
    monkeypatch.setitem(sys.modules, "OpenAgentLib.OpenAgentMixins", mixins)
    invalidations: list[None] = []
    monkeypatch.setattr(
        importlib, "invalidate_caches", lambda: invalidations.append(None)
    )

    _load_bootstrap(ROOT / "Src" / "OpenAgentMain.py")

    assert sys.modules["OpenAgentLib"] is package
    assert sys.modules["OpenAgentLib.OpenAgentMixins"] is mixins
    assert invalidations == []


def test_artifact_bootstrap_evicts_before_openagent_imports(
    monkeypatch, tmp_path: Path
) -> None:
    stale = types.ModuleType("OpenAgentLib.OpenAgentMixins")
    snapshot = _snapshot_evictable_modules()
    monkeypatch.setitem(sys.modules, "OpenAgentLib.OpenAgentMixins", stale)
    invalidations: list[None] = []
    monkeypatch.setattr(
        importlib, "invalidate_caches", lambda: invalidations.append(None)
    )

    try:
        _load_bootstrap(tmp_path / "OpenAgent.py")

        assert "OpenAgentLib.OpenAgentMixins" not in sys.modules
        assert invalidations == [None]
    finally:
        _restore_evictable_modules(snapshot)

    tree = ast.parse((ROOT / "Src" / "OpenAgentMain.py").read_text(encoding="utf-8"))
    guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Name)
        and node.test.func.id == "_runs_from_cubkit_artifact"
    )
    cubkit_import = next(
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "cubkit"
    )
    assert guard.lineno < cubkit_import.lineno


def test_cubkit_globals_mark_openagentmain_filename_as_artifact(
    monkeypatch, tmp_path: Path
) -> None:
    stale = types.ModuleType("OpenAgentLib.OpenAgentMixins")
    snapshot = _snapshot_evictable_modules()
    monkeypatch.setitem(sys.modules, "OpenAgentLib.OpenAgentMixins", stale)

    try:
        _load_bootstrap(
            tmp_path / "OpenAgentMain.py",
            cubkit_artifact=True,
        )

        assert "OpenAgentLib.OpenAgentMixins" not in sys.modules
    finally:
        _restore_evictable_modules(snapshot)
