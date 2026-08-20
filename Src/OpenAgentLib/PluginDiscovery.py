# SPDX-License-Identifier: MIT
"""Static, non-executing admission checks for external v2 plugin source files."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .PluginSDK import LegacyPluginMigrationError


@dataclass(frozen=True)
class StaticPluginSource:
    """A v2 source file approved for later isolated-host execution."""

    path: Path
    digest: str


def _assigns_manifest(targets: tuple[ast.expr, ...] | list[ast.expr]) -> bool:
    """Return whether an assignment publishes the conventional manifest name."""

    for target in targets:
        if isinstance(target, ast.Name) and target.id in {
            "MANIFEST",
            "PLUGIN_MANIFEST",
        }:
            return True
        if isinstance(target, (ast.Tuple, ast.List)) and _assigns_manifest(target.elts):
            return True
    return False


def inspect_v2_plugin_source(path: Path) -> StaticPluginSource:
    """Validate a v2 declaration without importing or executing ``path``."""

    source_path = Path(path).resolve()
    try:
        source = source_path.read_bytes()
        tree = ast.parse(source, filename=str(source_path))
    except (OSError, SyntaxError) as exc:
        raise LegacyPluginMigrationError(
            f"plugin {source_path.name} is not a valid v2 manifest source"
        ) from exc

    manifest_import = False
    telegram_builder_import = False
    manifest_assignment = False
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "OpenAgentLib.PluginSDK":
            manifest_import = manifest_import or any(
                alias.name == "PluginManifest" for alias in node.names
            )
        if isinstance(node, ast.ImportFrom) and node.module == "_telegram_v2":
            telegram_builder_import = telegram_builder_import or any(
                alias.name == "build_plugin" for alias in node.names
            )
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            if _assigns_manifest(targets) and isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name):
                    manifest_assignment = manifest_assignment or (
                        node.value.func.id == "PluginManifest"
                        or (
                            telegram_builder_import
                            and node.value.func.id == "build_plugin"
                        )
                    )

    if not manifest_assignment or not (manifest_import or telegram_builder_import):
        raise LegacyPluginMigrationError(
            f"plugin {source_path.name} uses the removed legacy execution format; "
            "migrate it to a v2 PluginManifest"
        )
    return StaticPluginSource(source_path, sha256(source).hexdigest())


def discover_v2_plugin_sources(root: Path) -> dict[str, StaticPluginSource]:
    """Admit deterministic top-level v2 modules without importing them."""

    root = Path(root)
    sources: dict[str, StaticPluginSource] = {}
    for path in sorted(root.glob("*.py")):
        if path.name == "__init__.py" or path.name.startswith("_"):
            continue
        source = inspect_v2_plugin_source(path)
        if path.stem in sources:
            raise LegacyPluginMigrationError(f"duplicate v2 plugin source {path.stem}")
        sources[path.stem] = source
    return sources
