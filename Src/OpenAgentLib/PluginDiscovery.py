# SPDX-License-Identifier: MIT
"""Static, non-executing admission checks for external v2 plugin source files."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .InstalledPluginRegistry import (
    InstalledPluginManifest,
    InstalledPluginRecord,
    InstalledPluginRegistry,
    InstalledPluginSource,
    InstalledPluginStatus,
    InstalledPluginTool,
)
from .PluginSDK import LegacyPluginMigrationError
from .ToolCompatibility import TOOL_COMPATIBILITY_MATRIX

_MAX_REPO_PLUGIN_BYTES = 200_000
_REPO_PLUGIN_OWNER = "hairpin01"


@dataclass(frozen=True)
class StaticPluginSource:
    """A v2 source file approved for later isolated-host execution."""

    path: Path
    digest: str


@dataclass(frozen=True)
class InstalledPluginAdmission:
    """One complete static installed-plugin declaration."""

    record: InstalledPluginRecord
    source_module: str


def _literal(node: ast.AST) -> object | None:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, MemoryError):
        pass
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"frozenset", "set", "tuple", "list"}
        and len(node.args) == 1
        and not node.keywords
    ):
        value = _literal(node.args[0])
        if isinstance(value, (list, tuple, set, frozenset)):
            return tuple(value)
    return None


def _assignment(node: ast.stmt) -> tuple[tuple[str, ...], ast.AST] | None:
    def names(target: ast.AST) -> tuple[str, ...]:
        if isinstance(target, ast.Name):
            return (target.id,)
        if isinstance(target, (ast.Tuple, ast.List)):
            return tuple(name for item in target.elts for name in names(item))
        return ()

    if isinstance(node, ast.Assign):
        assigned = tuple(name for target in node.targets for name in names(target))
        return (assigned, node.value) if assigned else None
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return ((node.target.id,), node.value)
    return None


def _call_keyword(call: ast.Call, name: str, position: int) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return call.args[position] if len(call.args) > position else None


def _literal_or_name(
    node: ast.AST | None, literals: Mapping[str, object]
) -> object | None:
    if isinstance(node, ast.Name):
        return literals.get(node.id)
    return _literal(node) if node is not None else None


def _human_plugin_name(plugin_id: str, stem: str) -> str:
    value = (plugin_id.rsplit(".", 1)[-1] or stem).replace("_", " ").replace("-", " ")
    return " ".join(part.capitalize() for part in value.split()) or "Plugin"


def _v2_tools(source_module: str, explicit: set[str]) -> list[str]:
    matrix_tools = {
        entry.canonical_id
        for entry in TOOL_COMPATIBILITY_MATRIX
        if entry.source_module == source_module
        and entry.migration_disposition != "reject"
    }
    return sorted(explicit or matrix_tools)


def parse_v2_plugin_metadata(
    source: str, source_name: str = "plugin.py"
) -> dict[str, object] | None:
    """Statically extract catalog data without importing or executing source."""

    if len(source.encode("utf-8")) > _MAX_REPO_PLUGIN_BYTES:
        return None
    try:
        tree = ast.parse(source, filename=source_name)
    except SyntaxError:
        return None
    stem = Path(source_name).stem
    literals: dict[str, object] = {}
    manifest_call: ast.Call | None = None
    builders: list[ast.Call] = []
    explicit_tools: set[str] = set()
    declared_schemas: dict[str, tuple[Mapping[str, object], Mapping[str, object]]] = {}
    for node in tree.body:
        assigned = _assignment(node)
        if assigned is not None:
            names, value = assigned
            literal = _literal(value)
            if literal is not None:
                literals.update({name: literal for name in names})
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                if value.func.id == "PluginManifest" and any(
                    name in {"MANIFEST", "PLUGIN_MANIFEST"} for name in names
                ):
                    manifest_call = value
                if value.func.id == "build_plugin":
                    builders.append(value)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "declaration"
            ):
                tool_id = _literal_or_name(
                    _call_keyword(call, "canonical_id", 0), literals
                )
                if isinstance(tool_id, str):
                    explicit_tools.add(tool_id)
                    input_schema = _literal_or_name(
                        _call_keyword(call, "input_schema", 1), literals
                    )
                    output_schema = _literal_or_name(
                        _call_keyword(call, "output_schema", 2), literals
                    )
                    if isinstance(input_schema, Mapping) and isinstance(
                        output_schema, Mapping
                    ):
                        declared_schemas[tool_id] = (input_schema, output_schema)
    if manifest_call is None and not builders:
        return None

    metadata = literals.get("CATALOG_METADATA")
    metadata = metadata if isinstance(metadata, dict) else {}
    plugin_id = version = api_version = entrypoint = ""
    capabilities: list[str] = []
    source_module = stem
    if manifest_call is not None:
        fields = {
            key: _literal(
                _call_keyword(manifest_call, key, index) or ast.Constant(None)
            )
            for index, key in enumerate(
                (
                    "plugin_id",
                    "version",
                    "api_version",
                    "entrypoint",
                    "tools",
                    "capabilities",
                )
            )
        }
        plugin_id = fields["plugin_id"] if isinstance(fields["plugin_id"], str) else ""
        version = fields["version"] if isinstance(fields["version"], str) else ""
        api_version = (
            fields["api_version"] if isinstance(fields["api_version"], str) else ""
        )
        entrypoint = (
            fields["entrypoint"] if isinstance(fields["entrypoint"], str) else ""
        )
        if isinstance(fields["capabilities"], (list, tuple, set, frozenset)):
            capabilities = sorted(
                str(value) for value in fields["capabilities"] if isinstance(value, str)
            )
        if entrypoint:
            source_module = entrypoint.split(".")[-2] if "." in entrypoint else stem
    if builders:
        module = _literal(builders[0].args[0]) if builders[0].args else None
        if not isinstance(module, str):
            return None
        source_module = module
        plugin_id = plugin_id or f"openagent.{module}"
        version = version or "2.0.0"
        api_version = api_version or "2"
        entrypoint = entrypoint or f"plugins.{stem}.HANDLERS"
        includes: set[str] = set()
        for call in builders:
            values = _literal(_call_keyword(call, "include", 1) or ast.Constant(None))
            if values is not None and not isinstance(
                values, (list, tuple, set, frozenset)
            ):
                return None
            if values is not None:
                includes.update(
                    str(value) for value in values if isinstance(value, str)
                )
        explicit_tools.update(includes)
    if not plugin_id or not version or not api_version or not entrypoint:
        return None
    tools = _v2_tools(source_module, explicit_tools)
    if not capabilities:
        capabilities = sorted(
            {
                entry.capability_class
                for entry in TOOL_COMPATIBILITY_MATRIX
                if entry.canonical_id in tools
                and entry.migration_disposition != "reject"
            }
        )
    description = str(
        metadata.get("description") or ast.get_docstring(tree) or ""
    ).strip()
    if not description:
        description = f"Version 2 OpenAgent plugin providing {source_module.replace('_', ' ')} tools."
    return {
        "name": str(metadata.get("name") or _human_plugin_name(plugin_id, stem)),
        "version": version,
        "api_version": api_version,
        "author": str(metadata.get("author") or _REPO_PLUGIN_OWNER),
        "description": description,
        "tools": tools,
        "tool_schemas": {
            tool_id: declared_schemas[tool_id]
            for tool_id in tools
            if tool_id in declared_schemas
        },
        "capabilities": capabilities,
        "permissions": capabilities,
        "requirements": [],
        "plugin_id": plugin_id,
        "entrypoint": entrypoint,
        "source_module": source_module,
    }


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


def inspect_installed_v2_plugin_source(
    path: Path, *, disabled: bool = False
) -> InstalledPluginAdmission:
    """Build an immutable installed record from source bytes only.

    The compatibility matrix is the reviewed declaration source for tool policy
    metadata.  The external module is never imported while rebuilding state.
    """

    source = inspect_v2_plugin_source(path)
    try:
        metadata = parse_v2_plugin_metadata(
            source.path.read_text(encoding="utf-8"), source.path.name
        )
    except OSError as exc:
        raise LegacyPluginMigrationError(
            f"plugin {source.path.name} cannot be read"
        ) from exc
    if metadata is None:
        raise LegacyPluginMigrationError(
            f"plugin {source.path.name} has no static v2 declaration"
        )
    source_module = str(metadata["source_module"])
    tool_ids = frozenset(str(value) for value in metadata["tools"])
    declared_schemas = metadata.get("tool_schemas")
    declared_schemas = declared_schemas if isinstance(declared_schemas, Mapping) else {}
    entries = tuple(
        entry
        for entry in TOOL_COMPATIBILITY_MATRIX
        if entry.source_module == source_module
        and entry.canonical_id in tool_ids
        and entry.migration_disposition != "reject"
    )
    if len(entries) != len(tool_ids):
        missing = sorted(tool_ids.difference(entry.canonical_id for entry in entries))
        raise LegacyPluginMigrationError(
            f"plugin {source.path.name} declares unreviewed tools: {', '.join(missing)}"
        )
    capabilities = frozenset(str(value) for value in metadata["capabilities"])
    tools = tuple(
        InstalledPluginTool(
            canonical_id=entry.canonical_id,
            aliases=entry.aliases,
            capabilities=frozenset({entry.capability_class}),
            input_schema=(
                declared_schemas[entry.canonical_id][0]
                if entry.canonical_id in declared_schemas
                else entry.v2_input_schema
            ),
            output_schema=(
                declared_schemas[entry.canonical_id][1]
                if entry.canonical_id in declared_schemas
                else entry.v2_output_schema
            ),
            confirmation=entry.confirmation_class,
            concurrency=entry.concurrency_class,
            idempotency=entry.idempotency_class,
            migration_disposition=entry.migration_disposition,
            description=entry.canonical_id,
        )
        for entry in entries
    )
    manifest_metadata: Mapping[str, object] = {
        "origin": "installed",
        "source_module": source_module,
        "author": metadata["author"],
        "description": metadata["description"],
        "permissions": metadata["permissions"],
        "requirements": metadata["requirements"],
    }
    record = InstalledPluginRecord(
        plugin_id=str(metadata["plugin_id"]),
        source=InstalledPluginSource(str(source.path), source.digest),
        manifest=InstalledPluginManifest(
            manifest_version="2",
            api_version=str(metadata["api_version"]),
            version=str(metadata["version"]),
            entrypoint=str(metadata["entrypoint"]),
            display_name=str(metadata["name"]),
            metadata=manifest_metadata,
            capabilities=capabilities,
            tools=tools,
        ),
        enabled=not disabled,
        status=(
            InstalledPluginStatus.DISABLED
            if disabled
            else InstalledPluginStatus.ADMITTED
        ),
    )
    return InstalledPluginAdmission(record=record, source_module=source_module)


def rebuild_installed_plugin_record(
    registry: InstalledPluginRegistry, admission: InstalledPluginAdmission
) -> InstalledPluginRecord:
    """Replace a changed static declaration with a fresh registry generation.

    Callers must bind the resulting installed record before activating it.  This
    helper deliberately has no relationship with an invoker or plugin module.
    """

    current = registry.find(admission.record.plugin_id)
    if current is None:
        admitted = registry.admit(admission.record)
        return (
            admitted
            if not admitted.enabled
            else registry.install(admitted, expected_generation=admitted.generation)
        )
    if (
        current.source == admission.record.source
        and current.manifest == admission.record.manifest
    ):
        return current
    if current.status is InstalledPluginStatus.ACTIVE:
        unloading = registry.begin_unload(
            current.plugin_id, expected_generation=current.generation
        )
        current = registry.complete_unload(
            unloading.plugin_id, expected_generation=unloading.generation
        )
    return registry.replace(admission.record, expected_generation=current.generation)


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
