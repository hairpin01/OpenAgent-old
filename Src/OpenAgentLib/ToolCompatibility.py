# SPDX-License-Identifier: MIT
"""Read-only compatibility inventory for the legacy tool surface.

The exported matrix is a checked-in snapshot, so importing this module never reads
tool source files.  AST declaration parsing remains available for explicit drift
comparisons without importing plugins with runtime-only dependencies.
"""
from __future__ import annotations

from ast import Assign, Call, ClassDef, Name, literal_eval, parse
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYSTEM_TOOLS_ROOT = PROJECT_ROOT / "Src" / "OpenAgentLib" / "SystemPlugins"
SIBLING_PLUGINS_ROOT = PROJECT_ROOT.parent / "repo-MCUB-fork" / "OpenAgent" / "plugins"

# Legacy aliases whose handlers have no matching tool_registry declaration must be
# deliberately remapped or rejected here.  They may never inherit declaration-order
# ownership from an unrelated canonical tool.
_EXPLICIT_PLUGIN_ALIAS_REMAPS = MappingProxyType({})
_REJECTED_LEGACY_ALIASES = MappingProxyType(
    {
        "chat.search": "cmd_search has no tool_registry canonical ID",
        "eval.python.telegram": "unsandboxed legacy eval alias is not migration-safe",
    }
)


class CompatibilityInventoryError(ValueError):
    """The declared legacy surface cannot be represented unambiguously."""


@dataclass(frozen=True)
class ToolCompatibility:
    """One canonical legacy tool and its explicit v2 migration contract."""

    canonical_id: str
    aliases: tuple[str, ...]
    source_family: str
    source_module: str
    legacy_arguments: Mapping[str, str]
    v2_input_schema: Mapping[str, str]
    v2_output_schema: Mapping[str, str]
    confirmation_class: str
    capability_class: str
    concurrency_class: str
    idempotency_class: str
    migration_disposition: str


def _normalized(value: object) -> str:
    return str(value or "").strip().lower()


def _readonly_mapping(values: Mapping[str, object]) -> Mapping[str, str]:
    return MappingProxyType({str(key): str(value) for key, value in values.items()})


def _literal_assignment(nodes: Iterable[Any], name: str, default: Any = None) -> Any:
    for node in nodes:
        if isinstance(node, Assign) and any(
            isinstance(target, Name) and target.id == name for target in node.targets
        ):
            return literal_eval(node.value)
    return default


def _system_declarations(root: Path) -> tuple[dict[str, Any], ...]:
    declarations: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.py")):
        if path.name in {"__init__.py", "base.py"}:
            continue
        tree = parse(path.read_text(encoding="utf-8"), filename=str(path))
        descriptor = next(
            (
                node.value
                for node in tree.body
                if isinstance(node, Assign)
                and any(
                    isinstance(target, Name) and target.id == "SYSTEM_TOOL"
                    for target in node.targets
                )
            ),
            None,
        )
        if descriptor is None:
            continue
        if not isinstance(descriptor, Call):
            raise CompatibilityInventoryError(f"invalid system descriptor in {path}")
        keywords = {keyword.arg: literal_eval(keyword.value) for keyword in descriptor.keywords}
        tool_class = _normalized(keywords.get("tool_class"))
        name = _normalized(keywords.get("name"))
        if not tool_class or not name:
            raise CompatibilityInventoryError(f"missing canonical ID in {path}")
        declarations.append(
            {
                "canonical_id": f"{tool_class}.{name}",
                "aliases": tuple(_normalized(alias) for alias in keywords.get("aliases", ())),
                "docs": keywords.get("docs", {}),
                "dangerous": bool(keywords.get("dangerous", False)),
                "parallel_safe": bool(keywords.get("parallel_safe", False)),
                "input_schema": keywords.get("input_schema", {}),
                "output_schema": keywords.get("output_schema", {}),
                "source_module": path.relative_to(root).with_suffix("").as_posix().replace("/", "."),
            }
        )
    return tuple(declarations)


def _plugin_declarations(root: Path) -> tuple[dict[str, Any], ...]:
    declarations: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.py")):
        if path.name == "__init__.py" or path.stem.startswith("_"):
            continue
        tree = parse(path.read_text(encoding="utf-8"), filename=str(path))
        classes = [node for node in tree.body if isinstance(node, ClassDef)]
        if len(classes) != 1:
            # v2 plugins expose a pure manifest rather than an executable
            # legacy class.  Resolve its frozen inventory by module name only;
            # importing it would violate the compatibility parser boundary.
            has_v2_manifest = any(
                isinstance(node, Assign)
                and isinstance(node.value, Call)
                and isinstance(node.value.func, Name)
                and node.value.func.id in {"build_plugin", "PluginManifest"}
                for node in tree.body
            )
            if not has_v2_manifest:
                raise CompatibilityInventoryError(f"expected one plugin class in {path}")
            entries = tuple(entry for entry in TOOL_COMPATIBILITY_MATRIX if entry.source_module == path.stem)
            if not entries:
                raise CompatibilityInventoryError(f"v2 plugin {path} has no frozen compatibility entries")
            tool_map = {
                name: entry.canonical_id
                for entry in entries
                for name in (entry.canonical_id, *entry.aliases)
            }
            declarations.append({
                "canonical_ids": tuple(entry.canonical_id for entry in entries),
                "tool_map": tool_map,
                "docs": {},
                "dangerous": frozenset(
                    entry.canonical_id for entry in entries if entry.confirmation_class == "required"
                ),
                "source_module": path.stem,
                "v2_manifest": True,
            })
            continue
        body = classes[0].body
        registry = _literal_assignment(body, "tool_registry", ())
        if not registry:
            raise CompatibilityInventoryError(f"missing tool_registry in {path}")
        declarations.append(
            {
                "canonical_ids": tuple(_normalized(tool) for tool in registry),
                "tool_map": {
                    _normalized(key): _normalized(value)
                    for key, value in _literal_assignment(body, "tool_map", {}).items()
                },
                "docs": _literal_assignment(body, "tool_docs", {}),
                "dangerous": {
                    _normalized(tool) for tool in _literal_assignment(body, "dangerous_tools", set())
                },
                "source_module": path.stem,
            }
        )
    return tuple(declarations)


def _plugin_aliases(canonical_ids: tuple[str, ...], tool_map: Mapping[str, str]) -> Mapping[str, tuple[str, ...]]:
    aliases: dict[str, list[str]] = {canonical_id: [] for canonical_id in canonical_ids}
    for alias, handler in sorted(tool_map.items()):
        if alias in aliases:
            continue
        if alias in _REJECTED_LEGACY_ALIASES:
            continue
        candidates = sorted(
            canonical_id
            for canonical_id in canonical_ids
            if tool_map.get(canonical_id) == handler
        )
        if not candidates:
            if alias in _REJECTED_LEGACY_ALIASES:
                continue
            explicit_target = _EXPLICIT_PLUGIN_ALIAS_REMAPS.get(alias)
            if explicit_target not in aliases:
                raise CompatibilityInventoryError(
                    f"legacy alias {alias!r} has no canonical target; explicitly reject or remap it"
                )
            aliases[explicit_target].append(alias)
            continue
        normalized_alias = alias.replace("_", ".").replace("_", "")
        target = next(
            (
                candidate
                for candidate in candidates
                if candidate == normalized_alias
                or candidate.replace("_", "") == alias.replace("_", "")
            ),
            candidates[0],
        )
        aliases[target].append(alias)
    return {canonical_id: tuple(values) for canonical_id, values in aliases.items()}


def _classification(canonical_id: str, source_family: str, dangerous: bool, parallel_safe: bool) -> tuple[str, str, str, str, str]:
    """Return conservative confirmation/capability/concurrency/idempotency/disposition."""
    group, _, action = canonical_id.partition(".")
    reject = canonical_id == "eval.python.telegram.help"
    if canonical_id == "eval.python":
        capability = "sandbox-local"
    elif group == "eval":
        capability = "runtime-control"
    elif group == "terminal":
        capability = "process" if action in {"run", "inspect"} else "filesystem-read"
    elif group == "ast_grep":
        capability = "filesystem-write" if action == "replace" else "process"
    elif group == "web":
        capability = "network"
    elif group == "file":
        if action == "send":
            capability = "telegram-write"
        elif action in {"download_media", "write", "edit", "patch"}:
            capability = "filesystem-write"
        else:
            capability = "filesystem-read"
    elif canonical_id == "profile.download_photo":
        capability = "filesystem-write"
    elif group == "message" and action in {"react", "typing"}:
        capability = "telegram-write"
    elif group == "moderation" and action in {"unban", "unmute"}:
        capability = "telegram-admin"
    elif group in {"chat", "contacts", "creation", "dialog", "message", "moderation", "profile"}:
        capability = "telegram-admin" if dangerous else "telegram-read"
    elif group in {"mcub", "task"}:
        capability = "runtime-control"
    elif group in {"todo", "thinking"} or canonical_id in {
        "context.clear", "context.discard", "context.prune",
        "context.remember", "context.regenerate",
    }:
        capability = "state-write"
    elif (group == "skills" and action in {"install", "save_from_ai", "import_md"}) or canonical_id == "skill.save":
        capability = "filesystem-write"
    elif group == "code" and action in {"generate_file", "generate_mcub_module", "attach_result"}:
        capability = "filesystem-write"
    elif group == "utility" and action == "error_file":
        capability = "filesystem-read"
    else:
        capability = "read-only"

    mutating = dangerous or capability not in {"read-only", "filesystem-read", "telegram-read"}
    confirmation = "required" if mutating else "none"
    concurrency = "parallel-read" if source_family == "system" and parallel_safe and not mutating else "serial"
    idempotency = "non-idempotent" if mutating else "idempotent"
    disposition = "reject" if reject else "migrate"
    return confirmation, capability, concurrency, idempotency, disposition


def _entry(
    canonical_id: str,
    aliases: tuple[str, ...],
    source_family: str,
    source_module: str,
    docs: Mapping[str, Any],
    dangerous: bool,
    parallel_safe: bool,
    input_schema: Mapping[str, Any] | None = None,
    output_schema: Mapping[str, Any] | None = None,
) -> ToolCompatibility:
    confirmation, capability, concurrency, idempotency, disposition = _classification(
        canonical_id, source_family, dangerous, parallel_safe
    )
    legacy_arguments = _readonly_mapping(
        {"attrs": docs.get("args", ""), "body": docs.get("body", "")}
    )
    v2_input = _readonly_mapping(input_schema or {"status": "placeholder", "type": "object"})
    v2_output = _readonly_mapping(output_schema or {"status": "placeholder", "legacy_result": "text"})
    return ToolCompatibility(
        canonical_id=canonical_id,
        aliases=tuple(alias for alias in aliases if alias and alias != canonical_id),
        source_family=source_family,
        source_module=source_module,
        legacy_arguments=legacy_arguments,
        v2_input_schema=v2_input,
        v2_output_schema=v2_output,
        confirmation_class=confirmation,
        capability_class=capability,
        concurrency_class=concurrency,
        idempotency_class=idempotency,
        migration_disposition=disposition,
    )


def discover_compatibility_matrix(
    system_root: Path = SYSTEM_TOOLS_ROOT,
    sibling_plugins_root: Path = SIBLING_PLUGINS_ROOT,
) -> tuple[ToolCompatibility, ...]:
    """Build the immutable matrix directly from all repository declarations."""
    entries: list[ToolCompatibility] = []
    for declaration in _system_declarations(system_root):
        entries.append(
            _entry(
                declaration["canonical_id"], declaration["aliases"], "system", declaration["source_module"],
                declaration["docs"], declaration["dangerous"], declaration["parallel_safe"],
                declaration["input_schema"], declaration["output_schema"],
            )
        )
    for declaration in _plugin_declarations(sibling_plugins_root):
        if declaration.get("v2_manifest"):
            entries.extend(
                entry for entry in TOOL_COMPATIBILITY_MATRIX
                if entry.source_module == declaration["source_module"]
            )
            continue
        aliases = _plugin_aliases(declaration["canonical_ids"], declaration["tool_map"])
        for canonical_id in declaration["canonical_ids"]:
            entries.append(
                _entry(
                    canonical_id, aliases[canonical_id], "sibling-plugin", declaration["source_module"],
                    declaration["docs"].get(canonical_id, {}), canonical_id in declaration["dangerous"], False,
                )
            )
    matrix = tuple(sorted(entries, key=lambda entry: entry.canonical_id))
    validate_compatibility_matrix(matrix)
    return matrix


def validate_compatibility_matrix(matrix: Iterable[ToolCompatibility]) -> None:
    """Reject duplicates without relying on legacy dispatch precedence."""
    canonical_ids: dict[str, ToolCompatibility] = {}
    names: dict[str, str] = {}
    required = (
        "source_family", "source_module", "confirmation_class", "capability_class",
        "concurrency_class", "idempotency_class", "migration_disposition",
    )
    for entry in matrix:
        canonical_id = _normalized(entry.canonical_id)
        if not canonical_id:
            raise CompatibilityInventoryError("missing canonical ID")
        if canonical_id in canonical_ids:
            raise CompatibilityInventoryError(f"duplicate canonical ID {canonical_id!r}")
        if any(not _normalized(getattr(entry, field)) for field in required):
            raise CompatibilityInventoryError(f"unclassified canonical ID {canonical_id!r}")
        if entry.migration_disposition not in {"migrate", "reject"}:
            raise CompatibilityInventoryError(f"unclassified canonical ID {canonical_id!r}")
        canonical_ids[canonical_id] = entry
        for name in (canonical_id, *entry.aliases):
            normalized_name = _normalized(name)
            if not normalized_name:
                raise CompatibilityInventoryError(f"empty alias for canonical ID {canonical_id!r}")
            owner = names.get(normalized_name)
            if owner is not None:
                raise CompatibilityInventoryError(
                    f"duplicate alias {normalized_name!r} for canonical IDs {owner!r} and {canonical_id!r}"
                )
            names[normalized_name] = canonical_id


# Generated from source declarations; update only after reviewing declaration drift.
_TOOL_COMPATIBILITY_SNAPSHOT = (
    ('ast_grep.replace', ('astgrep.replace',), 'sibling-plugin', 'ast_grep', (('attrs', 'pattern (str); rewrite/replace (str); lang/language (str); path/paths (str); glob/globs (str); apply/update (bool, default false)'), ('body', "optional 'pattern -> rewrite' format when attrs are omitted")), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'filesystem-write', 'serial', 'non-idempotent', 'migrate'),
    ('ast_grep.search', ('ast_grep', 'astgrep.search'), 'sibling-plugin', 'ast_grep', (('attrs', "pattern (str); lang/language (str); path/paths (str, default '.'); glob/globs (str); json (bool|string)"), ('body', 'pattern text when pattern attr is omitted')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'process', 'serial', 'non-idempotent', 'migrate'),
    ('chat.admins', (), 'sibling-plugin', 'chat', (('attrs', 'chat (str) — chat identifier'), ('body', 'chat identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('chat.common_with_user', (), 'sibling-plugin', 'chat', (('attrs', 'user (str) — target user'), ('body', 'user identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('chat.info', ('chat',), 'sibling-plugin', 'chat', (('attrs', 'chat (str) or query (str) — username, ID, or link'), ('body', 'chat identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('chat.invite_link', (), 'sibling-plugin', 'chat', (('attrs', 'chat (str) — target chat'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('chat.participants', (), 'sibling-plugin', 'chat', (('attrs', 'chat (str); limit (int) — max results (default 30)'), ('body', 'chat identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('chat.permissions', (), 'sibling-plugin', 'chat', (('attrs', 'chat (str) — chat identifier'), ('body', 'chat identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('chat.set_about', ('set_chat_about',), 'sibling-plugin', 'chat', (('attrs', 'chat (str) — target; about (str) or description (str) — new text'), ('body', 'about text')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('chat.set_title', ('set_chat_title',), 'sibling-plugin', 'chat', (('attrs', 'chat (str) — target; title (str) — new title'), ('body', 'title text')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('chat.set_username', (), 'sibling-plugin', 'chat', (('attrs', 'chat (str) — target; username (str) — new username'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('chat.slowmode', ('set_slowmode',), 'sibling-plugin', 'chat', (('attrs', 'chat (str) — target; seconds (int) — slowmode delay'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('code.attach_result', (), 'system', 'Code.attach_result', (('attrs', ''), ('body', '')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'filesystem-write', 'serial', 'non-idempotent', 'migrate'),
    ('code.choose_filename', (), 'system', 'Code.choose_filename', (('attrs', 'name/path'), ('body', 'optional filename')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'read-only', 'serial', 'idempotent', 'migrate'),
    ('code.generate_file', (), 'system', 'Code.generate_file', (('attrs', 'name/path'), ('body', 'file content')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'filesystem-write', 'serial', 'non-idempotent', 'migrate'),
    ('code.generate_mcub_module', (), 'system', 'Code.generate_mcub_module', (('attrs', 'name'), ('body', 'module code')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'filesystem-write', 'serial', 'non-idempotent', 'migrate'),
    ('code.read_docs', (), 'system', 'Code.read_docs', (('attrs', ''), ('body', '')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'read-only', 'parallel-read', 'idempotent', 'migrate'),
    ('contacts.add', (), 'sibling-plugin', 'contacts', (('attrs', 'user (str); first_name (str); last_name (str); phone (str)'), ('body', 'user identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('contacts.block', (), 'sibling-plugin', 'contacts', (('attrs', 'user (str) — user to block'), ('body', 'user identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('contacts.delete', (), 'sibling-plugin', 'contacts', (('attrs', 'user (str) — user to remove'), ('body', 'user identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('contacts.entity', (), 'sibling-plugin', 'contacts', (('attrs', 'user (str) — username or ID'), ('body', 'user identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('contacts.unblock', (), 'sibling-plugin', 'contacts', (('attrs', 'user (str) — user to unblock'), ('body', 'user identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('context.clear', (), 'system', 'Context.clear', (('attrs', ''), ('body', '')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'state-write', 'serial', 'non-idempotent', 'migrate'),
    ('context.discard', (), 'system', 'Context.discard', (('attrs', 'target/all; keep'), ('body', 'optional target list')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'state-write', 'serial', 'non-idempotent', 'migrate'),
    ('context.media_context', (), 'system', 'Context.media_context', (('attrs', ''), ('body', '')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'read-only', 'parallel-read', 'idempotent', 'migrate'),
    ('context.prune', (), 'system', 'Context.prune', (('attrs', 'target/all; keep'), ('body', 'optional target list')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'state-write', 'serial', 'non-idempotent', 'migrate'),
    ('context.regenerate', (), 'system', 'Context.regenerate', (('attrs', ''), ('body', '')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'state-write', 'serial', 'non-idempotent', 'migrate'),
    ('context.remember', (), 'system', 'Context.remember', (('attrs', ''), ('body', 'memory note')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'state-write', 'serial', 'non-idempotent', 'migrate'),
    ('context.reply_context', (), 'system', 'Context.reply_context', (('attrs', ''), ('body', '')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'read-only', 'parallel-read', 'idempotent', 'migrate'),
    ('context.tool_output', ('context.read_tool_output', 'tool_output.read'), 'system', 'Context.tool_output', (('attrs', 'path/file/id; latest=true; mode=head|tail|all; limit; offset'), ('body', 'optional saved output path or filename from the tool trace')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'read-only', 'parallel-read', 'idempotent', 'migrate'),
    ('creation.bot', ('create_bot',), 'sibling-plugin', 'creation', (('attrs', 'name (str) or title (str); username (str) or bot (str); about (str)'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('creation.channel', ('create_channel', 'create_group'), 'sibling-plugin', 'creation', (('attrs', 'title (str) or name (str); about (str) or description (str)'), ('body', 'title')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('creation.group', (), 'sibling-plugin', 'creation', (('attrs', 'title (str) or name (str); about (str) or description (str)'), ('body', 'title')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('creation.private_invite', ('join_chat',), 'sibling-plugin', 'creation', (('attrs', 'link (str) — invite link'), ('body', 'invite link')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('dialog.archive', (), 'sibling-plugin', 'dialog', (('attrs', 'chat (str) or id (str) — target'), ('body', 'chat identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('dialog.export_invite', (), 'sibling-plugin', 'dialog', (('attrs', 'chat (str) — target chat'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('dialog.get_photo', (), 'sibling-plugin', 'dialog', (('attrs', 'chat (str) — target'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('dialog.leave', (), 'sibling-plugin', 'dialog', (('attrs', 'chat (str) or id (str) — target'), ('body', 'chat identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('dialog.list_all', ('dialog.list', 'dialogs'), 'sibling-plugin', 'dialog', (('attrs', 'none'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('dialog.list_groups', (), 'sibling-plugin', 'dialog', (('attrs', 'none'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('dialog.list_private', (), 'sibling-plugin', 'dialog', (('attrs', 'none'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('dialog.search', (), 'sibling-plugin', 'dialog', (('attrs', 'q (str) or query (str) — search term'), ('body', 'search term')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('dialog.set_photo', (), 'sibling-plugin', 'dialog', (('attrs', 'chat (str) — target; photo (str) — file path'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('dialog.unarchive', (), 'sibling-plugin', 'dialog', (('attrs', 'chat (str) or id (str) — target'), ('body', 'chat identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('eval.python', ('eval',), 'sibling-plugin', 'eval', (('attrs', 'code (str) or expr (str); timeout (int)'), ('body', 'Python code. You may use await and return from the async function.')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'sandbox-local', 'serial', 'non-idempotent', 'migrate'),
    ('eval.python.telegram.help', (), 'sibling-plugin', 'eval', (('attrs', ''), ('body', '')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'runtime-control', 'serial', 'non-idempotent', 'reject'),
    ('file.download_media', ('download_media', 'file.download'), 'sibling-plugin', 'file', (('attrs', 'message (str) or msg (str) — message ID; chat (str) or from (str)'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'filesystem-write', 'serial', 'non-idempotent', 'migrate'),
    ('file.edit', (), 'sibling-plugin', 'file', (('attrs', 'path (str); search (str); replace (str); count (int)'), ('body', "'search -> replace' format")), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'filesystem-write', 'serial', 'non-idempotent', 'migrate'),
    ('file.patch', (), 'sibling-plugin', 'file', (('attrs', "path (str); reverse (str) — 'true' to reverse"), ('body', 'unified diff content')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'filesystem-write', 'serial', 'non-idempotent', 'migrate'),
    ('file.read_text', (), 'sibling-plugin', 'file', (('attrs', 'path (str) or file (str) or name (str)'), ('body', 'file path')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'filesystem-read', 'serial', 'idempotent', 'migrate'),
    ('file.send', ('send_file',), 'sibling-plugin', 'file', (('attrs', 'path (str) or file (str); chat (str) — target; caption (str)'), ('body', 'file path')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-write', 'serial', 'non-idempotent', 'migrate'),
    ('file.write', (), 'sibling-plugin', 'file', (('attrs', "path (str) — file; mode (str) — 'overwrite' or 'append'"), ('body', 'content to write')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'filesystem-write', 'serial', 'non-idempotent', 'migrate'),
    ('mcub.command', ('mcub',), 'sibling-plugin', 'mcub', (('attrs', 'command (str) or cmd (str) — command text (prefix auto-added)'), ('body', 'command text')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'runtime-control', 'serial', 'non-idempotent', 'migrate'),
    ('mcub.config', (), 'sibling-plugin', 'mcub', (('attrs', "command (str) or query (str) — config command like 'module.key=value'"), ('body', 'config command')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'runtime-control', 'serial', 'non-idempotent', 'migrate'),
    ('mcub.install', (), 'sibling-plugin', 'mcub', (('attrs', 'command (str) or query (str) — module URL or name'), ('body', 'URL or name')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'runtime-control', 'serial', 'non-idempotent', 'migrate'),
    ('mcub.modules', (), 'sibling-plugin', 'mcub', (('attrs', 'command (str) or query (str)'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'runtime-control', 'serial', 'non-idempotent', 'migrate'),
    ('mcub.reload', (), 'sibling-plugin', 'mcub', (('attrs', 'none'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'runtime-control', 'serial', 'non-idempotent', 'migrate'),
    ('message.delete', ('delete_messages',), 'sibling-plugin', 'message', (('attrs', 'ids (str) — comma-separated IDs'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('message.draft', (), 'sibling-plugin', 'message', (('attrs', 'chat (str); message (str) — draft text'), ('body', 'draft text')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('message.edit', (), 'sibling-plugin', 'message', (('attrs', 'message (str) or msg (str) — message ID or link'), ('body', 'new text')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('message.forward', ('forward_message',), 'sibling-plugin', 'message', (('attrs', 'from_chat (str); msg_id (int); to (str) — destination'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('message.get', (), 'sibling-plugin', 'message', (('attrs', 'chat (str); message (str) or msg (str) — message ID'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('message.history', ('history',), 'sibling-plugin', 'message', (('attrs', 'chat (str) — optional; limit (int) — max messages (default 20)'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('message.mark_read', (), 'sibling-plugin', 'message', (('attrs', 'chat (str); max_id (int) — optional'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('message.pin', ('pin_message',), 'sibling-plugin', 'message', (('attrs', 'chat (str) — target; message (str) or msg (str) — message ID'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('message.react', (), 'sibling-plugin', 'message', (('attrs', 'chat (str); message (str) or msg (str) — message ID; emoji (str) — reaction'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-write', 'serial', 'non-idempotent', 'migrate'),
    ('message.reply', (), 'sibling-plugin', 'message', (('attrs', 'chat (str); reply_to (int) — message ID; message (str) — content'), ('body', 'reply text')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('message.schedule', (), 'sibling-plugin', 'message', (('attrs', 'chat (str); message (str) — content; schedule (int) — unix timestamp'), ('body', 'message text')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('message.search', ('search_messages',), 'sibling-plugin', 'message', (('attrs', 'q (str) or query (str) — search text; chat (str) — optional scope'), ('body', 'query text')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('message.send_current', ('message.send', 'send_message'), 'sibling-plugin', 'message', (('attrs', 'message (str) or text (str) — content'), ('body', 'message text')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('message.send_target', (), 'sibling-plugin', 'message', (('attrs', 'chat (str) or to (str) — target; message (str) or text (str) — content'), ('body', 'message text')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('message.typing', (), 'sibling-plugin', 'message', (('attrs', 'chat (str) — target chat'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-write', 'serial', 'non-idempotent', 'migrate'),
    ('moderation.ban', ('ban_user', 'chat.ban'), 'sibling-plugin', 'moderation', (('attrs', 'chat (str); user (str); reason (str) — optional'), ('body', 'user identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('moderation.delete_messages', (), 'sibling-plugin', 'moderation', (('attrs', 'chat (str); ids (str) — comma-separated IDs'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('moderation.demote', ('chat.demote', 'demote_user'), 'sibling-plugin', 'moderation', (('attrs', 'chat (str); user (str)'), ('body', 'user identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('moderation.get_admins', (), 'sibling-plugin', 'moderation', (('attrs', 'chat (str)'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('moderation.kick', ('chat.kick', 'kick_user'), 'sibling-plugin', 'moderation', (('attrs', 'chat (str); user (str); reason (str) — optional'), ('body', 'user identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('moderation.mute', ('chat.mute', 'mute_user'), 'sibling-plugin', 'moderation', (('attrs', 'chat (str); user (str); until (int) — seconds (default 3600)'), ('body', 'user identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('moderation.pin', (), 'sibling-plugin', 'moderation', (('attrs', 'chat (str); message (str) or msg (str) — message ID'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('moderation.promote', ('chat.promote', 'promote_user'), 'sibling-plugin', 'moderation', (('attrs', 'chat (str); user (str); rank (str) — admin title'), ('body', 'user identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('moderation.unban', ('chat.unban', 'unban_user'), 'sibling-plugin', 'moderation', (('attrs', 'chat (str); user (str)'), ('body', 'user identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('moderation.unmute', ('chat.unmute', 'unmute_user'), 'sibling-plugin', 'moderation', (('attrs', 'chat (str); user (str)'), ('body', 'user identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('profile.common_chats', (), 'sibling-plugin', 'profile', (('attrs', 'target (str) or user (str)'), ('body', 'user identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('profile.download_photo', (), 'sibling-plugin', 'profile', (('attrs', 'target (str) or user (str)'), ('body', 'user identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'filesystem-write', 'serial', 'non-idempotent', 'migrate'),
    ('profile.get', ('profile',), 'sibling-plugin', 'profile', (('attrs', 'target (str) or user (str) — username or ID'), ('body', 'user identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('profile.get_full', (), 'sibling-plugin', 'profile', (('attrs', 'target (str) or user (str) — username or ID'), ('body', 'user identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('profile.get_me', (), 'sibling-plugin', 'profile', (('attrs', 'none'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('profile.get_photos', (), 'sibling-plugin', 'profile', (('attrs', "target (str) or user (str); download (str) — set to 'true' to download"), ('body', 'user identifier')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'telegram-read', 'serial', 'idempotent', 'migrate'),
    ('profile.set_photo', ('set_profile_photo',), 'sibling-plugin', 'profile', (('attrs', 'photo (str) or file (str) — path to image'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('profile.update_bio', ('profile.update', 'update_profile'), 'sibling-plugin', 'profile', (('attrs', 'bio (str) or about (str) — new bio text'), ('body', 'bio text')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('profile.update_name', (), 'sibling-plugin', 'profile', (('attrs', 'first_name (str); last_name (str) — optional'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('profile.update_username', (), 'sibling-plugin', 'profile', (('attrs', 'username (str) — new username (without @)'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'telegram-admin', 'serial', 'non-idempotent', 'migrate'),
    ('skill.save', ('skill',), 'system', 'SkillsAgent.save', (('attrs', 'name/title'), ('body', 'skill markdown/content')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'filesystem-write', 'serial', 'non-idempotent', 'migrate'),
    ('skills.activate', (), 'system', 'SkillsAgent.activate', (('attrs', 'query/name'), ('body', 'optional query')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'read-only', 'serial', 'idempotent', 'migrate'),
    ('skills.export_md', (), 'system', 'SkillsAgent.export_md', (('attrs', 'name'), ('body', 'optional skill name')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'read-only', 'serial', 'idempotent', 'migrate'),
    ('skills.import_md', (), 'system', 'SkillsAgent.import_md', (('attrs', 'name/title'), ('body', 'markdown content')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'filesystem-write', 'serial', 'non-idempotent', 'migrate'),
    ('skills.install', (), 'system', 'SkillsAgent.install', (('attrs', 'name'), ('body', 'optional skill name')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'filesystem-write', 'serial', 'non-idempotent', 'migrate'),
    ('skills.list', (), 'system', 'SkillsAgent.list', (('attrs', ''), ('body', '')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'read-only', 'parallel-read', 'idempotent', 'migrate'),
    ('skills.read', (), 'system', 'SkillsAgent.read', (('attrs', 'name'), ('body', 'optional skill name')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'read-only', 'parallel-read', 'idempotent', 'migrate'),
    ('skills.repo_list', (), 'system', 'SkillsAgent.repo_list', (('attrs', ''), ('body', '')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'read-only', 'parallel-read', 'idempotent', 'migrate'),
    ('skills.save_from_ai', (), 'system', 'SkillsAgent.save_from_ai', (('attrs', 'name/title'), ('body', 'skill content')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'filesystem-write', 'serial', 'non-idempotent', 'migrate'),
    ('task.background', (), 'sibling-plugin', 'task', (('attrs', 'tool/name (str) — tool to run; attrs (str) — inner attrs; label (str)'), ('body', 'body for the inner tool')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'runtime-control', 'serial', 'non-idempotent', 'migrate'),
    ('task.run_background', (), 'sibling-plugin', 'task', (('attrs', 'tool/name (str); attrs (str); label (str)'), ('body', 'body for the inner tool')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'runtime-control', 'serial', 'non-idempotent', 'migrate'),
    ('terminal.git_status', (), 'sibling-plugin', 'terminal', (('attrs', 'none'), ('body', 'not used')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'filesystem-read', 'serial', 'idempotent', 'migrate'),
    ('terminal.inspect', (), 'sibling-plugin', 'terminal', (('attrs', 'command (str) or cmd (str) — command to run'), ('body', 'command text')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'process', 'serial', 'non-idempotent', 'migrate'),
    ('terminal.list_files', (), 'sibling-plugin', 'terminal', (('attrs', 'path (str) — directory path (default: .)'), ('body', 'path text')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'filesystem-read', 'serial', 'idempotent', 'migrate'),
    ('terminal.read_file', (), 'sibling-plugin', 'terminal', (('attrs', 'path (str) or file (str) — file to read'), ('body', 'path text')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'filesystem-read', 'serial', 'idempotent', 'migrate'),
    ('terminal.run', ('terminal',), 'sibling-plugin', 'terminal', (('attrs', 'command (str) or cmd (str) — shell command to execute'), ('body', 'command text')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'process', 'serial', 'non-idempotent', 'migrate'),
    ('thinking.note', (), 'system', 'Thinking.note', (('attrs', 'note/text'), ('body', 'optional note text')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'state-write', 'serial', 'non-idempotent', 'migrate'),
    ('todo.add', (), 'system', 'Todo.add', (('attrs', 'text/task'), ('body', '')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'state-write', 'serial', 'non-idempotent', 'migrate'),
    ('todo.clear', (), 'system', 'Todo.clear', (('attrs', ''), ('body', '')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'state-write', 'serial', 'non-idempotent', 'migrate'),
    ('todo.close', (), 'system', 'Todo.close', (('attrs', 'id/index/text'), ('body', '')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'state-write', 'serial', 'non-idempotent', 'migrate'),
    ('todo.closeall', (), 'system', 'Todo.closeall', (('attrs', ''), ('body', '')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'state-write', 'serial', 'non-idempotent', 'migrate'),
    ('todo.current', (), 'system', 'Todo.current', (('attrs', ''), ('body', '')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'state-write', 'serial', 'non-idempotent', 'migrate'),
    ('todo.delete', (), 'system', 'Todo.delete', (('attrs', 'id/index/text'), ('body', '')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'state-write', 'serial', 'non-idempotent', 'migrate'),
    ('todo.edit', (), 'system', 'Todo.edit', (('attrs', 'id/index/text/status'), ('body', '')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'state-write', 'serial', 'non-idempotent', 'migrate'),
    ('utility.agent_log', (), 'system', 'Utility.agent_log', (('attrs', ''), ('body', '')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'read-only', 'parallel-read', 'idempotent', 'migrate'),
    ('utility.error_file', (), 'system', 'Utility.error_file', (('attrs', ''), ('body', '')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'filesystem-read', 'serial', 'idempotent', 'migrate'),
    ('utility.list_tools', (), 'system', 'Utility.list_tools', (('attrs', ''), ('body', '')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'read-only', 'parallel-read', 'idempotent', 'migrate'),
    ('utility.placeholders', (), 'system', 'Utility.placeholders', (('attrs', ''), ('body', '')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'read-only', 'parallel-read', 'idempotent', 'migrate'),
    ('utility.plugin_docs', (), 'system', 'Utility.plugin_docs', (('attrs', 'plugin/name (str, optional) — plugin to inspect'), ('body', 'optional plugin name')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'read-only', 'parallel-read', 'idempotent', 'migrate'),
    ('utility.random_template', (), 'system', 'Utility.random_template', (('attrs', ''), ('body', '')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'read-only', 'parallel-read', 'idempotent', 'migrate'),
    ('utility.search_tool', (), 'system', 'Utility.search_tool', (('attrs', 'query (str) — capability or tool name to search for'), ('body', 'optional natural-language search query')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'read-only', 'parallel-read', 'idempotent', 'migrate'),
    ('utility.token_usage', (), 'system', 'Utility.token_usage', (('attrs', ''), ('body', '')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'read-only', 'parallel-read', 'idempotent', 'migrate'),
    ('utility.tool_help', (), 'system', 'Utility.tool_help', (('attrs', 'tool (str) — exact tool name'), ('body', 'optional tool name')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'none', 'read-only', 'parallel-read', 'idempotent', 'migrate'),
    ('web.extract_links', (), 'sibling-plugin', 'web', (('attrs', 'url (str) — page to scan'), ('body', 'URL')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'network', 'serial', 'non-idempotent', 'migrate'),
    ('web.fetch_url', (), 'sibling-plugin', 'web', (('attrs', 'url (str) — the web page to fetch'), ('body', 'URL')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'network', 'serial', 'non-idempotent', 'migrate'),
    ('web.read_html', (), 'sibling-plugin', 'web', (('attrs', 'url (str) — page to read'), ('body', 'URL')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'network', 'serial', 'non-idempotent', 'migrate'),
    ('web.search', ('web_search',), 'sibling-plugin', 'web', (('attrs', 'query (str) or q (str) or url (str) — search query or web page URL'), ('body', 'query or URL')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'network', 'serial', 'non-idempotent', 'migrate'),
    ('web.summarize_page', (), 'sibling-plugin', 'web', (('attrs', 'url (str) — page to summarize'), ('body', 'URL')), (('status', 'placeholder'), ('type', 'object')), (('status', 'placeholder'), ('legacy_result', 'text')), 'required', 'network', 'serial', 'non-idempotent', 'migrate'),
)


TOOL_COMPATIBILITY_MATRIX = tuple(
    ToolCompatibility(
        canonical_id=row[0],
        aliases=row[1],
        source_family=row[2],
        source_module=row[3],
        legacy_arguments=_readonly_mapping(dict(row[4])),
        v2_input_schema=_readonly_mapping(dict(row[5])),
        v2_output_schema=_readonly_mapping(dict(row[6])),
        confirmation_class=row[7],
        capability_class=row[8],
        concurrency_class=row[9],
        idempotency_class=row[10],
        migration_disposition=row[11],
    )
    for row in _TOOL_COMPATIBILITY_SNAPSHOT
)
validate_compatibility_matrix(TOOL_COMPATIBILITY_MATRIX)


def compatibility_matrix() -> tuple[ToolCompatibility, ...]:
    """Return the immutable inventory tuple; entries contain read-only mappings."""
    return TOOL_COMPATIBILITY_MATRIX


__all__ = [
    "CompatibilityInventoryError",
    "SIBLING_PLUGINS_ROOT",
    "SYSTEM_TOOLS_ROOT",
    "TOOL_COMPATIBILITY_MATRIX",
    "ToolCompatibility",
    "compatibility_matrix",
    "discover_compatibility_matrix",
    "validate_compatibility_matrix",
]
