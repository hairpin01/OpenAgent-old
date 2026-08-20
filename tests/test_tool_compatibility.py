from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from conftest import load_source_module


compatibility = load_source_module(
    "openagent_tool_compatibility_test",
    "Src/OpenAgentLib/ToolCompatibility.py",
)


@pytest.fixture(scope="module")
def matrix():
    return compatibility.compatibility_matrix()


@pytest.fixture(params=compatibility.compatibility_matrix(), ids=lambda entry: entry.canonical_id)
def matrix_case(request):
    """Generated coverage case for each machine-readable inventory entry."""
    return request.param


def _system_ids(root: Path) -> set[str]:
    return {
        declaration["canonical_id"]
        for declaration in compatibility._system_declarations(root)
    }


def _sibling_ids(root: Path) -> set[str]:
    return {
        canonical_id
        for declaration in compatibility._plugin_declarations(root)
        for canonical_id in declaration["canonical_ids"]
    }


def test_complete_inventory(matrix) -> None:
    expected_system = _system_ids(compatibility.SYSTEM_TOOLS_ROOT)
    expected_sibling = _sibling_ids(compatibility.SIBLING_PLUGINS_ROOT)
    actual_system = {entry.canonical_id for entry in matrix if entry.source_family == "system"}
    actual_sibling = {entry.canonical_id for entry in matrix if entry.source_family == "sibling-plugin"}

    assert actual_system == expected_system, (
        f"missing system canonical IDs: {sorted(expected_system - actual_system)}; "
        f"unexpected: {sorted(actual_system - expected_system)}"
    )
    assert actual_sibling == expected_sibling, (
        f"missing sibling canonical IDs: {sorted(expected_sibling - actual_sibling)}; "
        f"unexpected: {sorted(actual_sibling - expected_sibling)}"
    )


def test_complete_alias_inventory(matrix) -> None:
    system_declarations = compatibility._system_declarations(compatibility.SYSTEM_TOOLS_ROOT)
    plugin_declarations = compatibility._plugin_declarations(compatibility.SIBLING_PLUGINS_ROOT)
    expected_system_aliases = {
        alias for declaration in system_declarations for alias in declaration["aliases"]
    }
    expected_plugin_aliases = {
        alias
        for declaration in plugin_declarations
        for alias in declaration["tool_map"]
        if alias not in declaration["canonical_ids"]
        and alias not in compatibility._REJECTED_LEGACY_ALIASES
    }
    actual_system_aliases = {
        alias for entry in matrix if entry.source_family == "system" for alias in entry.aliases
    }
    actual_plugin_aliases = {
        alias for entry in matrix if entry.source_family == "sibling-plugin" for alias in entry.aliases
    }

    assert actual_system_aliases == expected_system_aliases, (
        f"missing system aliases: {sorted(expected_system_aliases - actual_system_aliases)}; "
        f"unexpected: {sorted(actual_system_aliases - expected_system_aliases)}"
    )
    assert actual_plugin_aliases == expected_plugin_aliases, (
        f"missing sibling aliases: {sorted(expected_plugin_aliases - actual_plugin_aliases)}; "
        f"unexpected: {sorted(actual_plugin_aliases - expected_plugin_aliases)}"
    )


def test_static_snapshot_matches_ast_discovery(matrix) -> None:
    assert matrix == compatibility.discover_compatibility_matrix()


@pytest.mark.parametrize(
    ("canonical_id", "capability", "confirmation", "idempotency", "disposition"),
    (
        ("context.clear", "state-write", "required", "non-idempotent", "migrate"),
        ("skill.save", "filesystem-write", "required", "non-idempotent", "migrate"),
        ("file.write", "filesystem-write", "required", "non-idempotent", "migrate"),
        ("file.edit", "filesystem-write", "required", "non-idempotent", "migrate"),
        ("file.patch", "filesystem-write", "required", "non-idempotent", "migrate"),
        ("file.send", "telegram-write", "required", "non-idempotent", "migrate"),
        ("file.download_media", "filesystem-write", "required", "non-idempotent", "migrate"),
        ("profile.download_photo", "filesystem-write", "required", "non-idempotent", "migrate"),
        ("message.react", "telegram-write", "required", "non-idempotent", "migrate"),
        ("message.typing", "telegram-write", "required", "non-idempotent", "migrate"),
        ("moderation.unban", "telegram-admin", "required", "non-idempotent", "migrate"),
        ("moderation.unmute", "telegram-admin", "required", "non-idempotent", "migrate"),
        ("terminal.run", "process", "required", "non-idempotent", "migrate"),
        ("terminal.inspect", "process", "required", "non-idempotent", "migrate"),
        ("eval.python", "sandbox-local", "required", "non-idempotent", "migrate"),
        ("eval.python.telegram.help", "runtime-control", "required", "non-idempotent", "reject"),
    ),
)
def test_representative_high_risk_classification(
    matrix, canonical_id, capability, confirmation, idempotency, disposition
) -> None:
    entry = next(entry for entry in matrix if entry.canonical_id == canonical_id)

    assert entry.capability_class == capability
    assert entry.confirmation_class == confirmation
    assert entry.idempotency_class == idempotency
    assert entry.migration_disposition == disposition


_MUTATION_VERBS = {
    "activate", "add", "archive", "attach", "ban", "block", "clear",
    "delete", "demote", "discard", "download", "edit", "export", "forward",
    "generate", "import", "install", "kick", "leave", "mark", "mute", "patch",
    "pin", "promote", "prune", "react", "regenerate", "reload", "remember",
    "replace", "reply", "run", "save", "schedule", "send", "set", "slowmode",
    "typing", "unarchive", "unban", "unblock", "unmute", "unpin", "update",
    "write",
}
_REVIEWED_GENERIC_READ_ONLY_MUTATION_NAMES = {
    # These handlers only return existing text; their verb-like names do not mutate state.
    "context.reply_context",
    "skills.activate",
    "skills.export_md",
}


def test_mutation_verbs_are_not_generic_read_only_without_review(matrix) -> None:
    generic_read_only_mutation_names = {
        entry.canonical_id
        for entry in matrix
        if entry.capability_class == "read-only"
        and _MUTATION_VERBS.intersection(entry.canonical_id.rpartition(".")[2].split("_"))
    }

    assert generic_read_only_mutation_names == _REVIEWED_GENERIC_READ_ONLY_MUTATION_NAMES


def test_import_does_not_discover_tools(monkeypatch) -> None:
    def fail_discovery(*args, **kwargs):
        raise AssertionError("tool declarations were read during import")

    monkeypatch.setattr(Path, "rglob", fail_discovery)
    monkeypatch.setattr(Path, "glob", fail_discovery)

    imported = load_source_module(
        "openagent_tool_compatibility_static_import_test",
        "Src/OpenAgentLib/ToolCompatibility.py",
    )

    assert [entry.canonical_id for entry in imported.compatibility_matrix()] == [
        entry.canonical_id for entry in compatibility.compatibility_matrix()
    ]


def test_chat_search_is_explicitly_rejected(matrix) -> None:
    assert "chat.search" not in {
        alias for entry in matrix for alias in entry.aliases
    }
    assert compatibility._REJECTED_LEGACY_ALIASES["chat.search"] == (
        "cmd_search has no tool_registry canonical ID"
    )


def test_unsandboxed_eval_alias_is_explicitly_rejected(matrix) -> None:
    assert "eval.python.telegram" not in {
        alias for entry in matrix for alias in entry.aliases
    }
    assert "eval.python.telegram" in compatibility._REJECTED_LEGACY_ALIASES


def test_unmapped_plugin_alias_is_rejected() -> None:
    with pytest.raises(
        compatibility.CompatibilityInventoryError,
        match="explicitly reject or remap",
    ):
        compatibility._plugin_aliases(
            ("chat.info",),
            {"chat.info": "cmd_info", "chat.unknown": "cmd_unknown"},
        )


def test_every_entry_is_explicitly_classified(matrix_case) -> None:
    assert matrix_case.migration_disposition in {"migrate", "reject"}, matrix_case.canonical_id
    assert matrix_case.confirmation_class in {"none", "required"}, matrix_case.canonical_id
    assert matrix_case.capability_class, matrix_case.canonical_id
    assert matrix_case.concurrency_class in {"parallel-read", "serial"}, matrix_case.canonical_id
    assert matrix_case.idempotency_class in {"idempotent", "non-idempotent"}, matrix_case.canonical_id
    assert matrix_case.legacy_arguments["attrs"] is not None, matrix_case.canonical_id
    assert matrix_case.legacy_arguments["body"] is not None, matrix_case.canonical_id
    assert matrix_case.v2_input_schema["status"] == "placeholder", matrix_case.canonical_id
    assert matrix_case.v2_output_schema["status"] == "placeholder", matrix_case.canonical_id


def test_matrix_and_schemas_are_read_only(matrix) -> None:
    assert isinstance(matrix, tuple)
    assert isinstance(matrix[0].legacy_arguments, MappingProxyType)
    with pytest.raises(TypeError):
        matrix[0].legacy_arguments["attrs"] = "changed"


def test_duplicate_canonical_id_fails_deterministically(matrix) -> None:
    duplicate = replace(matrix[0], source_module="duplicate")
    with pytest.raises(compatibility.CompatibilityInventoryError, match="duplicate canonical ID '" + matrix[0].canonical_id + "'"):
        compatibility.validate_compatibility_matrix((matrix[0], duplicate))


def test_duplicate_alias_fails_with_owners(matrix) -> None:
    first, second = matrix[:2]
    colliding = replace(second, aliases=(first.canonical_id,))
    with pytest.raises(
        compatibility.CompatibilityInventoryError,
        match="duplicate alias '" + first.canonical_id + "'.*" + first.canonical_id + ".*" + second.canonical_id,
    ):
        compatibility.validate_compatibility_matrix((first, colliding))


def test_aliases_resolve_exactly_once(matrix) -> None:
    names = [name for entry in matrix for name in (entry.canonical_id, *entry.aliases)]
    assert len(names) == len(set(names))
