from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

from installed_plugin_consumer_matrix import (
    CONSUMERS,
    LIFECYCLE_STATES,
    PRESENTATION_FIELDS,
)

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {
    "_plugins",
    "_plugin_files",
    "_v2_plugin_sources",
    "_disabled_plugins",
    "_v2_plugin_invoker",
    "_plugin_unload_tasks",
    "_background_tool_tasks",
    "_v2_source_event",
    "_plugins_cache",
}
FORBIDDEN_INSTALLED_STATE_FIELDS = frozenset(
    {"_plugins", "_plugin_files", "_v2_plugin_sources", "_disabled_plugins"}
)


class _ConsumerVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.classes: list[str] = []
        self.functions: list[str] = []
        self.records: set[tuple[str, str]] = set()

    @property
    def owner(self) -> str:
        return ".".join((*self.classes, *self.functions))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes.append(node.name)
        self.generic_visit(node)
        self.classes.pop()

    def _visit_function(self, node: ast.AST, name: str) -> None:
        self.functions.append(name)
        self.generic_visit(node)
        self.functions.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr in FIELDS
        ):
            self.records.add((self.owner, node.attr))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "self"
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in FIELDS
        ):
            self.records.add((self.owner, node.args[1].value))
        self.generic_visit(node)


def _production_consumers() -> dict[tuple[str, str], set[str]]:
    found: dict[tuple[str, str], set[str]] = defaultdict(set)
    for path in (ROOT / "Src").rglob("*.py"):
        visitor = _ConsumerVisitor()
        visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        relative = path.relative_to(ROOT).as_posix()
        for owner, field in visitor.records:
            found[(relative, field)].add(owner)
    return found


def test_matrix_is_immutable_and_contract_complete() -> None:
    assert isinstance(CONSUMERS, tuple)
    assert CONSUMERS
    for entry in CONSUMERS:
        assert entry.path.startswith("Src/")
        assert entry.owner and entry.field in FIELDS and entry.access
        assert entry.behavior and entry.replacement.startswith("registry.")
        assert entry.migration_task in {"2", "3", "5", "6", "7", "8"}

    mapped = {(entry.path, entry.owner, entry.field) for entry in CONSUMERS}
    missing: list[str] = []
    for (path, field), owners in sorted(_production_consumers().items()):
        for owner in sorted(owners):
            if (path, owner, field) not in mapped:
                missing.append(f"{path}:{owner}:{field}")
    assert not missing, "Unmapped installed-plugin state consumer(s):\n" + "\n".join(
        missing
    )


def test_matrix_covers_lifecycle_states_and_presentation_contract() -> None:
    assert LIFECYCLE_STATES == {
        "admitted",
        "installed",
        "enabled",
        "active",
        "unloading",
        "failed",
        "disabled",
        "removed",
    }
    assert {
        "id",
        "manifest",
        "source",
        "path",
        "tools",
        "version",
        "author",
        "description",
    } <= PRESENTATION_FIELDS


def test_completed_cutover_has_no_forbidden_installed_state_consumers() -> None:
    """Keep the historical matrix while rejecting any residual production map."""

    residual = sorted(
        f"{path}:{owner}:{field}"
        for (path, field), owners in _production_consumers().items()
        if field in FORBIDDEN_INSTALLED_STATE_FIELDS
        for owner in owners
    )
    assert not residual, "Residual installed-plugin state consumer(s):\n" + "\n".join(
        residual
    )
