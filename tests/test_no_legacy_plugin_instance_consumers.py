from __future__ import annotations

import ast
from pathlib import Path

import pytest

from OpenAgentLib.Plugin.PluginsEngine import _OpenAgentPluginSkillMixin
from OpenAgentLib.PluginSDK import LegacyPluginMigrationError

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_FIELDS = frozenset(
    {"_plugins", "_plugin_files", "_v2_plugin_sources", "_disabled_plugins"}
)


class _LegacyStateVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.accesses: list[str] = []

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr in FORBIDDEN_FIELDS
        ):
            self.accesses.append(node.attr)
        self.generic_visit(node)


def test_production_has_no_legacy_installed_plugin_instance_maps() -> None:
    residual: list[str] = []
    for path in (ROOT / "Src").rglob("*.py"):
        visitor = _LegacyStateVisitor()
        visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        residual.extend(
            f"{path.relative_to(ROOT)}:{field}" for field in visitor.accesses
        )
    assert not residual


def test_legacy_registration_fails_without_storing_an_instance() -> None:
    mixin = object.__new__(_OpenAgentPluginSkillMixin)
    with pytest.raises(LegacyPluginMigrationError, match="removed legacy execution"):
        mixin._register_plugin(object())
    assert not hasattr(mixin, "_plugins")
