# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'skills',
    "name": 'export_md',
    "handler": '_skills_registry_tool',
    "docs": {'desc': 'Export/read an installed skill as markdown.', 'args': 'name', 'body': 'optional skill name'},
}
