# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'skills',
    "name": 'import_md',
    "handler": '_skills_registry_tool',
    "docs": {'desc': 'Import a skill from markdown body.', 'args': 'name/title', 'body': 'markdown content'},
}
