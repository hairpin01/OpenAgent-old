# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'skills',
    "name": 'read',
    "handler": '_skills_registry_tool',
    "docs": {'desc': 'Read an installed OpenAgent skill.', 'args': 'name', 'body': 'optional skill name'},
}
