# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'skills',
    "name": 'save_from_ai',
    "handler": '_skills_registry_tool',
    "docs": {'desc': 'Persist useful knowledge as an OpenAgent skill.', 'args': 'name/title', 'body': 'skill content'},
}
