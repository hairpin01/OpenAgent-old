# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'skills',
    "name": 'install',
    "handler": '_skills_registry_tool',
    "docs": {'desc': 'Install a skill from the configured skill repository.', 'args': 'name', 'body': 'optional skill name'},
}
