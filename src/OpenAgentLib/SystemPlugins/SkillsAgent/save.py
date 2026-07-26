# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'skill',
    "name": 'save',
    "handler": '_skills_registry_tool',
    "docs": {'desc': 'Save an OpenAgent skill from body text.', 'args': 'name/title', 'body': 'skill markdown/content'},
    "aliases": ('skill',),
}
