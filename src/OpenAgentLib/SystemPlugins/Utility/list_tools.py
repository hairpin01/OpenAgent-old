# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'utility',
    "name": 'list_tools',
    "handler": '_utility_registry_tool',
    "docs": {'desc': 'List all available core and plugin tools by category with short descriptions.'},
}
