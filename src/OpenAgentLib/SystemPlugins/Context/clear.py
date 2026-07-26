# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'context',
    "name": 'clear',
    "handler": '_context_registry_tool',
    "docs": {'desc': 'Clear the active OpenAgent session context.'},
}
