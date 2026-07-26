# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'utility',
    "name": 'tool_help',
    "handler": '_utility_registry_tool',
    "docs": {'desc': 'Show normalized documentation for one core/plugin tool.', 'args': 'tool (str) — exact tool name', 'body': 'optional tool name'},
}
