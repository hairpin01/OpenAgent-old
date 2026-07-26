# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'todo',
    "name": 'delete',
    "handler": '_todo_registry_tool',
    "docs": {'desc': 'Delete a TODO item.', 'args': 'id/index/text'},
}
