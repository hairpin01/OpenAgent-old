# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'todo',
    "name": 'add',
    "handler": '_todo_registry_tool',
    "docs": {'desc': 'Add a TODO item.', 'args': 'text/task'},
}
