# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'todo',
    "name": 'close',
    "handler": '_todo_registry_tool',
    "docs": {'desc': 'Mark a TODO item as closed.', 'args': 'id/index/text'},
}
