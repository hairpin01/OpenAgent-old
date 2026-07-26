# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'todo',
    "name": 'edit',
    "handler": '_todo_registry_tool',
    "docs": {'desc': 'Edit a TODO item.', 'args': 'id/index/text/status'},
}
