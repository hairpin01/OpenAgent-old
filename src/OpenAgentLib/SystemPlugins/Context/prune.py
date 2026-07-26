# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'context',
    "name": 'prune',
    "handler": '_context_registry_tool',
    "docs": {'desc': 'Prune internal OpenAgent context: history, tools, tool_memory, runtime_comments, or all.', 'args': 'target/all; keep', 'body': 'optional target list'},
}
