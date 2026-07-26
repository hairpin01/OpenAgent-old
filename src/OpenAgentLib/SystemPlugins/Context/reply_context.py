# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'context',
    "name": 'reply_context',
    "handler": '_context_registry_tool',
    "docs": {'desc': 'Read context from the replied message.'},
}
