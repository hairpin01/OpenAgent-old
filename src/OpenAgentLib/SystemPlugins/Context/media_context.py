# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'context',
    "name": 'media_context',
    "handler": '_context_registry_tool',
    "docs": {'desc': 'Read replied media/message context.'},
}
