# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'context',
    "name": 'remember',
    "handler": '_context_registry_tool',
    "docs": {'desc': 'Remember a note in the active chat context.', 'body': 'memory note'},
}
