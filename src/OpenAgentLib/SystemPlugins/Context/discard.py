# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'context',
    "name": 'discard',
    "handler": '_context_registry_tool',
    "docs": {'desc': 'Alias for context.prune.', 'args': 'target/all; keep', 'body': 'optional target list'},
}
