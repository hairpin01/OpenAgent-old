# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'utility',
    "name": 'token_usage',
    "handler": '_utility_registry_tool',
    "docs": {'desc': 'Show token usage from the last provider response.'},
}
