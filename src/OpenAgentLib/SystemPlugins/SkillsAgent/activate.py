# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'skills',
    "name": 'activate',
    "handler": '_skills_registry_tool',
    "docs": {'desc': 'Activate/load the best matching installed skill for the current task.', 'args': 'query/name', 'body': 'optional query'},
}
