# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'code',
    "name": 'generate_file',
    "handler": '_code_registry_tool',
    "docs": {'desc': 'Generate a text/code file and keep it for sending/attaching.', 'args': 'name/path', 'body': 'file content'},
}
