# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'code',
    "name": 'choose_filename',
    "handler": '_code_registry_tool',
    "docs": {'desc': 'Choose/sanitize a filename for generated code.', 'args': 'name/path', 'body': 'optional filename'},
}
