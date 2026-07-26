# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'code',
    "name": 'generate_mcub_module',
    "handler": '_code_registry_tool',
    "docs": {'desc': 'Generate an MCUB module file.', 'args': 'name', 'body': 'module code'},
}
