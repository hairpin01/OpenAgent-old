# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'utility',
    "name": 'plugin_docs',
    "handler": '_utility_registry_tool',
    "docs": {'desc': "Show activated plugin documentation and each plugin's tools.", 'args': 'plugin/name (str, optional) — plugin to inspect', 'body': 'optional plugin name'},
}
