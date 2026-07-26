# SPDX-License-Identifier: MIT
from __future__ import annotations

SYSTEM_TOOL = {
    "tool_class": 'thinking',
    "name": 'note',
    "handler": '_thinking_note_tool',
    "docs": {'desc': 'Record a concise progress/thinking note for the user.', 'args': 'note/text', 'body': 'optional note text'},
}
