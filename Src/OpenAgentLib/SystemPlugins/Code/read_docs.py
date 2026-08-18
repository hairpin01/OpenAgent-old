# SPDX-License-Identifier: MIT
from __future__ import annotations

from openagent_system_tool_api import SystemTool

SYSTEM_TOOL = SystemTool(
    tool_class="code",
    name="read_docs",
    handler="handle",
    parallel_safe=True,
    docs={"desc": "Read bundled/remote MCUB API documentation."},
)


async def handle(
    agent,
    tool_name: str,
    attrs_raw: str = "",
    body: str = "",
    source_event=None,
) -> str:
    return await agent._fetch_mcub_docs()
