from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import cubkit
import pytest

from command_testkit import FakeCommandEvent

SRC = Path(__file__).resolve().parents[1] / "Src"
sys.path.insert(0, str(SRC))
src_package = ModuleType("Src")
src_package.__path__ = [str(SRC)]
sys.modules.setdefault("Src", src_package)
original_load_strings = cubkit.load_strings
cubkit.load_strings = lambda: (lambda key, **_kwargs: key)
try:
    openagent_main = importlib.import_module("Src.OpenAgentMain")
finally:
    cubkit.load_strings = original_load_strings
OpenAgent = openagent_main.OpenAgent


class _DebugHarness:
    DEBUG = False
    _parse_oa_debug_tool_request = staticmethod(OpenAgent._parse_oa_debug_tool_request)

    def __init__(
        self, output: str = '{"status":"success","output":{"text":"ok"}}'
    ) -> None:
        self.output = output
        self.dispatches = []
        self.edits = []

    async def _dispatch_agent_tool_batch(self, tool_calls, **kwargs):
        self.dispatches.append((tool_calls, kwargs))
        return [self.output]

    async def edit(self, event, text, **kwargs):
        self.edits.append((event, text, kwargs))
        return event


class _CommandHarness:
    _oa_prompt_from_parser = OpenAgent._oa_prompt_from_parser
    _oa_new_chat_arg = OpenAgent._oa_new_chat_arg
    _oa_test_name = OpenAgent._oa_test_name
    _oa_flash_arg = OpenAgent._oa_flash_arg
    _oa_debug_tool_arg = staticmethod(OpenAgent._oa_debug_tool_arg)

    def __init__(self, raw_args: str) -> None:
        self.parser = SimpleNamespace(raw_args=raw_args)
        self.debug_requests = []

    def _oa_arg_parser(self, _event):
        return self.parser

    @staticmethod
    def _args_raw(_event) -> str:
        return ""

    async def _run_oa_debug_tool(self, event, value: str) -> None:
        self.debug_requests.append((event, value))


def test_debug_tool_parser_preserves_exact_json_arguments() -> None:
    parser = SimpleNamespace(
        raw_args='--debug=tool terminal.run {"argv":["pwd"],"cwd":"."}'
    )

    request = OpenAgent._oa_debug_tool_arg(parser)
    tool_name, arguments = OpenAgent._parse_oa_debug_tool_request(request or "")

    assert tool_name == "terminal.run"
    assert arguments == {"argv": ["pwd"], "cwd": "."}


@pytest.mark.parametrize(
    "raw_request,error",
    [
        ("terminal.run", "Usage:"),
        ("terminal.run []", "JSON object"),
        ("terminal.run {broken}", "Invalid tool arguments JSON"),
    ],
)
def test_debug_tool_parser_rejects_invalid_requests(
    raw_request: str, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        OpenAgent._parse_oa_debug_tool_request(raw_request)


@pytest.mark.asyncio
async def test_cmd_oa_intercepts_debug_tool_before_agent_loop() -> None:
    event = FakeCommandEvent('.oa --debug=tool terminal.run {"argv":["pwd"],"cwd":"."}')
    harness = _CommandHarness('--debug=tool terminal.run {"argv":["pwd"],"cwd":"."}')

    await OpenAgent.cmd_oa(harness, event)

    assert harness.debug_requests == [
        (event, 'terminal.run {"argv":["pwd"],"cwd":"."}')
    ]


@pytest.mark.asyncio
async def test_debug_build_dispatches_one_v2_tool() -> None:
    harness = _DebugHarness()
    harness.DEBUG = True
    event = FakeCommandEvent(".oa --debug=tool terminal.run")

    await OpenAgent._run_oa_debug_tool(
        harness,
        event,
        'terminal.run {"argv":["pwd"],"cwd":"."}',
    )

    assert len(harness.dispatches) == 1
    calls, kwargs = harness.dispatches[0]
    assert len(calls) == 1 and calls[0][0] == "terminal.run"
    assert "argv=" in calls[0][1] and 'cwd="."' in calls[0][1]
    assert kwargs["source_event"] is event
    assert kwargs["status_event"] is event
    assert kwargs["cancel_token"].startswith("debug-tool-")
    assert json.loads(harness.output)["status"] == "success"
    assert "success" in harness.edits[-1][1]


@pytest.mark.asyncio
async def test_release_build_rejects_debug_tool_without_dispatch() -> None:
    harness = _DebugHarness()
    event = FakeCommandEvent(".oa --debug=tool terminal.run")

    await OpenAgent._run_oa_debug_tool(
        harness,
        event,
        'terminal.run {"argv":["pwd"],"cwd":"."}',
    )

    assert harness.dispatches == []
    assert "unavailable in release builds" in harness.edits[-1][1]
