from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import cubkit
import pytest

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


class _Posted:
    """Stands for the message `event.reply()` posts for a guest query."""

    def __init__(self) -> None:
        self.edits: list[tuple[str, dict]] = []

    async def edit(self, text, **kwargs):
        self.edits.append((text, kwargs))
        return self


class _GuestEvent:
    """Minimal stand-in for a Telethon GuestMessage query event."""

    is_query = True

    def __init__(self, guest_args: str = "") -> None:
        self.guest_args = guest_args
        self.replies: list[tuple[str, dict]] = []
        self.posted = _Posted()

    async def reply(self, text="", **kwargs):
        self.replies.append((text, kwargs))
        return self.posted


class _PlainEvent:
    is_query = False

    def __init__(self) -> None:
        self.edits: list[tuple[str, dict]] = []

    async def edit(self, text, **kwargs):
        self.edits.append((text, kwargs))
        return self


class _StatusHarness:
    _oa_start_status = OpenAgent._oa_start_status
    _oa_ack = OpenAgent._oa_ack

    class log:  # noqa: N801 - test double
        @staticmethod
        def debug(*_args, **_kwargs):
            return None

    def __init__(self) -> None:
        self.inline_status_calls: list[tuple] = []

    async def _start_inline_status(self, event, text, buttons):
        self.inline_status_calls.append((event, text, buttons))
        return _Posted()

    async def edit(self, event, text, **kwargs):
        await event.edit(text, **kwargs)


class _ParserHarness:
    _oa_arg_parser = OpenAgent._oa_arg_parser
    _oa_flash_arg = OpenAgent._oa_flash_arg
    _oa_new_chat_arg = OpenAgent._oa_new_chat_arg
    _oa_prompt_from_parser = OpenAgent._oa_prompt_from_parser

    @staticmethod
    def get_prefix() -> str:
        return "."

    def args(self, event):
        import utils

        return utils.parse_arguments(event.raw_text, prefix=self.get_prefix())


def test_guest_command_is_registered() -> None:
    patterns = {pattern for pattern, _func, _meta in OpenAgent._guest_cmd_registry}
    assert "oa" in patterns


@pytest.mark.parametrize(
    "guest_args,expected_raw",
    [
        ("привет", "привет"),
        ("привет --flash", "привет --flash"),
        ("привет --new=Работа", "привет --new=Работа"),
    ],
)
def test_guest_arg_parser_reads_prefixless_query(guest_args, expected_raw) -> None:
    event = _GuestEvent(guest_args)
    parser = _ParserHarness()._oa_arg_parser(event)

    assert parser is not None
    assert parser.raw_args == expected_raw
    assert parser.command == "oa"


def test_guest_arg_parser_keeps_flags_working() -> None:
    harness = _ParserHarness()
    parser = harness._oa_arg_parser(_GuestEvent("привет --flash --new=Работа"))

    assert harness._oa_flash_arg(parser) is True
    assert harness._oa_new_chat_arg(parser) == (True, "Работа")
    assert harness._oa_prompt_from_parser(parser) == "привет"


def test_guest_arg_parser_is_ignored_without_guest_args() -> None:
    assert _ParserHarness()._oa_arg_parser(_PlainEvent()) is None


@pytest.mark.asyncio
async def test_guest_status_posts_with_event_reply() -> None:
    harness, event = _StatusHarness(), _GuestEvent()
    buttons = [["stop"]]

    posted = await harness._oa_start_status(event, "thinking", buttons)

    assert posted is event.posted
    assert event.replies == [("thinking", {"parse_mode": "html", "buttons": buttons})]
    assert posted._openagent_status_buttons == buttons
    assert harness.inline_status_calls == []


@pytest.mark.asyncio
async def test_regular_status_still_uses_inline_form() -> None:
    harness, event = _StatusHarness(), _PlainEvent()

    posted = await harness._oa_start_status(event, "thinking", [["stop"]])

    assert harness.inline_status_calls == [(event, "thinking", [["stop"]])]
    assert posted is not event


@pytest.mark.asyncio
async def test_guest_status_falls_back_to_inline_when_query_answered() -> None:
    """An already answered query returns None, so the status falls back to inline."""
    harness, event = _StatusHarness(), _GuestEvent()

    async def _already_answered(*_args, **_kwargs):
        return None

    event.reply = _already_answered
    posted = await harness._oa_start_status(event, "thinking", [["stop"]])

    assert posted is not event.posted
    assert harness.inline_status_calls == [(event, "thinking", [["stop"]])]


@pytest.mark.asyncio
async def test_ack_guest_posts_instead_of_editing() -> None:
    harness, event = _StatusHarness(), _GuestEvent()

    await harness._oa_ack(event, "готово")

    assert event.replies == [("готово", {"parse_mode": "html"})]
    assert event.posted.edits == []


@pytest.mark.asyncio
async def test_ack_regular_edits_source_message() -> None:
    harness, event = _StatusHarness(), _PlainEvent()

    await harness._oa_ack(event, "готово")

    assert event.edits == [("готово", {"as_html": True})]