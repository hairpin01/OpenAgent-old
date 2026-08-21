from __future__ import annotations

import asyncio
from typing import Any

from OpenAgentLib.Plugin.PluginsEngine import _OpenAgentAgentLoopMixin


class _ResponseLoopHarness(_OpenAgentAgentLoopMixin):
    AGENT_MAX_STEPS = 12
    DEBUG = False
    PROVIDERS = {"openai"}

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.provider_calls: list[list[dict[str, Any]]] = []
        self._cancelled_generations: set[str] = set()
        self._last_token_usage: dict[str, int] = {}
        self.config = {
            "agent_max_model_calls": 10,
            "agent_deadline": 30,
            "agent_max_steps": 6,
            "context_window_tokens": 16000,
            "context_reserve_tokens": 2400,
            "max_tokens": 800,
        }

    def _provider(self) -> str:
        return "openai"

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        return provider

    @staticmethod
    def _api_key() -> str:
        return "test-key"

    @staticmethod
    def _model(_provider: str) -> str:
        return "test-model"

    @staticmethod
    def _build_openai_content(prompt: str, _attachments: list[dict[str, str]]) -> str:
        return prompt

    @staticmethod
    def _build_google_content(prompt: str, _attachments: list[dict[str, str]]) -> str:
        return prompt

    @staticmethod
    def _event_chat_id(_event: object | None) -> int:
        return 0

    async def _compact_chat_history_if_needed(self, *_args: object) -> bool:
        return False

    @staticmethod
    def _history_for_chat(_chat_id: int) -> list[dict[str, str]]:
        return []

    @staticmethod
    def _tool_memory_prompt(_chat_id: int) -> str:
        return ""

    @staticmethod
    def _system_prompt(_prompt: str, *, flash_mode: bool) -> str:
        del flash_mode
        return "test system prompt"

    async def _run_plugin_hooks(self, *_args: object) -> None:
        return None

    @staticmethod
    def _runtime_comment_message(_cancel_token: str | None) -> None:
        return None

    @staticmethod
    def _extract_tool_calls(_answer: str) -> list[tuple[str, str, str]]:
        return []

    @staticmethod
    def _invalid_tool_call_error(_answer: str) -> None:
        return None

    async def _ask_provider_with_reconnect(
        self,
        _provider: str,
        messages: list[dict[str, Any]],
        _api_key: str,
        *,
        status_event: object | None,
        agent_log: list[str],
        started_at: float | None,
        thinking_notes: list[str],
        max_tokens_override: int | None,
        before_attempt: Any,
    ) -> str:
        del status_event, agent_log, started_at, thinking_notes, max_tokens_override
        before_attempt()
        self.provider_calls.append(list(messages))
        assert self.responses, "unexpected provider call"
        return self.responses.pop(0)

    @staticmethod
    def strings(key: str) -> str:
        return f"<{key}>"


def _is_completion_gate(call: list[dict[str, Any]]) -> bool:
    return any(
        message.get("role") == "system"
        and "completion gate" in str(message.get("content", ""))
        for message in call
    )


def test_explicit_final_returns_without_completion_gate() -> None:
    async def scenario() -> None:
        harness = _ResponseLoopHarness(["```final\nHello.\n```"])

        answer, agent_log, thinking_notes, _trace = await harness._ask_agent("Say hello")

        assert answer == "Hello."
        assert thinking_notes == []
        assert agent_log == ["answer.accepted"]
        assert len(harness.provider_calls) == 1
        assert not any(_is_completion_gate(call) for call in harness.provider_calls)

    asyncio.run(scenario())


def test_plain_direct_answer_uses_gate_without_thinking_notes() -> None:
    async def scenario() -> None:
        harness = _ResponseLoopHarness(["Hello!", "ACCEPT"])

        answer, agent_log, thinking_notes, _trace = await harness._ask_agent("Say hello")

        assert answer == "Hello!"
        assert thinking_notes == []
        assert agent_log == ["answer.accepted"]
        assert len(harness.provider_calls) == 2
        assert _is_completion_gate(harness.provider_calls[1])

    asyncio.run(scenario())


def test_rejected_acknowledgements_are_not_journaled_before_final() -> None:
    async def scenario() -> None:
        acknowledgement = ".... поняла.... шмелька...."
        harness = _ResponseLoopHarness(
            [
                acknowledgement,
                "CONTINUE",
                acknowledgement,
                "CONTINUE",
                "```final\nГотово.\n```",
            ]
        )

        answer, agent_log, thinking_notes, _trace = await harness._ask_agent("Выполни задачу")

        assert answer == "Готово."
        assert thinking_notes == []
        assert "thinking.model_progress" not in agent_log
        assert "router.action" not in agent_log
        assert len(harness.provider_calls) == 5
        assert sum(_is_completion_gate(call) for call in harness.provider_calls) == 2

    asyncio.run(scenario())


def test_forced_final_retries_malformed_output_without_gate() -> None:
    async def scenario() -> None:
        harness = _ResponseLoopHarness(["Почти готово.", "```final\nГотово.\n```"])
        harness.AGENT_MAX_STEPS = 0

        answer, _agent_log, thinking_notes, _trace = await harness._ask_agent("Выполни задачу")

        assert answer == "Готово."
        assert thinking_notes == []
        assert len(harness.provider_calls) == 2
        assert not any(_is_completion_gate(call) for call in harness.provider_calls)
        correction = harness.provider_calls[1][-1]
        assert correction["role"] == "user"
        assert "one valid ```final``` block" in correction["content"]

    asyncio.run(scenario())
