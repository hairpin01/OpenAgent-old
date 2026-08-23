from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from OpenAgentLib.Plugin.PluginsEngine import (
    ProviderResponse,
    ProviderToolCall,
    ProviderToolTurn,
    _OpenAgentAgentLoopMixin,
)
from OpenAgentLib.ToolKernel import (
    ToolError,
    ToolErrorCode,
    ToolResult,
    ToolResultStatus,
)


class _ResponseLoopHarness(_OpenAgentAgentLoopMixin):
    AGENT_MAX_STEPS = 12
    DEBUG = False
    PROVIDERS = {"openai"}

    def __init__(self, responses: list[str | ProviderResponse]) -> None:
        self.responses = list(responses)
        self.provider_calls: list[list[dict[str, Any]]] = []
        self.provider_native_requirements: list[tuple[bool, bool]] = []
        self.dispatched_tool_calls: list[list[tuple[str, str, str]]] = []
        self.tool_outputs = ['{"status":"success"}']
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
    def _extract_tool_calls(answer: str) -> list[tuple[str, str, str]]:
        if '"tool":"utility.search_tool"' in answer:
            return [("utility.search_tool", "", "")]
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
        allow_native_tools: bool = False,
        require_native_tools: bool = False,
    ) -> str | ProviderResponse:
        del status_event, agent_log, started_at, thinking_notes, max_tokens_override
        before_attempt()
        self.provider_calls.append(list(messages))
        self.provider_native_requirements.append(
            (allow_native_tools, require_native_tools)
        )
        assert self.responses, "unexpected provider call"
        return self.responses.pop(0)

    async def _dispatch_agent_tool_batch(
        self, tool_calls: list[tuple[str, str, str]], **_kwargs: Any
    ) -> list[str]:
        self.dispatched_tool_calls.append(tool_calls)
        return list(self.tool_outputs)

    @staticmethod
    def _parse_xml_attrs(_attrs: str) -> dict[str, str]:
        return {}

    @staticmethod
    def _remember_tool_output(_chat_id: int, _tool: str, _output: str) -> None:
        return None

    @staticmethod
    def _format_tool_call_for_context(
        _chat_id: int, _tool: str, _attrs: str, _body: str, output: str
    ) -> str:
        return output

    @staticmethod
    def strings(key: str) -> str:
        return f"<{key}>"


class _NativeResponseLoopHarness(_ResponseLoopHarness):
    def __init__(self, responses: list[str | ProviderResponse]) -> None:
        super().__init__(responses)
        self.rendered_tool_turns: list[ProviderToolTurn] = []

    def _render_provider_tool_results(
        self,
        tool_turn: ProviderToolTurn,
        raw_outputs: list[str],
    ) -> list[dict[str, Any]] | None:
        self.rendered_tool_turns.append(tool_turn)
        return [
            {
                "role": "assistant",
                "content": "",
                "native_assistant_turn": tool_turn.native_assistant_turn,
            },
            {
                "role": "tool",
                "tool_call_id": tool_turn.calls[0].call_id,
                "content": raw_outputs[0],
            },
        ]


def _is_completion_gate(call: list[dict[str, Any]]) -> bool:
    return any(
        message.get("role") == "system"
        and "completion gate" in str(message.get("content", ""))
        for message in call
    )


def test_explicit_final_returns_without_completion_gate() -> None:
    async def scenario() -> None:
        harness = _ResponseLoopHarness(["```final\nHello.\n```"])

        answer, agent_log, thinking_notes, _trace = await harness._ask_agent(
            "Say hello"
        )

        assert answer == "Hello."
        assert thinking_notes == []
        assert agent_log == ["answer.accepted"]
        assert len(harness.provider_calls) == 1
        assert not any(_is_completion_gate(call) for call in harness.provider_calls)

    asyncio.run(scenario())


def test_plain_direct_answer_uses_gate_without_thinking_notes() -> None:
    async def scenario() -> None:
        harness = _ResponseLoopHarness(["Hello!", "ACCEPT"])

        answer, agent_log, thinking_notes, _trace = await harness._ask_agent(
            "Say hello"
        )

        assert answer == "Hello!"
        assert thinking_notes == []
        assert agent_log == ["answer.accepted"]
        assert len(harness.provider_calls) == 2
        assert _is_completion_gate(harness.provider_calls[1])

    asyncio.run(scenario())


def test_rejected_plain_answer_requires_native_tools_then_resets_after_batch() -> None:
    async def scenario() -> None:
        harness = _ResponseLoopHarness(
            [
                "I do not have access.",
                "CONTINUE",
                '```tool_call\n{"tool":"utility.search_tool","args":{}}\n```',
                "```final\nCompleted.\n```",
            ]
        )

        answer, _agent_log, thinking_notes, _trace = await harness._ask_agent(
            "Inspect the workspace"
        )

        assert answer == "Completed."
        assert thinking_notes == []
        assert harness.provider_native_requirements == [
            (True, False),
            (False, False),
            (True, True),
            (True, False),
        ]
        assert harness.dispatched_tool_calls == [[("utility.search_tool", "", "")]]
        assert _is_completion_gate(harness.provider_calls[1])
        assert not _is_completion_gate(harness.provider_calls[2])
        assert not _is_completion_gate(harness.provider_calls[3])

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

        answer, agent_log, thinking_notes, _trace = await harness._ask_agent(
            "Выполни задачу"
        )

        assert answer == "Готово."
        assert thinking_notes == []
        assert "thinking.model_progress" not in agent_log
        assert "router.action" not in agent_log
        assert len(harness.provider_calls) == 5
        assert sum(_is_completion_gate(call) for call in harness.provider_calls) == 2

    asyncio.run(scenario())


def test_legacy_tool_result_reinjection_uses_generic_messages() -> None:
    async def scenario() -> None:
        tool_call = '```tool_call\n{"tool":"utility.search_tool","args":{}}\n```'
        harness = _ResponseLoopHarness([tool_call, "```final\nCompleted.\n```"])

        answer, _agent_log, _thinking_notes, _trace = await harness._ask_agent(
            "Search the workspace"
        )

        assert answer == "Completed."
        followup_messages = harness.provider_calls[1]
        assert followup_messages[-2] == {"role": "assistant", "content": tool_call}
        assert followup_messages[-1]["role"] == "user"
        assert followup_messages[-1]["content"].startswith('{"status":"success"}')
        assert "Progress reminder:" in followup_messages[-1]["content"]

    asyncio.run(scenario())


def test_native_tool_turn_uses_renderer_and_preserves_opaque_call_id() -> None:
    async def scenario() -> None:
        tool_turn = ProviderToolTurn(
            calls=(
                ProviderToolCall(
                    provider_kind="openai-responses",
                    call_id="call:opaque/1",
                    tool_name="utility.search_tool",
                    arguments={"query": "workspace"},
                    raw_arguments='{"query":"workspace"}',
                ),
            ),
            native_assistant_turn={
                "id": "assistant-turn-1",
                "output": [{"type": "function_call"}],
            },
        )
        harness = _NativeResponseLoopHarness(
            [
                ProviderResponse(content="", tool_turn=tool_turn),
                "```final\nCompleted.\n```",
            ]
        )

        answer, _agent_log, _thinking_notes, _trace = await harness._ask_agent(
            "Search the workspace"
        )

        assert answer == "Completed."
        assert harness.rendered_tool_turns == [tool_turn]
        assert harness.provider_calls[1][-2:] == [
            {
                "role": "assistant",
                "content": "",
                "native_assistant_turn": tool_turn.native_assistant_turn,
            },
            {
                "role": "tool",
                "tool_call_id": "call:opaque/1",
                "content": '{"status":"success"}',
            },
        ]

    asyncio.run(scenario())


def test_openai_chat_tool_turn_executes_one_batch_and_renders_tool_message() -> None:
    async def scenario() -> None:
        assistant = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "chat-call/opaque:1",
                    "type": "function",
                    "function": {
                        "name": "utility_search_native",
                        "arguments": '{"query":"workspace"}',
                    },
                }
            ],
        }
        tool_turn = ProviderToolTurn(
            calls=(
                ProviderToolCall(
                    provider_kind="openai-chat",
                    call_id="chat-call/opaque:1",
                    tool_name="utility.search_tool",
                    arguments={"query": "workspace"},
                    raw_arguments='{"query":"workspace"}',
                ),
            ),
            native_assistant_turn=assistant,
        )
        harness = _ResponseLoopHarness(
            [ProviderResponse(content="", tool_turn=tool_turn), "```final\nDone.\n```"]
        )

        answer, _agent_log, _thinking_notes, _trace = await harness._ask_agent(
            "Search the workspace"
        )

        assert answer == "Done."
        assert len(harness.provider_calls) == 2
        assert len(harness.dispatched_tool_calls) == 1
        assert harness.provider_calls[1][-2:] == [
            assistant,
            {
                "role": "tool",
                "tool_call_id": "chat-call/opaque:1",
                "content": '{"status":"success"}',
            },
        ]

    asyncio.run(scenario())


def test_openai_responses_tool_turn_executes_one_batch_and_renders_output() -> None:
    async def scenario() -> None:
        function_call = {
            "type": "function_call",
            "id": "item_not_for_correlation",
            "call_id": "response-call/opaque:1",
            "name": "utility_search_native",
            "arguments": '{"query":"workspace"}',
        }
        tool_turn = ProviderToolTurn(
            calls=(
                ProviderToolCall(
                    provider_kind="openai-responses",
                    call_id="response-call/opaque:1",
                    tool_name="utility.search_tool",
                    arguments={"query": "workspace"},
                    raw_arguments='{"query":"workspace"}',
                ),
            ),
            native_assistant_turn=[function_call],
        )
        harness = _ResponseLoopHarness(
            [ProviderResponse(content="", tool_turn=tool_turn), "```final\nDone.\n```"]
        )

        answer, _agent_log, _thinking_notes, _trace = await harness._ask_agent(
            "Search the workspace"
        )

        assert answer == "Done."
        assert len(harness.provider_calls) == 2
        assert len(harness.dispatched_tool_calls) == 1
        assert harness.provider_calls[1][-2:] == [
            function_call,
            {
                "type": "function_call_output",
                "call_id": "response-call/opaque:1",
                "output": '{"status":"success"}',
            },
        ]

    asyncio.run(scenario())


def test_anthropic_tool_turn_executes_once_and_reaches_final_text() -> None:
    async def scenario() -> None:
        assistant = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_opaque/1",
                    "name": "utility_search_native",
                    "input": {"query": "workspace"},
                }
            ],
        }
        tool_turn = ProviderToolTurn(
            calls=(
                ProviderToolCall(
                    provider_kind="anthropic-messages",
                    call_id="toolu_opaque/1",
                    tool_name="utility.search_tool",
                    arguments={"query": "workspace"},
                    raw_arguments={"query": "workspace"},
                ),
            ),
            native_assistant_turn=assistant,
        )
        harness = _ResponseLoopHarness(
            [ProviderResponse(content="", tool_turn=tool_turn), "```final\nDone.\n```"]
        )

        answer, _agent_log, _thinking_notes, _trace = await harness._ask_agent(
            "Search the workspace"
        )

        assert answer == "Done."
        assert len(harness.provider_calls) == 2
        assert len(harness.dispatched_tool_calls) == 1
        assert harness.provider_calls[1][-2:] == [
            assistant,
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_opaque/1",
                        "content": '{"status": "success"}',
                    }
                ],
            },
        ]

    asyncio.run(scenario())


def test_gemini_tool_turn_executes_once_and_reaches_final_text() -> None:
    async def scenario() -> None:
        assistant = {
            "role": "model",
            "parts": [
                {
                    "functionCall": {
                        "id": "gemini-opaque/call:1",
                        "name": "utility_search_native",
                        "args": {"query": "workspace"},
                    }
                }
            ],
        }
        tool_turn = ProviderToolTurn(
            calls=(
                ProviderToolCall(
                    provider_kind="google-gemini",
                    call_id="gemini-opaque/call:1",
                    tool_name="utility.search_tool",
                    arguments={"query": "workspace"},
                    raw_arguments={"query": "workspace"},
                ),
            ),
            native_assistant_turn=assistant,
        )
        harness = _ResponseLoopHarness(
            [ProviderResponse(content="", tool_turn=tool_turn), "```final\nDone.\n```"]
        )

        answer, _agent_log, _thinking_notes, _trace = await harness._ask_agent(
            "Search the workspace"
        )

        assert answer == "Done."
        assert len(harness.provider_calls) == 2
        assert len(harness.dispatched_tool_calls) == 1
        assert harness.provider_calls[1][-2:] == [
            assistant,
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "id": "gemini-opaque/call:1",
                            "name": "utility_search_native",
                            "response": {"status": "success"},
                        }
                    }
                ],
            },
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "provider_kind",
    ("openai-chat", "openai-responses", "google-gemini", "anthropic-messages"),
)
@pytest.mark.parametrize(
    ("tool_name", "error_code", "safe_message"),
    (
        (
            "utility.search_tool",
            ToolErrorCode.POLICY_DENIED,
            "tool execution was denied by policy",
        ),
        ("utility.search_tool", ToolErrorCode.HANDLER_FAILED, "tool handler failed"),
        (
            "terminal.run",
            ToolErrorCode.HOST_FAILED,
            "isolated host execution failed",
        ),
    ),
)
def test_native_tool_error_results_are_rendered_then_allow_final_text(
    provider_kind: str,
    tool_name: str,
    error_code: ToolErrorCode,
    safe_message: str,
) -> None:
    async def scenario() -> None:
        call_id = f"{provider_kind}:{error_code.value}"
        native_name = f"{tool_name.replace('.', '_')}_native"
        if provider_kind == "openai-chat":
            assistant: Any = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": native_name,
                            "arguments": '{"query":"workspace"}',
                        },
                    }
                ],
            }
            raw_arguments: Any = '{"query":"workspace"}'
        elif provider_kind == "openai-responses":
            assistant = [
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": native_name,
                    "arguments": '{"query":"workspace"}',
                }
            ]
            raw_arguments = '{"query":"workspace"}'
        elif provider_kind == "google-gemini":
            assistant = {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "id": call_id,
                            "name": native_name,
                            "args": {"query": "workspace"},
                        }
                    }
                ],
            }
            raw_arguments = {"query": "workspace"}
        else:
            assistant = {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": native_name,
                        "input": {"query": "workspace"},
                    }
                ],
            }
            raw_arguments = {"query": "workspace"}

        tool_turn = ProviderToolTurn(
            calls=(
                ProviderToolCall(
                    provider_kind=provider_kind,
                    call_id=call_id,
                    tool_name=tool_name,
                    arguments={"query": "workspace"},
                    raw_arguments=raw_arguments,
                ),
            ),
            native_assistant_turn=assistant,
        )
        result = ToolResult(
            call_id,
            ToolResultStatus.ERROR,
            error=ToolError(
                error_code,
                "Traceback from /private/path command=secret API_TOKEN=hidden",
                correlation_id=f"diagnostic-{error_code.value}",
            ),
        )
        harness = _ResponseLoopHarness(
            [
                ProviderResponse(content="", tool_turn=tool_turn),
                "```final\nRecovered.\n```",
            ]
        )
        harness.tool_outputs = [harness._v2_result_text(result)]

        answer, _agent_log, _thinking_notes, _trace = await harness._ask_agent(
            "Search the workspace"
        )

        assert answer == "Recovered."
        assert len(harness.provider_calls) == 2
        assert len(harness.dispatched_tool_calls) == 1
        assert [call[0] for call in harness.dispatched_tool_calls[0]] == [tool_name]
        expected_result = {
            "call_id": call_id,
            "error": {
                "code": error_code.value,
                "correlation_id": f"diagnostic-{error_code.value}",
                "message": safe_message,
            },
            "output": None,
            "retryable": False,
            "status": "error",
        }
        followup = harness.provider_calls[1][-1]
        if provider_kind == "openai-chat":
            assert followup["tool_call_id"] == call_id
            assert json.loads(followup["content"]) == expected_result
        elif provider_kind == "openai-responses":
            assert followup["call_id"] == call_id
            assert json.loads(followup["output"]) == expected_result
        elif provider_kind == "google-gemini":
            response = followup["parts"][0]["functionResponse"]
            assert response["id"] == call_id
            assert response["name"] == native_name
            assert response["response"] == expected_result
        else:
            result_block = followup["content"][0]
            assert result_block["tool_use_id"] == call_id
            assert result_block["is_error"] is True
            assert json.loads(result_block["content"]) == expected_result

    asyncio.run(scenario())


def test_forced_final_retries_malformed_output_without_gate() -> None:
    async def scenario() -> None:
        harness = _ResponseLoopHarness(["Почти готово.", "```final\nГотово.\n```"])
        harness.AGENT_MAX_STEPS = 0

        answer, _agent_log, thinking_notes, _trace = await harness._ask_agent(
            "Выполни задачу"
        )

        assert answer == "Готово."
        assert thinking_notes == []
        assert len(harness.provider_calls) == 2
        assert not any(_is_completion_gate(call) for call in harness.provider_calls)
        correction = harness.provider_calls[1][-1]
        assert correction["role"] == "user"
        assert "one valid ```final``` block" in correction["content"]

    asyncio.run(scenario())
