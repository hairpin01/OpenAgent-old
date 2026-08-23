# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace
from typing import Any

import pytest

import OpenAgentLib.NativeToolCalls as native_tools
from OpenAgentLib.NativeToolCalls import (
    build_native_tool_catalog,
    native_function_name,
    native_response_to_fences,
    native_tool_call_error,
)
from OpenAgentLib.Plugin.PluginsEngine import (
    ProviderResponse,
    _OpenAgentAgentLoopMixin,
    _OpenAgentStatusMixin,
)
from OpenAgentLib.ToolKernel import (
    MigrationDisposition,
    ToolRegistry,
    ToolResult,
    ToolResultStatus,
)
from tool_testkit import build_tool_spec


def _specs(count: int = 3):
    result = [
        build_tool_spec("utility.search_tool", aliases=()),
        build_tool_spec("file.read_text", aliases=()),
        build_tool_spec("terminal.run", aliases=()),
    ]
    result.extend(
        build_tool_spec(f"plugin.tool_{index:03d}", aliases=())
        for index in range(count)
    )
    return result


def test_native_names_are_safe_stable_and_roundtrip() -> None:
    catalog = build_native_tool_catalog(
        _specs(), "inspect /home/alina/test/desktop-fly/"
    )
    assert all(len(name) <= 64 for name in catalog.native_to_canonical)
    assert all(
        name.replace("_", "").replace("-", "").isalnum()
        for name in catalog.native_to_canonical
    )
    for native, canonical in catalog.native_to_canonical.items():
        assert catalog.canonical_to_native[canonical] == native
        assert native == native_function_name(canonical)


def test_invalid_generated_native_name_raises_value_error(monkeypatch) -> None:
    monkeypatch.setattr(native_tools, "_SAFE_NAME_RE", re.compile(r"never-matches"))

    with pytest.raises(ValueError, match="generated native function name is invalid"):
        native_function_name("file.read_text")


def test_catalog_cap_schema_and_path_relevance() -> None:
    specs = _specs(140)
    catalog = build_native_tool_catalog(specs, "/home/alina/test/desktop-fly/", cap=4)
    canonical = set(catalog.canonical_to_native)
    assert len(catalog.tools) == 4
    assert "utility.search_tool" in canonical
    assert {"file.read_text", "terminal.run"} <= canonical
    parameters = catalog.tools[0]["function"]["parameters"]
    assert isinstance(parameters, dict)
    assert parameters == json.loads(json.dumps(parameters))


def test_rejected_specs_are_never_advertised() -> None:
    rejected = build_tool_spec(
        "file.rejected", aliases=(), migration_disposition=MigrationDisposition.REJECT
    )
    catalog = build_native_tool_catalog([*_specs(), rejected], "read a file")
    assert "file.rejected" not in catalog.canonical_to_native


def test_native_response_preserves_multiple_calls_and_null_content() -> None:
    catalog = build_native_tool_catalog(_specs(), "read a file")
    read_name = catalog.canonical_to_native["file.read_text"]
    search_name = catalog.canonical_to_native["utility.search_tool"]
    result = native_response_to_fences(
        {
            "content": None,
            "tool_calls": [
                {
                    "id": "one",
                    "function": {"name": read_name, "arguments": '{"value":"x"}'},
                },
                {"id": "two", "function": {"name": search_name, "arguments": "{}"}},
            ],
        },
        catalog,
    )
    assert result.count("```tool_call") == 2
    assert '"provider_call_id": "one"' in result
    assert '"tool": "file.read_text"' in result


def test_native_response_rejects_invalid_calls_without_fences() -> None:
    catalog = build_native_tool_catalog(_specs(), "read a file")
    name = catalog.canonical_to_native["file.read_text"]
    for call in (
        {"function": {"name": name, "arguments": "{}"}},
        {"id": "one", "function": {"name": "not_advertised", "arguments": "{}"}},
        {"id": "one", "function": {"name": name, "arguments": "{"}},
        {"id": "one", "function": {"name": name, "arguments": '{"undeclared":1}'}},
    ):
        result = native_response_to_fences(
            {"content": None, "tool_calls": [call]}, catalog
        )
        assert native_tool_call_error(result)
        assert "```tool_call" not in result
    duplicate = native_response_to_fences(
        {
            "content": None,
            "tool_calls": [
                {"id": "same", "function": {"name": name, "arguments": "{}"}},
                {"id": "same", "function": {"name": name, "arguments": "{}"}},
            ],
        },
        catalog,
    )
    assert native_tool_call_error(duplicate)


class _OpenAIHarness(_OpenAgentAgentLoopMixin):
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.urls: list[str] = []
        self.payloads: list[dict[str, Any]] = []
        self.headers: list[dict[str, str] | None] = []
        self.config = {"temperature": 0, "max_tokens": 100, "timeout": 1}
        self._v2_runtime = SimpleNamespace(registry=ToolRegistry(_specs()))

    @staticmethod
    def _base_url(_provider: str) -> str:
        return "https://provider.example/v1"

    @staticmethod
    def _model(_provider: str) -> str:
        return "test-model"

    @staticmethod
    def _uses_completion_tokens(_provider: str) -> bool:
        return False

    @staticmethod
    def _reasoning_effort() -> str:
        return "off"

    def _set_token_usage(self, _usage: object, _provider: str) -> None:
        return None

    async def _post_json(
        self, url: str, payload: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        self.urls.append(url)
        self.payloads.append(dict(payload))
        self.headers.append(kwargs.get("headers"))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item  # type: ignore[return-value]


class _GoogleHarness(_OpenAIHarness):
    @staticmethod
    def _build_google_parts(
        content: str | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"text": content}]
        return content


class _AnthropicHarness(_OpenAIHarness):
    @staticmethod
    def _base_url(_provider: str) -> str:
        return "https://provider.example"


def test_anthropic_messages_tool_use_round_trip_preserves_id_and_blocks() -> None:
    async def scenario() -> None:
        catalog = build_native_tool_catalog(_specs(), "read /tmp/a")
        name = catalog.canonical_to_native["file.read_text"]
        tool_use = {
            "type": "tool_use",
            "id": "toolu_opaque/1",
            "name": name,
            "input": {"value": "/tmp/a"},
        }
        assistant_blocks = [
            {"type": "text", "text": "I will read it."},
            tool_use,
        ]
        harness = _AnthropicHarness(
            [
                {
                    "role": "assistant",
                    "content": assistant_blocks,
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 9, "output_tokens": 4},
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "```final\nRead it.\n```"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 15, "output_tokens": 5},
                },
            ]
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "Use tools safely."},
            {"role": "user", "content": "read /tmp/a"},
        ]

        first = await harness._ask_provider_once(
            "anthropic", messages, "anthropic-key", allow_native_tools=True
        )
        assert isinstance(first, ProviderResponse)
        assert first.tool_turn is not None
        call = first.tool_turn.calls[0]
        assert call.call_id == "toolu_opaque/1"
        assert call.tool_name == "file.read_text"
        assert dict(call.arguments) == {"value": "/tmp/a"}
        assert first.tool_turn.native_assistant_turn == {
            "role": "assistant",
            "content": assistant_blocks,
        }
        assert harness.urls[0] == "https://provider.example/v1/messages"
        assert harness.headers[0] == {
            "x-api-key": "anthropic-key",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        assert harness.payloads[0]["system"] == "Use tools safely."
        assert harness.payloads[0]["stream"] is False
        assert harness.payloads[0]["messages"] == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "read /tmp/a"}],
            }
        ]
        advertised = next(
            tool for tool in harness.payloads[0]["tools"] if tool["name"] == name
        )
        assert set(advertised) == {"name", "description", "input_schema"}
        assert (
            advertised["input_schema"]
            == catalog.tools[list(catalog.canonical_to_native).index("file.read_text")][
                "function"
            ]["parameters"]
        )

        followup = harness._render_provider_tool_results(
            first.tool_turn,
            [
                '{"call_id":"local","error":null,"output":{"text":"ok"},"status":"success"}'
            ],
        )
        assert followup == [
            {"role": "assistant", "content": assistant_blocks},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_opaque/1",
                        "content": (
                            '{"call_id": "local", "error": null, '
                            '"output": {"text": "ok"}, "status": "success"}'
                        ),
                    }
                ],
            },
        ]
        final = await harness._ask_provider_once(
            "anthropic",
            [*messages, *followup],
            "anthropic-key",
            allow_native_tools=True,
        )
        assert final == "```final\nRead it.\n```"
        assert harness.payloads[1]["messages"][-2:] == followup

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "blocks",
    [
        [{"type": "tool_use", "id": "", "name": "unused", "input": {}}],
        [
            {"type": "tool_use", "id": "same", "name": "unused", "input": {}},
            {"type": "tool_use", "id": "same", "name": "unused", "input": {}},
        ],
        [{"type": "tool_use", "id": "one", "name": "unused", "input": []}],
        [
            {
                "type": "tool_use",
                "id": "one",
                "name": "unused",
                "input": {"undeclared": True},
            }
        ],
        [{"type": "tool_use", "id": "one", "name": "unknown", "input": {}}],
    ],
)
def test_anthropic_rejects_malformed_or_unregistered_tool_use(
    blocks: list[dict[str, Any]],
) -> None:
    catalog = build_native_tool_catalog(_specs(), "read /tmp/a")
    name = catalog.canonical_to_native["file.read_text"]
    for block in blocks:
        if block["name"] == "unused":
            block["name"] = name

    response = _OpenAgentAgentLoopMixin._anthropic_native_tool_response(
        content=blocks, native_catalog=catalog
    )

    assert response.tool_turn is None
    assert native_tool_call_error(response.content)


@pytest.mark.parametrize(
    "parts",
    [
        [{"functionCall": {"name": "unused", "args": {}}}],
        [
            {"functionCall": {"id": "same", "name": "unused", "args": {}}},
            {"functionCall": {"id": "same", "name": "unused", "args": {}}},
        ],
        [
            {
                "functionCall": {
                    "id": "one",
                    "name": "unused",
                    "args": {"undeclared": True},
                }
            }
        ],
    ],
    ids=("missing-id", "duplicate-id", "schema-invalid-args"),
)
def test_google_rejects_invalid_function_calls_before_dispatch(
    parts: list[dict[str, Any]],
) -> None:
    catalog = build_native_tool_catalog(_specs(), "read /tmp/a")
    name = catalog.canonical_to_native["file.read_text"]
    for part in parts:
        part["functionCall"]["name"] = name

    response = _OpenAgentAgentLoopMixin._google_native_tool_response(
        content={"role": "model", "parts": parts}, native_catalog=catalog
    )

    assert response.tool_turn is None
    assert native_tool_call_error(response.content)


def test_google_native_call_preserves_object_args_and_round_trips_result() -> None:
    async def scenario() -> None:
        catalog = build_native_tool_catalog(_specs(), "read /tmp/a")
        name = catalog.canonical_to_native["file.read_text"]
        args = {"value": "/tmp/a"}
        model_content = {
            "role": "model",
            "parts": [
                {
                    "functionCall": {
                        "id": "gemini:opaque/call-1",
                        "name": name,
                        "args": args,
                    }
                }
            ],
        }
        harness = _GoogleHarness(
            [
                {
                    "candidates": [{"content": model_content}],
                    "usageMetadata": {
                        "promptTokenCount": 7,
                        "candidatesTokenCount": 3,
                        "totalTokenCount": 10,
                    },
                },
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"text": "```final\nRead it.\n```"}],
                            }
                        }
                    ]
                },
            ]
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": "read /tmp/a"}]

        first = await harness._ask_provider_once(
            "google", messages, "key", allow_native_tools=True
        )
        assert isinstance(first, ProviderResponse)
        assert first.tool_turn is not None
        call = first.tool_turn.calls[0]
        assert call.call_id == "gemini:opaque/call-1"
        assert call.tool_name == "file.read_text"
        assert dict(call.arguments) == args
        assert call.raw_arguments is args
        assert first.tool_turn.native_assistant_turn == model_content
        followup = harness._render_provider_tool_results(
            first.tool_turn,
            [
                '{"call_id":"local","error":null,"output":{"text":"ok"},"status":"success"}'
            ],
        )
        assert followup == [
            model_content,
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "id": "gemini:opaque/call-1",
                            "name": name,
                            "response": {
                                "call_id": "local",
                                "error": None,
                                "output": {"text": "ok"},
                                "status": "success",
                            },
                        }
                    }
                ],
            },
        ]

        final = await harness._ask_provider_once(
            "google", [*messages, *followup], "key", allow_native_tools=True
        )
        assert final == "```final\nRead it.\n```"
        declarations = harness.payloads[0]["tools"][0]["functionDeclarations"]
        assert any(item["name"] == name for item in declarations)
        assert harness.payloads[1]["contents"][-2:] == followup

    asyncio.run(scenario())


def test_google_multiple_calls_pair_opaque_ids_with_success_and_failure() -> None:
    async def scenario() -> None:
        catalog = build_native_tool_catalog(_specs(), "search and read")
        search_name = catalog.canonical_to_native["utility.search_tool"]
        read_name = catalog.canonical_to_native["file.read_text"]
        content = {
            "role": "model",
            "parts": [
                {
                    "functionCall": {
                        "id": "first/id",
                        "name": search_name,
                        "args": {"value": "needle"},
                    }
                },
                {
                    "functionCall": {
                        "id": "second:id",
                        "name": read_name,
                        "args": {"value": "/tmp/a"},
                    }
                },
            ],
        }
        harness = _GoogleHarness([{"candidates": [{"content": content}]}])

        response = await harness._ask_google(
            [{"role": "user", "content": "search and read"}],
            "key",
            allow_native_tools=True,
            require_native_tools=True,
        )
        assert isinstance(response, ProviderResponse)
        assert response.tool_turn is not None
        assert [call.call_id for call in response.tool_turn.calls] == [
            "first/id",
            "second:id",
        ]
        rendered = harness._render_provider_tool_results(
            response.tool_turn,
            [
                '{"status":"success","output":{"matches":1}}',
                '{"status":"error","error":"denied","output":{}}',
            ],
        )
        assert rendered is not None
        function_responses = rendered[1]["parts"]
        assert [item["functionResponse"]["id"] for item in function_responses] == [
            "first/id",
            "second:id",
        ]
        assert [item["functionResponse"]["name"] for item in function_responses] == [
            search_name,
            read_name,
        ]
        assert function_responses[1]["functionResponse"]["response"] == {
            "status": "error",
            "error": "denied",
            "output": {},
        }
        assert harness.payloads[0]["toolConfig"] == {
            "functionCallingConfig": {"mode": "ANY"}
        }

    asyncio.run(scenario())


def test_google_idless_call_is_diagnostic_then_accepts_explicit_final() -> None:
    async def scenario() -> None:
        catalog = build_native_tool_catalog(_specs(), "read /tmp/a")
        name = catalog.canonical_to_native["file.read_text"]
        harness = _GoogleHarness(
            [
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {
                                        "functionCall": {
                                            "name": name,
                                            "args": {"value": "/tmp/a"},
                                        }
                                    }
                                ],
                            }
                        }
                    ]
                },
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {"text": "```final\nUnable to call it.\n```"}
                                ],
                            }
                        }
                    ]
                },
            ]
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": "read /tmp/a"}]

        diagnostic = await harness._ask_google(messages, "key", allow_native_tools=True)
        assert isinstance(diagnostic, str)
        assert native_tool_call_error(diagnostic) == (
            "functionCall IDs must be unique non-empty strings"
        )
        messages.extend(
            [
                {"role": "assistant", "content": diagnostic},
                {"role": "user", "content": "Return an explicit final response."},
            ]
        )
        final = await harness._ask_google(messages, "key", allow_native_tools=True)
        assert final == "```final\nUnable to call it.\n```"

    asyncio.run(scenario())


def test_openai_payload_native_tools_and_unsupported_cache() -> None:
    async def scenario() -> None:
        harness = _OpenAIHarness(
            [
                RuntimeError("tools is unsupported"),
                {"choices": [{"message": {"content": "text"}}]},
                {"choices": [{"message": {"content": "next"}}]},
                {"choices": [{"message": {"content": "control"}}]},
            ]
        )
        messages = [
            {"role": "user", "content": "inspect /home/alina/test/desktop-fly/"}
        ]
        assert (
            await harness._ask_openai_compatible(
                "openai", messages, "key", allow_native_tools=True
            )
            == "text"
        )
        assert len(harness.payloads) == 2
        assert "tools" in harness.payloads[0] and "tools" not in harness.payloads[1]
        assert "tool_choice" not in harness.payloads[1]
        assert (
            await harness._ask_openai_compatible(
                "openai", messages, "key", allow_native_tools=True
            )
            == "next"
        )
        assert "tools" not in harness.payloads[2]
        await harness._ask_openai_compatible(
            "openai", messages, "key", allow_native_tools=False
        )

    asyncio.run(scenario())


def test_openai_payload_requires_native_tools_only_with_native_catalog() -> None:
    async def scenario() -> None:
        harness = _OpenAIHarness(
            [
                {"choices": [{"message": {"content": "tool turn"}}]},
                {"choices": [{"message": {"content": "plain turn"}}]},
            ]
        )
        messages = [{"role": "user", "content": "inspect /tmp"}]

        await harness._ask_openai_compatible(
            "openai",
            messages,
            "key",
            allow_native_tools=True,
            require_native_tools=True,
        )
        await harness._ask_openai_compatible(
            "openai",
            messages,
            "key",
            allow_native_tools=False,
            require_native_tools=True,
        )

        assert harness.payloads[0]["tool_choice"] == "required"
        assert harness.payloads[0]["tools"]
        assert "tools" not in harness.payloads[1]
        assert "tool_choice" not in harness.payloads[1]

    asyncio.run(scenario())


def test_openai_chat_native_response_preserves_assistant_and_call_id() -> None:
    async def scenario() -> None:
        harness = _OpenAIHarness([])
        catalog = build_native_tool_catalog(_specs(), "read /tmp/a")
        name = catalog.canonical_to_native["file.read_text"]
        harness.responses.append(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": name,
                                        "arguments": '{"value":"/tmp/a"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )
        response = await harness._ask_openai_compatible(
            "openai",
            [{"role": "user", "content": "read /tmp/a"}],
            "key",
            allow_native_tools=True,
        )
        assert isinstance(response, ProviderResponse)
        assert response.tool_turn is not None
        assert response.tool_turn.calls[0].tool_name == "file.read_text"
        assert response.tool_turn.calls[0].call_id == "call_1"
        assert response.tool_turn.native_assistant_turn == {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": name,
                        "arguments": '{"value":"/tmp/a"}',
                    },
                }
            ],
        }
        assert harness.urls == ["https://provider.example/v1/chat/completions"]

    asyncio.run(scenario())


def test_openai_responses_round_trip_uses_call_id_and_responses_endpoint() -> None:
    async def scenario() -> None:
        catalog = build_native_tool_catalog(_specs(), "read /tmp/a")
        name = catalog.canonical_to_native["file.read_text"]
        function_call = {
            "type": "function_call",
            "id": "item_ignored",
            "call_id": "call:opaque/response-1",
            "name": name,
            "arguments": '{"value":"/tmp/a"}',
        }
        harness = _OpenAIHarness(
            [
                {"id": "resp_1", "output": [function_call]},
                {
                    "id": "resp_2",
                    "output": [],
                    "output_text": "```final\nRead it.\n```",
                },
            ]
        )
        harness.config["openai_api_mode"] = "responses"
        messages: list[dict[str, Any]] = [{"role": "user", "content": "read /tmp/a"}]

        first = await harness._ask_provider_once(
            "openai", messages, "key", allow_native_tools=True
        )
        assert isinstance(first, ProviderResponse)
        assert first.tool_turn is not None
        call = first.tool_turn.calls[0]
        assert call.call_id == "call:opaque/response-1"
        assert call.call_id != function_call["id"]
        followup = harness._render_provider_tool_results(
            first.tool_turn, ['{"status":"success"}']
        )
        assert followup is not None
        final = await harness._ask_provider_once(
            "openai", [*messages, *followup], "key", allow_native_tools=True
        )

        assert final == "```final\nRead it.\n```"
        assert harness.urls == [
            "https://provider.example/v1/responses",
            "https://provider.example/v1/responses",
        ]
        assert "messages" not in harness.payloads[0]
        assert harness.payloads[0]["input"] == messages
        assert harness.payloads[0]["tools"][0] == {
            "type": "function",
            "name": harness.payloads[0]["tools"][0]["name"],
            "description": harness.payloads[0]["tools"][0]["description"],
            "parameters": harness.payloads[0]["tools"][0]["parameters"],
        }
        assert harness.payloads[1]["input"][-2:] == [
            function_call,
            {
                "type": "function_call_output",
                "call_id": "call:opaque/response-1",
                "output": '{"status":"success"}',
            },
        ]

    asyncio.run(scenario())


def test_native_file_call_reaches_v2_batch_executor() -> None:
    class Executor:
        def __init__(self) -> None:
            self.calls: tuple[object, ...] = ()

        async def execute_batch(
            self, calls: tuple[object, ...], _requests: tuple[object, ...]
        ):
            self.calls = calls
            return (
                tuple(
                    ToolResult(call.call_id, ToolResultStatus.SUCCESS, {})
                    for call in calls
                ),
                (),
            )

    class Harness(_OpenAgentAgentLoopMixin, _OpenAgentStatusMixin):
        TOOL_CALL_JSON_RE = re.compile(r"```tool_call\s*\n(.*?)```", re.DOTALL)

        def __init__(self) -> None:
            self.executor = Executor()
            self._v2_runtime = SimpleNamespace(
                registry=ToolRegistry(_specs()), executor=self.executor
            )
            self._v2_source_event = None
            self._cancelled_generations: set[str] = set()

        @staticmethod
        def _event_chat_id(_event: object) -> int:
            return 0

        def _tool_names(self) -> set[str]:
            return {spec.canonical_id for spec in self._v2_runtime.registry.specs()}

        @staticmethod
        def _parse_xml_attrs(_attrs: str) -> dict[str, str]:
            return {"value": "/tmp/a"}

    async def scenario() -> None:
        harness = Harness()
        catalog = build_native_tool_catalog(
            harness._v2_runtime.registry.specs(), "/tmp/a"
        )
        converted = native_response_to_fences(
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "read",
                        "function": {
                            "name": catalog.canonical_to_native["file.read_text"],
                            "arguments": '{"value":"/tmp/a"}',
                        },
                    }
                ],
            },
            catalog,
        )
        calls = harness._extract_tool_calls(converted)
        await harness._dispatch_agent_tool_batch(
            calls,
            source_event=None,
            status_event=None,
            agent_log=[],
            started_at=None,
            thinking_notes=[],
            cancel_token=None,
        )
        assert [call.canonical_id for call in harness.executor.calls] == [
            "file.read_text"
        ]

    asyncio.run(scenario())
