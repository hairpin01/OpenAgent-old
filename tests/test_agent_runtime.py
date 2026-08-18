from __future__ import annotations

import asyncio

from conftest import load_source_module


runtime = load_source_module(
    "openagent_agent_runtime_test",
    "Src/OpenAgentLib/AgentRuntime.py",
)


def test_thinking_is_skipped_when_disabled() -> None:
    assert not runtime.should_request_thinking(
        "Implement a large migration with many steps",
        "off",
        flash_mode=False,
    )
    assert not runtime.should_request_thinking(
        "Implement a large migration with many steps",
        "high",
        flash_mode=True,
    )


def test_thinking_low_uses_complexity_gate() -> None:
    assert not runtime.should_request_thinking("hello", "low", flash_mode=False)
    assert runtime.should_request_thinking(
        "Please refactor this module and verify the migration",
        "low",
        flash_mode=False,
    )


def test_retry_policy_only_accepts_transient_failures() -> None:
    assert runtime.is_transient_provider_error(RuntimeError("HTTP 429: busy"))
    assert runtime.is_transient_provider_error(asyncio.TimeoutError())
    assert not runtime.is_transient_provider_error(RuntimeError("HTTP 400: bad request"))
    assert not runtime.is_transient_provider_error(RuntimeError("invalid API key"))
    assert [runtime.retry_delay(i) for i in range(1, 6)] == [0.5, 1.0, 2.0, 4.0, 8.0]


def test_model_call_budget_counts_attempts_and_reserves_final_call() -> None:
    budget = runtime.ModelCallBudget(3)
    assert (
        budget.tool_rounds(
            thinking_enabled=True,
            configured_steps=10,
            final_attempts=1,
        )
        == 1
    )
    budget.reserve()
    budget.reserve()
    budget.reserve()
    try:
        budget.reserve()
    except RuntimeError as exc:
        assert "budget exhausted" in str(exc)
    else:
        raise AssertionError("budget must reject a fourth provider attempt")


def test_single_call_budget_disables_tool_rounds() -> None:
    budget = runtime.ModelCallBudget(1)
    assert budget.tool_rounds(thinking_enabled=False, configured_steps=6) == 0


def test_model_call_budget_can_reserve_two_final_attempts() -> None:
    budget = runtime.ModelCallBudget(8)
    assert (
        budget.tool_rounds(
            thinking_enabled=False,
            configured_steps=6,
            final_attempts=2,
        )
        == 6
    )
    budget.reserve()
    assert budget.remaining == 7


def test_token_budget_preserves_system_and_newest_messages() -> None:
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "old " * 200},
        {"role": "assistant", "content": "old answer " * 200},
        {"role": "user", "content": "latest request"},
    ]
    trimmed = runtime.trim_messages_to_budget(messages, 30)
    assert trimmed[0] == messages[0]
    assert trimmed[-1] == messages[-1]
    assert messages[1] not in trimmed
    assert runtime.estimate_messages_tokens(trimmed) <= 30


def test_oversized_latest_request_is_truncated_not_dropped() -> None:
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "old"},
        {"role": "user", "content": "important " * 1000},
    ]
    trimmed = runtime.trim_messages_to_budget(messages, 80)
    assert trimmed[-1]["role"] == "user"
    assert trimmed[-1]["content"].startswith("important")
    assert "truncated to context budget" in trimmed[-1]["content"]
    assert runtime.estimate_messages_tokens(trimmed) <= 80


def test_oversized_system_control_prompt_is_preserved() -> None:
    messages = [
        {"role": "system", "content": "CONTROL TOOL RULES " + "docs " * 1000},
        {"role": "user", "content": "latest request"},
    ]
    trimmed = runtime.trim_messages_to_budget(messages, 100)
    assert trimmed[0]["role"] == "system"
    assert trimmed[0]["content"].startswith("CONTROL TOOL RULES")
    assert "truncated to context budget" in trimmed[0]["content"]
    assert trimmed[-1] == messages[-1]
    assert runtime.estimate_messages_tokens(trimmed) <= 100


def test_non_ascii_token_estimate_is_conservative() -> None:
    assert runtime.estimate_text_tokens("привет") > runtime.estimate_text_tokens("hello!")


def test_relevant_tools_keep_discovery_and_requested_group() -> None:
    names = {
        "utility.list_tools",
        "utility.tool_help",
        "utility.plugin_docs",
        "thinking.note",
        "code.read_docs",
        "code.generate_file",
        "todo.current",
    }
    selected = runtime.relevant_tool_names("Fix code in a file", names)
    assert "code.read_docs" in selected
    assert "code.generate_file" in selected
    assert "utility.list_tools" in selected
    assert "todo.current" not in selected
