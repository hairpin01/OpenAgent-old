from __future__ import annotations

import asyncio

from conftest import load_source_module

runtime = load_source_module(
    "openagent_agent_runtime_test",
    "Src/OpenAgentLib/AgentRuntime.py",
)


def test_explicit_final_fence_is_required() -> None:
    assert runtime.extract_explicit_final("Поняла, сейчас сделаю") is None
    assert (
        runtime.extract_explicit_final("```final\nГотово: команда выполнена.\n```")
        == "Готово: команда выполнена."
    )


def test_explicit_final_tag_is_supported_for_provider_compatibility() -> None:
    assert runtime.extract_explicit_final("<final>Done.</final>") == "Done."
    assert runtime.extract_explicit_final("```final\n\n```") is None
    assert runtime.extract_explicit_final("```finale\nDone.\n```") is None


def test_tool_calls_inside_final_are_inert_and_rejected() -> None:
    nested = (
        "```final\n"
        "Example only:\n"
        "```tool_call\n"
        '{"tool":"terminal.run","args":{"cmd":"rm -rf /"}}\n'
        "```\n"
        "```"
    )
    assert runtime.extract_explicit_final(nested) is None
    assert "terminal.run" not in runtime.strip_explicit_final_regions(nested)


def test_final_tag_regions_are_removed_before_tool_scan() -> None:
    nested = "<final>Do not execute <tool_call>terminal.run</tool_call></final>"
    assert runtime.extract_explicit_final(nested) is None
    assert runtime.strip_explicit_final_regions(nested) == ""


def test_intermediate_note_normalizes_terra_style_promise() -> None:
    raw = ".... поняла.... шмелька.... сейчас запущу terminal.run"
    assert runtime.intermediate_note(raw) == raw


def test_intermediate_note_removes_tool_blocks() -> None:
    raw = (
        "Запускаю проверку.\n"
        "```tool_call\n"
        '{"tool":"terminal.run","args":{"cmd":"pwd"}}\n'
        "```"
    )
    assert runtime.intermediate_note(raw) == "Запускаю проверку."


def test_agent_output_classification_drives_loop() -> None:
    assert runtime.classify_agent_output(
        "Поняла, сейчас вызову terminal.run",
        has_tool_calls=False,
    ) == ("intermediate", "Поняла, сейчас вызову terminal.run")
    assert runtime.classify_agent_output(
        "```final\nГотово.\n```",
        has_tool_calls=False,
    ) == ("final", "Готово.")
    assert runtime.classify_agent_output(
        "```final\nНе выполнять до результата tool.\n```",
        has_tool_calls=True,
    ) == ("tools", "")


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
    assert not runtime.is_transient_provider_error(
        RuntimeError("HTTP 400: bad request")
    )
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
    assert runtime.estimate_text_tokens("привет") > runtime.estimate_text_tokens(
        "hello!"
    )


def test_tool_index_is_complete_and_prompt_independent() -> None:
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
    assert "todo.current" in selected
    assert selected == runtime.relevant_tool_names("совсем другой запрос", names)


def test_tool_index_is_not_silently_truncated() -> None:
    names = {f"plugin.tool_{index:03d}" for index in range(200)}
    assert len(runtime.relevant_tool_names("anything", names)) == 200


def test_search_tool_docs_uses_name_and_documentation() -> None:
    docs = {
        "terminal.run": {
            "desc": "Run a shell command in the workspace",
            "args": "cmd",
        },
        "message.send": {"desc": "Send a Telegram message"},
        "utility.list_tools": {"desc": "List tools"},
    }
    assert runtime.search_tool_docs("terminal", docs)[0] == "terminal.run"
    assert runtime.search_tool_docs("shell command", docs)[0] == "terminal.run"
    assert runtime.search_tool_docs("Telegram message", docs)[0] == "message.send"
    assert runtime.search_tool_docs("missing capability", docs) == ()


def test_completion_gate_requires_exact_verdict() -> None:
    assert runtime.accepts_completion_verdict("ACCEPT")
    assert runtime.accepts_completion_verdict("  accept\n")
    assert not runtime.accepts_completion_verdict("ACCEPT because it looks fine")
    assert not runtime.accepts_completion_verdict("CONTINUE")
    assert not runtime.accepts_completion_verdict("щас гляну муху")


def test_model_tool_reason_is_optional_and_model_authored() -> None:
    assert runtime.model_tool_reason({"reason": "Проверяю каталог перед анализом"}) == (
        "Проверяю каталог перед анализом"
    )
    assert runtime.model_tool_reason({"comment": "Looking up the docs"}) == (
        "Looking up the docs"
    )
    assert (
        runtime.model_tool_reason({"purpose": "Compare results"}) == "Compare results"
    )
    assert runtime.model_tool_reason({}) == ""


def test_json_tool_reason_survives_legacy_conversion() -> None:
    converted = runtime.json_tool_payload_to_legacy(
        {
            "tool": "terminal.run",
            "reason": "Проверяю & объясняю",
            "args": {"cmd": "ls -la"},
        },
        {"terminal.run"},
    )
    assert converted is not None
    tool, attrs, body = converted
    assert tool == "terminal.run"
    assert 'reason="Проверяю &amp; объясняю"' in attrs
    assert 'cmd="ls -la"' in attrs
    assert body == ""
    assert (
        runtime.json_tool_payload_to_legacy(
            {"tool": "unknown.tool", "reason": "nope"},
            {"terminal.run"},
        )
        is None
    )


def test_action_router_replaces_repeated_promise_with_strict_block_request() -> None:
    messages = runtime.build_action_router_messages(
        "изучи $HOME/test/desktop-fly",
        "сейчас просмотрю структуру проекта",
        {"terminal.run", "utility.search_tool", "file.read"},
    )
    assert messages[0]["role"] == "system"
    assert "Output ONLY" in messages[0]["content"]
    assert "```tool_call```" in messages[0]["content"]
    assert "utility.search_tool" in messages[0]["content"]
    assert "terminal.run" in messages[0]["content"]
    assert "optional top-level `reason`" in messages[0]["content"]
    assert "сейчас просмотрю" in messages[1]["content"]
