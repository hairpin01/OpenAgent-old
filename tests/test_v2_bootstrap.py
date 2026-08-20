from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path

from OpenAgentLib.ToolCompatibility import TOOL_COMPATIBILITY_MATRIX
from OpenAgentLib.ToolKernel import ToolContext, ToolResultStatus
from OpenAgentLib.ToolPolicy import (
    ConfirmationState,
    ToolConfirmationGrant,
    ToolPolicyRequest,
)
from OpenAgentLib.TodoService import OpenAgentTodoService
from OpenAgentLib.V2Bootstrap import build_v2_tool_runtime


@dataclass
class _Session:
    messages: list[dict[str, str]]


@dataclass(frozen=True)
class _Event:
    chat_id: int


class _RuntimeApp:
    """Concrete runtime state used by the native bootstrap integration probe."""

    def __init__(self) -> None:
        self._v2_source_event = _Event(7)
        self._active_session = {7: "session-7"}
        self._sessions = {"session-7": _Session([{"content": "keep me"}])}
        self._tool_memory = {7: ["memory"]}
        self._runtime_comments = {}
        self._placeholder_context = {}
        self._last_generated_file: dict[str, str] | None = None
        self._todo_items_cache: list[dict[str, str]] = []
        self._todo_service = OpenAgentTodoService()
        self._last_token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        self.config = {"todo_status_emojis": ""}

    def _event_chat_id(self, event: _Event) -> int:
        return event.chat_id

    def _get_active_session(self, chat_id: int) -> _Session:
        return self._sessions[self._active_session[chat_id]]

    def _touch_session(self, session: _Session) -> None:
        del session

    def _todo_items(self) -> list[dict[str, str]]:
        return [dict(item) for item in self._todo_items_cache]

    def _todo_parse_html_text(self, text: str) -> str:
        return self._todo_service.parse_html_text(text)

    def _todo_normalize_status(self, status: str) -> str:
        return self._todo_service.normalize_status(status)

    def _todo_target_index(
        self, items: list[dict[str, str]], attrs: dict[str, str], body: str
    ) -> tuple[int | None, str]:
        return self._todo_service.target_index(items, attrs, body)

    async def _save_todo_items(
        self, items: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        self._todo_items_cache = self._todo_service.clean_items(items)
        return self._todo_items()

    def _format_todo_placeholder(self) -> str:
        return self._todo_service.format_placeholder(self._todo_items_cache)

    def _safe_generated_filename(self, filename: str) -> str:
        name = Path(filename).name or "generated.py"
        return name if "." in name else f"{name}.py"


def test_bootstrap_builds_native_and_sibling_registry_and_executes_model_call() -> None:
    app = _RuntimeApp()
    runtime = build_v2_tool_runtime(app)
    expected = sum(
        entry.migration_disposition == "migrate" for entry in TOOL_COMPATIBILITY_MATRIX
    )
    assert len(runtime.registry.specs()) == expected
    assert runtime.registry.resolve("terminal").canonical_id == "terminal.run"

    async def execute(tool: str, arguments: dict[str, object]):
        output, results, traces = await runtime.execute_model_output(
            f'```tool_call\n{json.dumps({"tool": tool, "args": arguments})}\n```',
            context=ToolContext("test"),
            request_for=lambda call: ToolPolicyRequest(
                enabled_tool_ids=frozenset({call.canonical_id}),
                granted_capabilities=call.spec.capabilities,
                confirmation=ConfirmationState.APPROVED,
                confirmation_grant=ToolConfirmationGrant.for_call("test", call),
                remaining_calls=1,
            ),
        )
        assert output.calls[0].canonical_id == tool
        assert results[0].status is ToolResultStatus.SUCCESS
        assert traces[0].call_id == results[0].call_id
        return results[0].output

    async def scenario() -> None:
        listed = await execute("utility.list_tools", {})
        assert "utility.list_tools" in listed["result"]

        added = await execute("todo.add", {"text": "native task"})
        assert "TODO item added" in added["result"]
        assert app._todo_items_cache == [{"text": "native task", "status": "pending"}]

        cleared = await execute("context.clear", {})
        assert cleared["result"] == "Context cleared"
        assert app._sessions["session-7"].messages == []

        generated = await execute(
            "code.generate_file", {"path": "answer.py", "content": "print(1)"}
        )
        assert "Generated file prepared: answer.py" in generated["result"]
        assert app._last_generated_file == {
            "name": "answer.py",
            "content": "print(1)",
        }

    asyncio.run(scenario())
