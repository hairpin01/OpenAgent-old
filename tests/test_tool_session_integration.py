from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from OpenAgentLib.ContextService import OpenAgentContextService
from OpenAgentLib.Manager.Session import SessionManager
from OpenAgentLib.ResponseAgent import _OpenAgentResponseMixin


class _Logger:
    def warning(self, _message: str) -> None:
        pass


def _manager(path: Path) -> SessionManager:
    return SessionManager(
        path,
        logger=_Logger(),
        model_getter=lambda: "test-model",
        default_name_getter=lambda: "new",
        session_limit=5,
    )


def _terminal(
    *,
    call_id: str = "call-1",
    status: str = "success",
    output: object = None,
    error_code: str | None = None,
    updated_at: datetime | None = None,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    state = {
        "success": "completed",
        "error": "failed",
        "cancelled": "cancelled",
        "timed_out": "timed_out",
    }[status]
    result = SimpleNamespace(
        call_id=call_id,
        status=SimpleNamespace(value=status),
        output=output,
        error=(
            SimpleNamespace(code=SimpleNamespace(value=error_code))
            if error_code
            else None
        ),
    )
    trace = SimpleNamespace(
        call_id=call_id,
        correlation_id="correlation-1",
        state=SimpleNamespace(value=state),
        updated_at=updated_at or datetime.now(timezone.utc),
    )
    return trace, result


def test_terminal_trace_persists_reloads_and_never_replays(tmp_path: Path) -> None:
    async def scenario() -> None:
        path = tmp_path / "sessions.json"
        manager = _manager(path)
        session = manager.new_session(1)
        trace, result = _terminal(output={"ok": True})
        assert manager.record_terminal_trace(session.id, trace, result)
        await manager.close()

        reloaded = _manager(path)
        await reloaded.load()
        restored = reloaded.get_active_session(1)
        assert restored.tool_traces[0]["status"] == "success"
        assert restored.tool_traces[0]["output"] == {"ok": True}
        assert not hasattr(reloaded, "executor")
        assert reloaded._save_task is None

    asyncio.run(scenario())


def test_trace_redaction_spill_context_and_response_summary(tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = _manager(tmp_path / "sessions.json")
        session = manager.new_session(1)
        redacted_trace, redacted_result = _terminal(
            call_id="redacted",
            output={
                "api_key": "never-persist",
                "nested": {"password": "never-persist"},
            },
        )
        assert manager.record_terminal_trace(
            session.id, redacted_trace, redacted_result
        )
        trace, result = _terminal(
            call_id="spill",
            output={
                "api_key": "never-persist",
                "nested": {"password": "never-persist"},
                "spill_ref": "openagent_tool_outputs/1/result.txt",
            },
        )
        assert manager.record_terminal_trace(session.id, trace, result)
        assert session.tool_traces[0]["output"] == {
            "api_key": "[redacted]",
            "nested": {"password": "[redacted]"},
        }
        record = session.tool_traces[1]
        assert record["output"] == {"spill_ref": "openagent_tool_outputs/1/result.txt"}
        assert "never-persist" not in json.dumps(record)

        entries = OpenAgentContextService().context_entries(
            "prompt", "answer", [record]
        )
        assert entries[1]["content"].endswith("openagent_tool_outputs/1/result.txt")
        summary = _OpenAgentResponseMixin()._tool_terminal_status_summary([record])
        assert summary == "Tool status: spill: success"
        await manager.close()

    asyncio.run(scenario())


def test_host_crash_cancellation_and_timeout_are_terminal_not_pending(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manager = _manager(tmp_path / "sessions.json")
        session = manager.new_session(1)
        for call_id, status, code in (
            ("crash", "error", "host_failed"),
            ("cancel", "cancelled", "cancelled"),
            ("timeout", "timed_out", "timed_out"),
        ):
            trace, result = _terminal(call_id=call_id, status=status, error_code=code)
            assert manager.record_terminal_trace(session.id, trace, result)
        assert all("output" not in record for record in session.tool_traces)
        await manager.close()

        reloaded = _manager(tmp_path / "sessions.json")
        await reloaded.load()
        records = reloaded.get_active_session(1).tool_traces
        assert {(record["call_id"], record["status"]) for record in records} == {
            ("crash", "error"),
            ("cancel", "cancelled"),
            ("timeout", "timed_out"),
        }
        assert all(record["status"] != "pending" for record in records)

    asyncio.run(scenario())


def test_malformed_trace_records_fail_closed_and_legacy_sessions_still_load(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        path = tmp_path / "sessions.json"
        path.write_text(
            json.dumps(
                {
                    "sessions": [
                        {
                            "id": "legacy",
                            "name": "new",
                            "chat_id": 1,
                            "created_at": 1,
                            "updated_at": 1,
                            "tool_traces": {
                                "version": 1,
                                "records": [{"status": "pending"}],
                            },
                        },
                        {
                            "id": "no-traces",
                            "name": "new",
                            "chat_id": 2,
                            "created_at": 1,
                            "updated_at": 1,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        manager = _manager(path)
        await manager.load()
        assert manager.sessions["legacy"].tool_traces == []
        assert manager.sessions["no-traces"].tool_traces == []
        await manager.close()

    asyncio.run(scenario())


def test_duplicate_terminal_call_keeps_one_deterministic_latest_record(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manager = _manager(tmp_path / "sessions.json")
        session = manager.new_session(1)
        first_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        first = _terminal(output={"value": "old"}, updated_at=first_time)
        second = _terminal(
            output={"value": "new"}, updated_at=first_time + timedelta(seconds=1)
        )
        assert manager.record_terminal_trace(session.id, *first)
        assert manager.record_terminal_trace(session.id, *second)
        assert session.tool_traces == [
            {**session.tool_traces[0], "output": {"value": "new"}}
        ]
        await manager.close()

    asyncio.run(scenario())


def test_terminal_trace_uses_existing_debounce_and_close_flush(tmp_path: Path) -> None:
    async def scenario() -> None:
        path = tmp_path / "sessions.json"
        manager = _manager(path)
        manager._save_debounce_seconds = 60
        session = manager.new_session(1)
        for index in range(8):
            trace, result = _terminal(call_id=f"call-{index}", output={"index": index})
            assert manager.record_terminal_trace(session.id, trace, result)
        assert manager._save_task is not None
        assert manager._save_generation == 9
        await manager.close()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert len(payload["sessions"][0]["tool_traces"]["records"]) == 8
        assert manager._saved_generation == manager._save_generation

    asyncio.run(scenario())
