from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from conftest import ROOT


sys.path.insert(0, str(ROOT / "Src"))

from OpenAgentLib.PluginHost import (  # noqa: E402
    MAX_IPC_FRAME_BYTES,
    PLUGIN_HOST_PROTOCOL_VERSION,
    PluginHost,
    PluginHostCallError,
    PluginHostConfig,
    PluginHostErrorCode,
    PluginHostProtocolError,
    PluginHostRequest,
    PluginHostSandboxUnavailable,
    PluginHostStatus,
    SandboxMount,
)


def _request(
    number: int,
    operation: str,
    payload: dict | None = None,
    *,
    retryable: bool = False,
) -> PluginHostRequest:
    return PluginHostRequest(
        request_id=f"request-{number}",
        call_id=f"call-{number}",
        operation=operation,
        payload=payload or {},
        retryable=retryable,
    )


def _call(
    host: PluginHost,
    request: PluginHostRequest,
    **kwargs: object,
) -> object:
    return asyncio.run(host.call(request, **kwargs))


def test_round_trip_preserves_call_and_request_trace() -> None:
    outcome = _call(PluginHost(), _request(1, "echo", {"value": {"name": "Ada"}}))

    assert outcome.response.status is PluginHostStatus.SUCCESS
    assert outcome.response.result == {"value": {"name": "Ada"}}
    assert outcome.response.trace.request_id == "request-1"
    assert outcome.response.trace.call_id == "call-1"
    assert outcome.response.trace.state.value == "completed"
    with pytest.raises(TypeError):
        outcome.response.result["value"] = "mutated"


def test_protocol_rejects_unknown_versions_malformed_messages_and_oversized_frames() -> None:
    request = _request(2, "ping")
    envelope = request.to_envelope()
    envelope["version"] = "999"
    with pytest.raises(PluginHostProtocolError, match="version"):
        PluginHostRequest.from_json_line(json.dumps(envelope).encode() + b"\n")
    with pytest.raises(PluginHostProtocolError, match="JSON"):
        PluginHostRequest.from_json_line(b"not-json\n")
    with pytest.raises(PluginHostProtocolError, match="size limit"):
        PluginHostRequest.from_json_line(b"{" + b"x" * MAX_IPC_FRAME_BYTES + b"}\n")


def test_host_rejects_oversized_request_before_worker_launch() -> None:
    request = _request(3, "echo", {"value": "x" * MAX_IPC_FRAME_BYTES})

    with pytest.raises(PluginHostCallError) as error:
        _call(PluginHost(), request)

    assert error.value.code is PluginHostErrorCode.FRAME_TOO_LARGE


def test_duplicate_request_id_is_rejected() -> None:
    host = PluginHost()
    _call(host, _request(4, "ping"))

    with pytest.raises(PluginHostCallError) as error:
        _call(host, _request(4, "ping"))

    assert error.value.code is PluginHostErrorCode.DUPLICATE_REQUEST_ID


@pytest.mark.parametrize(
    ("operation", "expected"),
    (
        ("malformed_response", PluginHostErrorCode.MALFORMED_MESSAGE),
        ("oversized_response", PluginHostErrorCode.FRAME_TOO_LARGE),
        ("wrong_request_id", PluginHostErrorCode.RESPONSE_MISMATCH),
        ("duplicate_response", PluginHostErrorCode.DUPLICATE_RESPONSE),
    ),
)
def test_host_rejects_invalid_worker_responses(
    operation: str, expected: PluginHostErrorCode
) -> None:
    with pytest.raises(PluginHostCallError) as error:
        _call(PluginHost(), _request(10, operation))

    assert error.value.code is expected


def test_environment_is_filtered_and_project_runtime_is_not_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLUGIN_HOST_TEST_SECRET", "must-not-cross-boundary")
    host = PluginHost()

    environment = _call(host, _request(20, "environment"))
    runtime_paths = _call(host, _request(21, "runtime_paths"))

    assert "PLUGIN_HOST_TEST_SECRET" not in environment.response.result["keys"]
    assert all(str(ROOT) not in path for path in runtime_paths.response.result["paths"])


def test_unmounted_file_is_denied(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("parent-only", encoding="utf-8")

    outcome = _call(
        PluginHost(), _request(30, "read_file", {"path": str(secret)})
    )

    assert outcome.response.status is PluginHostStatus.ERROR
    assert outcome.response.error is not None
    assert outcome.response.error.code is PluginHostErrorCode.WORKER_ERROR
    assert "read denied" in outcome.response.error.message


def test_read_only_mount_denies_write(tmp_path: Path) -> None:
    protected = tmp_path / "protected.txt"
    protected.write_text("original", encoding="utf-8")

    outcome = _call(
        PluginHost(),
        _request(31, "write_file", {"path": "/mnt/protected.txt", "content": "changed"}),
        mounts=(SandboxMount(protected, "/mnt/protected.txt", read_only=True),),
    )

    assert outcome.response.status is PluginHostStatus.ERROR
    assert protected.read_text(encoding="utf-8") == "original"


def test_declared_writable_mount_allows_write(tmp_path: Path) -> None:
    output = tmp_path / "output.txt"
    output.write_text("", encoding="utf-8")

    outcome = _call(
        PluginHost(),
        _request(32, "write_file", {"path": "/mnt/output.txt", "content": "written"}),
        mounts=(SandboxMount(output, "/mnt/output.txt", read_only=False),),
    )

    assert outcome.response.status is PluginHostStatus.SUCCESS
    assert output.read_text(encoding="utf-8") == "written"


def test_direct_network_is_unavailable() -> None:
    outcome = _call(PluginHost(), _request(40, "network_probe"))

    assert outcome.response.status is PluginHostStatus.SUCCESS
    assert outcome.response.result == {"connected": False}


def test_crash_is_non_durable_and_preserves_caller_retryability() -> None:
    with pytest.raises(PluginHostCallError) as error:
        _call(PluginHost(), _request(50, "crash", retryable=True))

    assert error.value.code is PluginHostErrorCode.CHILD_CRASHED
    assert error.value.retryable is True


def test_timeout_kills_and_reaps_worker() -> None:
    with pytest.raises(PluginHostCallError) as error:
        _call(
            PluginHost(),
            _request(60, "sleep", {"seconds": 2}, retryable=True),
            wall_timeout=0.1,
        )

    assert error.value.code is PluginHostErrorCode.TIMED_OUT
    assert error.value.retryable is True


def test_cancellation_kills_and_reaps_worker() -> None:
    async def cancel_call() -> PluginHostCallError:
        task = asyncio.create_task(
            PluginHost().call(_request(70, "sleep", {"seconds": 2}, retryable=True))
        )
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(PluginHostCallError) as error:
            await task
        return error.value

    error = asyncio.run(cancel_call())

    assert error.code is PluginHostErrorCode.CANCELLED
    assert error.retryable is True


def test_invalid_mounts_fail_before_launch(tmp_path: Path) -> None:
    mounted = tmp_path / "mounted.txt"
    mounted.write_text("data", encoding="utf-8")

    with pytest.raises(PluginHostCallError) as error:
        _call(
            PluginHost(),
            _request(80, "ping"),
            mounts=(SandboxMount(mounted, "/etc/passwd", read_only=True),),
        )

    assert error.value.code is PluginHostErrorCode.INVALID_MOUNT


@pytest.mark.parametrize("bwrap_path", ("/missing/bwrap", "/usr/bin/false"))
def test_missing_or_unusable_sandbox_fails_closed_without_unsandboxed_worker(
    bwrap_path: str,
) -> None:
    host = PluginHost(PluginHostConfig(bwrap_path=bwrap_path))

    with pytest.raises(PluginHostSandboxUnavailable) as error:
        _call(host, _request(90, "crash"))

    assert error.value.code is PluginHostErrorCode.SANDBOX_UNAVAILABLE


def test_launcher_command_requires_all_isolation_primitives(tmp_path: Path) -> None:
    host = PluginHost()
    command = host.build_command(
        worker_path=tmp_path / "worker.py",
        working_directory=tmp_path / "work",
        mounts=(),
        limits=host.config.limits,
    )

    assert command[0] == "/usr/bin/bwrap"
    assert "--unshare-all" in command
    assert "--unshare-user" in command
    assert "--unshare-net" in command
    assert "--clearenv" in command
    assert "--die-with-parent" in command
    assert "--ro-bind" in command
    assert "--bind" in command
    assert "-I" in command
    assert "-S" in command
    assert PLUGIN_HOST_PROTOCOL_VERSION in command
