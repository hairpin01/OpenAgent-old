# SPDX-License-Identifier: MIT
"""Fail-closed Linux launcher and JSON-lines transport for isolated plugins.

This module intentionally launches only the deterministic worker from
``PluginHostWorker.py``.  Later manifest and capability work can build on this
transport without gaining ambient access to the parent process.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from enum import Enum
import json
from math import isfinite
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from types import MappingProxyType
from typing import Any, Awaitable, BinaryIO, Callable, Mapping, Sequence

from .PluginCapabilities import (
    CapabilityProtocolError,
    CapabilityRequest,
    CapabilityResponse,
)

PLUGIN_HOST_PROTOCOL_VERSION = "1"
MAX_IPC_FRAME_BYTES = 64 * 1024
PLUGIN_MOUNT_ROOT = "/mnt"
_MESSAGE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_OPERATION_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class PluginHostErrorCode(str, Enum):
    """Stable failures at the parent/worker process boundary."""

    CANCELLED = "cancelled"
    CHILD_CRASHED = "child_crashed"
    DUPLICATE_REQUEST_ID = "duplicate_request_id"
    DUPLICATE_RESPONSE = "duplicate_response"
    FRAME_TOO_LARGE = "frame_too_large"
    INVALID_MOUNT = "invalid_mount"
    INVALID_REQUEST = "invalid_request"
    MALFORMED_MESSAGE = "malformed_message"
    RESPONSE_MISMATCH = "response_mismatch"
    SANDBOX_UNAVAILABLE = "sandbox_unavailable"
    TIMED_OUT = "timed_out"
    UNSUPPORTED_PROTOCOL_VERSION = "unsupported_protocol_version"
    WORKER_ERROR = "worker_error"


class PluginHostStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class PluginHostTraceState(str, Enum):
    REQUESTED = "requested"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class PluginHostFailure:
    """A serializable, non-durable transport or worker failure."""

    code: PluginHostErrorCode
    message: str
    request_id: str | None = None
    call_id: str | None = None
    retryable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.code, PluginHostErrorCode):
            object.__setattr__(self, "code", PluginHostErrorCode(self.code))
        message = _required_text(self.message, "failure message")
        object.__setattr__(self, "message", message)
        if self.request_id is not None:
            object.__setattr__(
                self, "request_id", _validate_message_id(self.request_id, "request_id")
            )
        if self.call_id is not None:
            object.__setattr__(
                self, "call_id", _validate_message_id(self.call_id, "call_id")
            )
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be a bool")

    def to_envelope(self) -> dict[str, Any]:
        return {"code": self.code.value, "message": self.message}

    @classmethod
    def from_envelope(cls, value: Any) -> PluginHostFailure:
        if not isinstance(value, Mapping) or set(value) != {"code", "message"}:
            raise PluginHostProtocolError(
                "error details must contain only code and message"
            )
        return cls(code=value["code"], message=value["message"])


class PluginHostCallError(RuntimeError):
    """A typed failure that ended the current non-durable host call."""

    def __init__(self, details: PluginHostFailure) -> None:
        self.details = details
        super().__init__(details.message)

    @property
    def code(self) -> PluginHostErrorCode:
        return self.details.code

    @property
    def retryable(self) -> bool:
        return self.details.retryable


class PluginHostSandboxUnavailable(PluginHostCallError):
    """The required Linux isolation boundary could not be started."""


class PluginHostProtocolError(ValueError):
    """An IPC envelope is malformed before it can be trusted."""


@dataclass(frozen=True)
class PluginHostTrace:
    """Explicit request/call correlation transported in every envelope."""

    request_id: str
    call_id: str
    state: PluginHostTraceState

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_id", _validate_message_id(self.request_id, "request_id")
        )
        object.__setattr__(
            self, "call_id", _validate_message_id(self.call_id, "call_id")
        )
        if not isinstance(self.state, PluginHostTraceState):
            object.__setattr__(self, "state", PluginHostTraceState(self.state))

    def to_envelope(self) -> dict[str, str]:
        return {
            "request_id": self.request_id,
            "call_id": self.call_id,
            "state": self.state.value,
        }

    @classmethod
    def from_envelope(cls, value: Any) -> PluginHostTrace:
        if not isinstance(value, Mapping) or set(value) != {
            "request_id",
            "call_id",
            "state",
        }:
            raise PluginHostProtocolError(
                "trace must contain request_id, call_id, and state"
            )
        return cls(
            request_id=value["request_id"],
            call_id=value["call_id"],
            state=value["state"],
        )


@dataclass(frozen=True)
class PluginHostRequest:
    """One immutable, caller-classified operation sent to the isolated worker."""

    request_id: str
    call_id: str
    operation: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    retryable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_id", _validate_message_id(self.request_id, "request_id")
        )
        object.__setattr__(
            self, "call_id", _validate_message_id(self.call_id, "call_id")
        )
        if not isinstance(self.operation, str) or not _OPERATION_RE.fullmatch(
            self.operation
        ):
            raise ValueError("operation must be a lower-case protocol operation")
        frozen = _freeze_json_value(self.payload)
        if not isinstance(frozen, Mapping):
            raise TypeError("request payload must be a JSON object")
        object.__setattr__(self, "payload", frozen)
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be a bool")

    @property
    def trace(self) -> PluginHostTrace:
        return PluginHostTrace(
            request_id=self.request_id,
            call_id=self.call_id,
            state=PluginHostTraceState.REQUESTED,
        )

    def to_envelope(self) -> dict[str, Any]:
        return {
            "version": PLUGIN_HOST_PROTOCOL_VERSION,
            "kind": "request",
            "request_id": self.request_id,
            "call_id": self.call_id,
            "operation": self.operation,
            "payload": _thaw_json_value(self.payload),
            "retryable": self.retryable,
            "trace": self.trace.to_envelope(),
        }

    def to_json_line(self, *, max_frame_bytes: int = MAX_IPC_FRAME_BYTES) -> bytes:
        return _encode_json_line(self.to_envelope(), max_frame_bytes=max_frame_bytes)

    @classmethod
    def from_envelope(cls, value: Any) -> PluginHostRequest:
        envelope = _validate_envelope(value, kind="request")
        expected = {
            "version",
            "kind",
            "request_id",
            "call_id",
            "operation",
            "payload",
            "retryable",
            "trace",
        }
        if set(envelope) != expected:
            raise PluginHostProtocolError("request envelope has unexpected fields")
        request = cls(
            request_id=envelope["request_id"],
            call_id=envelope["call_id"],
            operation=envelope["operation"],
            payload=envelope["payload"],
            retryable=envelope["retryable"],
        )
        if PluginHostTrace.from_envelope(envelope["trace"]) != request.trace:
            raise PluginHostProtocolError(
                "request trace does not match request identity"
            )
        return request

    @classmethod
    def from_json_line(
        cls, value: bytes, *, max_frame_bytes: int = MAX_IPC_FRAME_BYTES
    ) -> PluginHostRequest:
        return cls.from_envelope(
            _decode_json_line(value, max_frame_bytes=max_frame_bytes)
        )


@dataclass(frozen=True)
class PluginHostResponse:
    """Validated worker result/error with exact request/call correlation."""

    request_id: str
    call_id: str
    status: PluginHostStatus
    trace: PluginHostTrace
    result: Any = None
    error: PluginHostFailure | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_id", _validate_message_id(self.request_id, "request_id")
        )
        object.__setattr__(
            self, "call_id", _validate_message_id(self.call_id, "call_id")
        )
        if not isinstance(self.status, PluginHostStatus):
            object.__setattr__(self, "status", PluginHostStatus(self.status))
        if not isinstance(self.trace, PluginHostTrace):
            raise TypeError("response trace must be a PluginHostTrace")
        if (self.trace.request_id, self.trace.call_id) != (
            self.request_id,
            self.call_id,
        ):
            raise ValueError("response trace must match response identity")
        frozen = _freeze_json_value(self.result)
        object.__setattr__(self, "result", frozen)
        if self.error is not None and not isinstance(self.error, PluginHostFailure):
            raise TypeError("response error must be a PluginHostFailure")
        if self.status is PluginHostStatus.SUCCESS and self.error is not None:
            raise ValueError("successful responses cannot have an error")
        if self.status is PluginHostStatus.ERROR and self.error is None:
            raise ValueError("failed responses require an error")
        expected_trace_state = (
            PluginHostTraceState.COMPLETED
            if self.status is PluginHostStatus.SUCCESS
            else PluginHostTraceState.FAILED
        )
        if self.trace.state is not expected_trace_state:
            raise ValueError("response trace state does not match response status")

    def to_envelope(self) -> dict[str, Any]:
        return {
            "version": PLUGIN_HOST_PROTOCOL_VERSION,
            "kind": "response",
            "request_id": self.request_id,
            "call_id": self.call_id,
            "status": self.status.value,
            "result": _thaw_json_value(self.result),
            "error": self.error.to_envelope() if self.error is not None else None,
            "trace": self.trace.to_envelope(),
        }

    def to_json_line(self, *, max_frame_bytes: int = MAX_IPC_FRAME_BYTES) -> bytes:
        return _encode_json_line(self.to_envelope(), max_frame_bytes=max_frame_bytes)

    @classmethod
    def from_envelope(cls, value: Any) -> PluginHostResponse:
        envelope = _validate_envelope(value, kind="response")
        expected = {
            "version",
            "kind",
            "request_id",
            "call_id",
            "status",
            "result",
            "error",
            "trace",
        }
        if set(envelope) != expected:
            raise PluginHostProtocolError("response envelope has unexpected fields")
        error = (
            None
            if envelope["error"] is None
            else PluginHostFailure.from_envelope(envelope["error"])
        )
        return cls(
            request_id=envelope["request_id"],
            call_id=envelope["call_id"],
            status=envelope["status"],
            result=envelope["result"],
            error=error,
            trace=PluginHostTrace.from_envelope(envelope["trace"]),
        )

    @classmethod
    def from_json_line(
        cls, value: bytes, *, max_frame_bytes: int = MAX_IPC_FRAME_BYTES
    ) -> PluginHostResponse:
        return cls.from_envelope(
            _decode_json_line(value, max_frame_bytes=max_frame_bytes)
        )


@dataclass(frozen=True)
class PluginHostOutcome:
    """A completed worker response; transport failures instead raise typed errors."""

    request: PluginHostRequest
    response: PluginHostResponse

    def __post_init__(self) -> None:
        if not isinstance(self.request, PluginHostRequest):
            raise TypeError("outcomes require a PluginHostRequest")
        if not isinstance(self.response, PluginHostResponse):
            raise TypeError("outcomes require a PluginHostResponse")
        if (self.response.request_id, self.response.call_id) != (
            self.request.request_id,
            self.request.call_id,
        ):
            raise ValueError("outcome response must match its request")

    @property
    def retryable(self) -> bool:
        """The trusted policy classification, never a worker-provided decision."""

        return self.request.retryable


@dataclass(frozen=True)
class SandboxMount:
    """One explicit parent path exposed at an isolated path in the worker."""

    source: Path | str
    target: str
    read_only: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", Path(self.source))
        if not isinstance(self.target, str):
            raise TypeError("mount target must be a string")
        if not isinstance(self.read_only, bool):
            raise TypeError("mount read_only must be a bool")


@dataclass(frozen=True)
class WorkerResourceLimits:
    """Positive per-worker resource limits applied before request processing."""

    cpu_seconds: int = 5
    memory_bytes: int = 128 * 1024 * 1024
    file_size_bytes: int = 1024 * 1024
    process_count: int = 8
    open_files: int = 32

    def __post_init__(self) -> None:
        for field_name in (
            "cpu_seconds",
            "memory_bytes",
            "file_size_bytes",
            "process_count",
            "open_files",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


@dataclass(frozen=True)
class PluginHostConfig:
    """Static launcher settings; no parent environment is ever inherited."""

    bwrap_path: Path | str = Path("/usr/bin/bwrap")
    python_path: Path | str = Path(getattr(sys, "_base_executable", sys.executable))
    default_wall_timeout: float = 5.0
    max_frame_bytes: int = MAX_IPC_FRAME_BYTES
    limits: WorkerResourceLimits = field(default_factory=WorkerResourceLimits)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bwrap_path", Path(self.bwrap_path))
        object.__setattr__(self, "python_path", Path(self.python_path))
        if (
            not isinstance(self.default_wall_timeout, (int, float))
            or isinstance(self.default_wall_timeout, bool)
            or not isfinite(self.default_wall_timeout)
            or self.default_wall_timeout <= 0
        ):
            raise ValueError("default_wall_timeout must be a positive finite number")
        if not isinstance(self.max_frame_bytes, int) or self.max_frame_bytes < 256:
            raise ValueError("max_frame_bytes must be an integer of at least 256")
        if not isinstance(self.limits, WorkerResourceLimits):
            raise TypeError("limits must be WorkerResourceLimits")


class PluginHost:
    """Launch a single deterministic worker behind a required Bubblewrap boundary."""

    _SAFE_ENVIRONMENT = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
    }

    def __init__(self, config: PluginHostConfig | None = None) -> None:
        self.config = config or PluginHostConfig()
        self._request_ids: set[str] = set()

    async def call(
        self,
        request: PluginHostRequest,
        *,
        mounts: Sequence[SandboxMount] = (),
        limits: WorkerResourceLimits | None = None,
        wall_timeout: float | None = None,
        capability_handler: (
            Callable[
                [CapabilityRequest], CapabilityResponse | Awaitable[CapabilityResponse]
            ]
            | None
        ) = None,
    ) -> PluginHostOutcome:
        """Run exactly one request or fail without an unsandboxed fallback."""

        if not isinstance(request, PluginHostRequest):
            raise TypeError("request must be a PluginHostRequest")
        self._claim_request_id(request)
        timeout = self._validated_timeout(request, wall_timeout)
        effective_limits = limits or self.config.limits
        if not isinstance(effective_limits, WorkerResourceLimits):
            raise TypeError("limits must be WorkerResourceLimits")
        try:
            self._ensure_sandbox_available(request)
            validated_mounts = _validate_mounts(mounts)
            request_frame = request.to_json_line(
                max_frame_bytes=self.config.max_frame_bytes
            )
        except PluginHostCallError:
            raise
        except PluginHostProtocolError as exc:
            raise self._failure(
                request, _protocol_error_code(exc), f"invalid worker request: {exc}"
            ) from exc
        except (TypeError, ValueError, OSError) as exc:
            raise self._failure(
                request, PluginHostErrorCode.INVALID_MOUNT, str(exc)
            ) from exc

        try:
            with tempfile.TemporaryDirectory(
                prefix="openagent_plugin_host_"
            ) as temporary:
                private_dir = Path(temporary)
                worker_path = private_dir / "PluginHostWorker.py"
                working_directory = private_dir / "work"
                shutil.copyfile(self._worker_source_path(), worker_path)
                working_directory.mkdir(mode=0o700)
                command = self.build_command(
                    worker_path=worker_path,
                    working_directory=working_directory,
                    mounts=validated_mounts,
                    limits=effective_limits,
                )
                try:
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        close_fds=True,
                        start_new_session=True,
                        cwd=private_dir,
                        env=dict(self._SAFE_ENVIRONMENT),
                    )
                except OSError as exc:
                    raise self._sandbox_unavailable(
                        request, f"could not start bwrap: {exc}"
                    ) from exc
                return await self._exchange(
                    process, request, request_frame, timeout, capability_handler
                )
        except PluginHostCallError:
            raise

    def build_command(
        self,
        *,
        worker_path: Path,
        working_directory: Path,
        mounts: Sequence[SandboxMount],
        limits: WorkerResourceLimits,
    ) -> tuple[str, ...]:
        """Return the argv-only, deny-by-default Bubblewrap invocation."""

        command: list[str] = [
            str(self.config.bwrap_path),
            "--die-with-parent",
            "--unshare-all",
            "--unshare-user",
            "--unshare-net",
            "--disable-userns",
            "--clearenv",
        ]
        for name, value in self._SAFE_ENVIRONMENT.items():
            command.extend(("--setenv", name, value))

        # The base interpreter and standard library are the only shared runtime.
        for runtime_path in (Path("/usr"), Path("/lib"), Path("/lib64")):
            if runtime_path.exists():
                command.extend(("--ro-bind", str(runtime_path), str(runtime_path)))
        command.extend(("--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"))
        command.extend(
            ("--dir", "/worker", "--dir", "/work", "--dir", PLUGIN_MOUNT_ROOT)
        )
        command.extend(("--ro-bind", str(worker_path), "/worker/PluginHostWorker.py"))
        command.extend(("--bind", str(working_directory), "/work"))

        created_directories = {PLUGIN_MOUNT_ROOT}
        for mount in mounts:
            for directory in _mount_parent_directories(mount.target):
                if directory not in created_directories:
                    command.extend(("--dir", directory))
                    created_directories.add(directory)
            command.extend(
                (
                    "--ro-bind" if mount.read_only else "--bind",
                    str(mount.source),
                    mount.target,
                )
            )

        command.extend(
            (
                "--chdir",
                "/work",
                "--",
                str(self.config.python_path),
                "-I",
                "-S",
                "-B",
                "/worker/PluginHostWorker.py",
                "--protocol-version",
                PLUGIN_HOST_PROTOCOL_VERSION,
                "--max-frame-bytes",
                str(self.config.max_frame_bytes),
                "--cpu-seconds",
                str(limits.cpu_seconds),
                "--memory-bytes",
                str(limits.memory_bytes),
                "--file-size-bytes",
                str(limits.file_size_bytes),
                "--process-count",
                str(limits.process_count),
                "--open-files",
                str(limits.open_files),
            )
        )
        return tuple(command)

    def _claim_request_id(self, request: PluginHostRequest) -> None:
        if request.request_id in self._request_ids:
            raise self._failure(
                request,
                PluginHostErrorCode.DUPLICATE_REQUEST_ID,
                "request_id has already been used by this host",
            )
        self._request_ids.add(request.request_id)

    def _validated_timeout(
        self, request: PluginHostRequest, supplied_timeout: float | None
    ) -> float:
        timeout = (
            self.config.default_wall_timeout
            if supplied_timeout is None
            else supplied_timeout
        )
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not isfinite(timeout)
            or timeout <= 0
        ):
            raise self._failure(
                request,
                PluginHostErrorCode.INVALID_REQUEST,
                "wall timeout must be positive and finite",
            )
        return float(timeout)

    def _ensure_sandbox_available(self, request: PluginHostRequest) -> None:
        if not sys.platform.startswith("linux"):
            raise self._sandbox_unavailable(
                request, "isolated plugins require Linux Bubblewrap"
            )
        bwrap = self.config.bwrap_path
        if (
            not bwrap.is_absolute()
            or not bwrap.is_file()
            or not os.access(bwrap, os.X_OK)
        ):
            raise self._sandbox_unavailable(
                request, "configured bwrap executable is unavailable"
            )
        try:
            probe = subprocess.run(
                (str(bwrap), "--version"),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                env=dict(self._SAFE_ENVIRONMENT),
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise self._sandbox_unavailable(
                request, "configured bwrap executable is unusable"
            ) from exc
        if probe.returncode != 0:
            raise self._sandbox_unavailable(
                request, "configured bwrap executable is unusable"
            )
        python_path = self.config.python_path
        try:
            resolved_python = python_path.resolve(strict=True)
        except OSError as exc:
            raise self._sandbox_unavailable(
                request, "configured Python runtime is unavailable"
            ) from exc
        if not _is_relative_to(resolved_python, Path("/usr")) or not os.access(
            resolved_python, os.X_OK
        ):
            raise self._sandbox_unavailable(
                request,
                "configured Python runtime is outside the minimal /usr runtime mount",
            )
        if not self._worker_source_path().is_file():
            raise self._sandbox_unavailable(
                request, "isolated worker source is unavailable"
            )

    @staticmethod
    def _worker_source_path() -> Path:
        return Path(__file__).with_name("PluginHostWorker.py")

    async def _exchange(
        self,
        process: subprocess.Popen[bytes],
        request: PluginHostRequest,
        request_frame: bytes,
        timeout: float,
        capability_handler: (
            Callable[
                [CapabilityRequest], CapabilityResponse | Awaitable[CapabilityResponse]
            ]
            | None
        ),
    ) -> PluginHostOutcome:
        deadline = time.monotonic() + timeout
        try:
            await self._await_with_deadline(
                asyncio.to_thread(
                    _write_frame,
                    process.stdin,
                    request_frame,
                ),
                deadline,
            )
            line = await self._await_with_deadline(
                self._read_terminal_frame(
                    process, request, capability_handler, deadline
                ),
                deadline,
            )
            if line is None:
                return_code = await self._await_with_deadline(
                    asyncio.to_thread(process.wait), deadline
                )
                stderr = await asyncio.to_thread(
                    _read_limited, process.stderr, self.config.max_frame_bytes
                )
                if _is_sandbox_startup_error(return_code, stderr):
                    raise self._sandbox_unavailable(
                        request, _sandbox_error_message(stderr)
                    )
                raise self._failure(
                    request,
                    PluginHostErrorCode.CHILD_CRASHED,
                    f"isolated worker exited without a response (status {return_code})",
                )

            try:
                response = PluginHostResponse.from_json_line(
                    line, max_frame_bytes=self.config.max_frame_bytes
                )
            except PluginHostProtocolError as exc:
                raise self._failure(
                    request,
                    _protocol_error_code(exc),
                    f"invalid worker response: {exc}",
                ) from exc
            if (response.request_id, response.call_id) != (
                request.request_id,
                request.call_id,
            ):
                raise self._failure(
                    request,
                    PluginHostErrorCode.RESPONSE_MISMATCH,
                    "worker response identity does not match the active request",
                )

            if process.stdin is not None:
                process.stdin.close()

            return_code = await self._await_with_deadline(
                asyncio.to_thread(process.wait), deadline
            )
            extra_output = await asyncio.to_thread(
                _read_limited, process.stdout, self.config.max_frame_bytes
            )
            if extra_output:
                raise self._failure(
                    request,
                    PluginHostErrorCode.DUPLICATE_RESPONSE,
                    "worker emitted more than one response frame",
                )
            if return_code != 0:
                raise self._failure(
                    request,
                    PluginHostErrorCode.CHILD_CRASHED,
                    f"isolated worker exited after responding (status {return_code})",
                )
            return PluginHostOutcome(request=request, response=response)

        except asyncio.TimeoutError as exc:
            await self._stop_process(process)
            raise self._failure(
                request,
                PluginHostErrorCode.TIMED_OUT,
                "isolated worker exceeded its wall timeout",
            ) from exc
        except asyncio.CancelledError as exc:
            await self._stop_process(process)
            raise self._failure(
                request,
                PluginHostErrorCode.CANCELLED,
                "isolated worker call was cancelled",
            ) from exc
        except PluginHostCallError:
            await self._stop_process(process)
            raise
        except PluginHostProtocolError as exc:
            await self._stop_process(process)
            raise self._failure(
                request, _protocol_error_code(exc), f"invalid worker response: {exc}"
            ) from exc
        except (BrokenPipeError, OSError) as exc:
            await self._stop_process(process)
            stderr = await asyncio.to_thread(
                _read_limited, process.stderr, self.config.max_frame_bytes
            )
            if _is_sandbox_startup_error(process.returncode or 1, stderr):
                raise self._sandbox_unavailable(
                    request, _sandbox_error_message(stderr)
                ) from exc
            raise self._failure(
                request,
                PluginHostErrorCode.CHILD_CRASHED,
                f"isolated worker I/O failed: {exc}",
            ) from exc
        finally:
            _close_stream(process.stdin)
            _close_stream(process.stdout)
            _close_stream(process.stderr)

    async def _read_terminal_frame(
        self,
        process: subprocess.Popen[bytes],
        request: PluginHostRequest,
        capability_handler: (
            Callable[
                [CapabilityRequest], CapabilityResponse | Awaitable[CapabilityResponse]
            ]
            | None
        ),
        deadline: float,
    ) -> bytes | None:
        """Only a terminal response ends an exchange; capability frames are routed back."""
        while True:
            line = await self._await_with_deadline(
                asyncio.to_thread(
                    _read_json_line, process.stdout, self.config.max_frame_bytes
                ),
                deadline,
            )
            if line is None:
                return None
            envelope = _decode_json_line(
                line, max_frame_bytes=self.config.max_frame_bytes
            )
            if envelope.get("kind") == "response":
                return line
            if (
                envelope.get("kind") != "capability-request"
                or capability_handler is None
            ):
                raise PluginHostProtocolError(
                    "worker emitted an unexpected non-terminal frame"
                )
            try:
                capability_request = CapabilityRequest.from_envelope(envelope)
            except CapabilityProtocolError as exc:
                raise PluginHostProtocolError(
                    f"malformed capability request: {exc}"
                ) from exc
            if (capability_request.host_request_id, capability_request.call_id) != (
                request.request_id,
                request.call_id,
            ):
                raise PluginHostProtocolError(
                    "capability request identity does not match active request"
                )
            capability_response = capability_handler(capability_request)
            if hasattr(capability_response, "__await__"):
                capability_response = await self._await_with_deadline(
                    capability_response, deadline
                )
            if not isinstance(capability_response, CapabilityResponse):
                raise PluginHostProtocolError(
                    "capability handler returned an invalid response"
                )
            if (
                capability_response.host_request_id,
                capability_response.call_id,
                capability_response.capability_request_id,
            ) != (
                request.request_id,
                request.call_id,
                capability_request.capability_request_id,
            ):
                raise PluginHostProtocolError(
                    "capability response identity does not match request"
                )
            await self._await_with_deadline(
                asyncio.to_thread(
                    _write_frame,
                    process.stdin,
                    _encode_json_line(
                        capability_response.to_envelope(),
                        max_frame_bytes=self.config.max_frame_bytes,
                    ),
                ),
                deadline,
            )

    async def _await_with_deadline(self, awaitable: Any, deadline: float) -> Any:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError
        return await asyncio.wait_for(awaitable, timeout=remaining)

    async def _stop_process(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            with contextlib.suppress(OSError):
                await asyncio.to_thread(process.wait)
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            with contextlib.suppress(OSError):
                process.terminate()
        try:
            await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=0.5)
            return
        except asyncio.TimeoutError:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            with contextlib.suppress(OSError):
                process.kill()
        with contextlib.suppress(OSError):
            await asyncio.to_thread(process.wait)

    @staticmethod
    def _failure(
        request: PluginHostRequest, code: PluginHostErrorCode, message: str
    ) -> PluginHostCallError:
        return PluginHostCallError(
            PluginHostFailure(
                code=code,
                message=message,
                request_id=request.request_id,
                call_id=request.call_id,
                retryable=request.retryable,
            )
        )

    @staticmethod
    def _sandbox_unavailable(
        request: PluginHostRequest, message: str
    ) -> PluginHostSandboxUnavailable:
        return PluginHostSandboxUnavailable(
            PluginHostFailure(
                code=PluginHostErrorCode.SANDBOX_UNAVAILABLE,
                message=message,
                request_id=request.request_id,
                call_id=request.call_id,
                retryable=request.retryable,
            )
        )


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")
    return value.strip()


def _validate_message_id(value: Any, field_name: str) -> str:
    value = _required_text(value, field_name)
    if not _MESSAGE_ID_RE.fullmatch(value):
        raise ValueError(f"{field_name} is malformed")
    return value


def _freeze_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise TypeError("JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            frozen[key] = _freeze_json_value(child)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(child) for child in value)
    raise TypeError("values must be JSON-compatible")


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(child) for child in value]
    return value


def _encode_json_line(value: Mapping[str, Any], *, max_frame_bytes: int) -> bytes:
    _reject_oversized_json_strings(value, max_frame_bytes)
    try:
        encoder = json.JSONEncoder(
            allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        encoded = bytearray()
        for chunk in encoder.iterencode(value):
            chunk_bytes = chunk.encode("utf-8")
            if len(encoded) + len(chunk_bytes) + 1 > max_frame_bytes:
                raise PluginHostProtocolError(
                    "IPC frame exceeds the configured size limit"
                )
            encoded.extend(chunk_bytes)
        encoded.append(ord("\n"))
    except PluginHostProtocolError:
        raise
    except (TypeError, ValueError) as exc:
        raise PluginHostProtocolError("envelope is not JSON serializable") from exc
    return bytes(encoded)


def _reject_oversized_json_strings(value: Any, max_frame_bytes: int) -> None:
    if isinstance(value, str):
        if len(value) > max_frame_bytes:
            raise PluginHostProtocolError("IPC frame exceeds the configured size limit")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_oversized_json_strings(key, max_frame_bytes)
            _reject_oversized_json_strings(child, max_frame_bytes)
    elif isinstance(value, tuple):
        for child in value:
            _reject_oversized_json_strings(child, max_frame_bytes)


def _decode_json_line(value: bytes, *, max_frame_bytes: int) -> Mapping[str, Any]:
    if not isinstance(value, bytes):
        raise PluginHostProtocolError("IPC frames must be bytes")
    if not value.endswith(b"\n"):
        raise PluginHostProtocolError("IPC frame is not newline terminated")
    if len(value) > max_frame_bytes:
        raise PluginHostProtocolError("IPC frame exceeds the configured size limit")
    try:
        decoded = json.loads(value[:-1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PluginHostProtocolError("IPC frame is not valid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise PluginHostProtocolError("IPC envelopes must be JSON objects")
    return decoded


def _validate_envelope(value: Any, *, kind: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PluginHostProtocolError("IPC envelopes must be JSON objects")
    if value.get("version") != PLUGIN_HOST_PROTOCOL_VERSION:
        raise PluginHostProtocolError("unsupported IPC protocol version")
    if value.get("kind") != kind:
        raise PluginHostProtocolError(f"expected a {kind} envelope")
    return value


def _validate_mounts(mounts: Sequence[SandboxMount]) -> tuple[SandboxMount, ...]:
    validated: list[SandboxMount] = []
    targets: list[PurePosixPath] = []
    sources: list[Path] = []
    protected_write_roots = (Path.home().resolve(), Path.cwd().resolve())
    for mount in mounts:
        if not isinstance(mount, SandboxMount):
            raise TypeError("mounts must contain SandboxMount values")
        if not mount.source.is_absolute():
            raise ValueError("mount sources must be absolute")
        source = mount.source.resolve(strict=True)
        target = _validate_mount_target(mount.target)
        if any(_paths_overlap(target, existing) for existing in targets):
            raise ValueError("mount targets cannot duplicate or overlap")
        if any(_paths_overlap(source, existing) for existing in sources):
            raise ValueError("mount sources cannot duplicate or overlap")
        if not mount.read_only and any(
            _is_relative_to(root, source) for root in protected_write_roots
        ):
            raise ValueError("read-write mounts cannot expose the home or project root")
        validated.append(
            SandboxMount(source=source, target=str(target), read_only=mount.read_only)
        )
        targets.append(target)
        sources.append(source)
    return tuple(validated)


def _validate_mount_target(value: str) -> PurePosixPath:
    if (
        not value.startswith("/")
        or "//" in value
        or "/./" in value
        or ".." in value.split("/")
    ):
        raise ValueError("mount targets must be normalized absolute paths")
    target = PurePosixPath(value)
    root = PurePosixPath(PLUGIN_MOUNT_ROOT)
    if target == root or not _is_relative_to(target, root):
        raise ValueError(f"mount targets must be below {PLUGIN_MOUNT_ROOT}")
    return target


def _mount_parent_directories(target: str) -> tuple[str, ...]:
    path = PurePosixPath(target)
    root = PurePosixPath(PLUGIN_MOUNT_ROOT)
    directories = [parent for parent in path.parents if parent != PurePosixPath("/")]
    return tuple(
        str(directory) for directory in reversed(directories) if directory != root
    )


def _paths_overlap(left: Path | PurePosixPath, right: Path | PurePosixPath) -> bool:
    return _is_relative_to(left, right) or _is_relative_to(right, left)


def _is_relative_to(path: Path | PurePosixPath, parent: Path | PurePosixPath) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _write_frame(stream: BinaryIO | None, frame: bytes) -> None:
    if stream is None:
        raise BrokenPipeError("worker stdin is unavailable")
    stream.write(frame)
    stream.flush()


def _write_and_close_stdin(stream: BinaryIO | None, frame: bytes) -> None:
    try:
        _write_frame(stream, frame)
    finally:
        if stream is not None:
            with contextlib.suppress(OSError):
                stream.close()


def _read_json_line(stream: BinaryIO | None, max_frame_bytes: int) -> bytes | None:
    if stream is None:
        raise BrokenPipeError("worker stdout is unavailable")
    line = stream.readline(max_frame_bytes + 1)
    if not line:
        return None
    if len(line) > max_frame_bytes or not line.endswith(b"\n"):
        raise PluginHostProtocolError("IPC frame exceeds the configured size limit")
    return line


def _read_limited(stream: BinaryIO | None, max_frame_bytes: int) -> bytes:
    if stream is None:
        return b""
    return stream.read(max_frame_bytes + 1)


def _close_stream(stream: BinaryIO | None) -> None:
    if stream is not None:
        with contextlib.suppress(OSError):
            stream.close()


def _protocol_error_code(error: PluginHostProtocolError) -> PluginHostErrorCode:
    if "version" in str(error):
        return PluginHostErrorCode.UNSUPPORTED_PROTOCOL_VERSION
    if "size limit" in str(error):
        return PluginHostErrorCode.FRAME_TOO_LARGE
    return PluginHostErrorCode.MALFORMED_MESSAGE


def _is_sandbox_startup_error(return_code: int, stderr: bytes) -> bool:
    if return_code == 0:
        return False
    lowered = stderr.lower()
    return any(
        marker in lowered
        for marker in (
            b"operation not permitted",
            b"creating new namespace failed",
            b"no permissions to create new namespace",
            b"bwrap:",
        )
    )


def _sandbox_error_message(stderr: bytes) -> str:
    message = stderr.decode("utf-8", "replace").strip()
    return f"Bubblewrap sandbox setup failed: {message[:512] or 'unknown error'}"
