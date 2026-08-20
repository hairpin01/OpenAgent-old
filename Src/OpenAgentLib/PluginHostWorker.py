# SPDX-License-Identifier: MIT
"""Deterministic, dependency-free endpoint used to validate PluginHost isolation.

This is not a plugin runtime.  It exposes a deliberately small set of test
operations and never imports project or third-party plugin code.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from math import isfinite
import os
from pathlib import Path
import resource
import socket
import sys
import time
from typing import Any, Mapping

PLUGIN_HOST_PROTOCOL_VERSION = "1"
_MESSAGE_ID_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-"
)


class WorkerProtocolError(ValueError):
    """The parent sent a malformed or unsupported transport frame."""


class WorkerOperationError(ValueError):
    """A controlled test operation could not be completed."""


class _CapabilityTransport:
    """The plugin's only parent interaction: correlated JSON capability frames."""

    def __init__(self, max_frame_bytes: int) -> None:
        self._max_frame_bytes = max_frame_bytes

    def request(self, frame: Mapping[str, Any]) -> Mapping[str, Any]:
        _write_response(frame, self._max_frame_bytes)
        try:
            response = json.loads(
                _read_frame(self._max_frame_bytes)[:-1].decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkerProtocolError("capability response is not JSON") from exc
        if (
            not isinstance(response, dict)
            or response.get("kind") != "capability-response"
        ):
            raise WorkerProtocolError("capability response is malformed")
        if response.get("capability_request_id") != frame.get("capability_request_id"):
            raise WorkerProtocolError("capability response identity is invalid")
        return response


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--protocol-version", required=True)
    parser.add_argument("--max-frame-bytes", type=int, required=True)
    parser.add_argument("--cpu-seconds", type=int, required=True)
    parser.add_argument("--memory-bytes", type=int, required=True)
    parser.add_argument("--file-size-bytes", type=int, required=True)
    parser.add_argument("--process-count", type=int, required=True)
    parser.add_argument("--open-files", type=int, required=True)
    return parser.parse_args()


def _validate_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise WorkerProtocolError(f"{field_name} is malformed")
    if value[0] not in _MESSAGE_ID_CHARS or not value[0].isalnum():
        raise WorkerProtocolError(f"{field_name} is malformed")
    if any(character not in _MESSAGE_ID_CHARS for character in value):
        raise WorkerProtocolError(f"{field_name} is malformed")
    return value


def _read_frame(max_frame_bytes: int) -> bytes:
    frame = sys.stdin.buffer.readline(max_frame_bytes + 1)
    if not frame or len(frame) > max_frame_bytes or not frame.endswith(b"\n"):
        raise WorkerProtocolError("request frame is malformed or too large")
    return frame


def _decode_request(frame: bytes, protocol_version: str) -> dict[str, Any]:
    try:
        request = json.loads(frame[:-1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerProtocolError("request frame is not JSON") from exc
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
    if not isinstance(request, dict) or set(request) != expected:
        raise WorkerProtocolError("request envelope fields are invalid")
    if request["version"] != protocol_version:
        raise WorkerProtocolError("request protocol version is unsupported")
    if request["kind"] != "request":
        raise WorkerProtocolError("request envelope kind is invalid")
    request_id = _validate_id(request["request_id"], "request_id")
    call_id = _validate_id(request["call_id"], "call_id")
    if not isinstance(request["operation"], str):
        raise WorkerProtocolError("request operation is invalid")
    if not isinstance(request["payload"], dict) or not isinstance(
        request["retryable"], bool
    ):
        raise WorkerProtocolError("request payload is invalid")
    trace = request["trace"]
    if not isinstance(trace, dict) or trace != {
        "request_id": request_id,
        "call_id": call_id,
        "state": "requested",
    }:
        raise WorkerProtocolError("request trace does not match request identity")
    return request


def _response(
    request: Mapping[str, Any], *, result: Any = None, error: str | None = None
) -> dict[str, Any]:
    failed = error is not None
    return {
        "version": PLUGIN_HOST_PROTOCOL_VERSION,
        "kind": "response",
        "request_id": request["request_id"],
        "call_id": request["call_id"],
        "status": "error" if failed else "success",
        "result": None if failed else result,
        "error": {"code": "worker_error", "message": error} if failed else None,
        "trace": {
            "request_id": request["request_id"],
            "call_id": request["call_id"],
            "state": "failed" if failed else "completed",
        },
    }


def _apply_limit(kind: int, requested: int) -> None:
    if not isinstance(requested, int) or requested <= 0:
        raise WorkerOperationError("resource limits must be positive")
    _, hard = resource.getrlimit(kind)
    effective = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
    if effective <= 0:
        raise WorkerOperationError("resource limit cannot be enforced")
    resource.setrlimit(kind, (effective, effective))


def apply_resource_limits(args: argparse.Namespace) -> None:
    """Set every positive limit before reading or executing a request."""

    _apply_limit(resource.RLIMIT_CPU, args.cpu_seconds)
    _apply_limit(resource.RLIMIT_AS, args.memory_bytes)
    _apply_limit(resource.RLIMIT_FSIZE, args.file_size_bytes)
    _apply_limit(resource.RLIMIT_NPROC, args.process_count)
    _apply_limit(resource.RLIMIT_NOFILE, args.open_files)


def _path_argument(payload: Mapping[str, Any]) -> Path:
    value = payload.get("path")
    if not isinstance(value, str) or not value.startswith("/"):
        raise WorkerOperationError("path must be absolute")
    return Path(value)


def _plugin_call(request: Mapping[str, Any], max_frame_bytes: int) -> Mapping[str, Any]:
    """Import one verified mounted v2 entrypoint inside the sandbox only."""

    payload = request["payload"]
    expected = {
        "module",
        "plugin_id",
        "plugin_version",
        "canonical_tool_id",
        "entrypoint",
        "source_sha256",
        "arguments",
        "context",
        "grant_id",
    }
    if set(payload) != expected or not isinstance(payload["arguments"], dict):
        raise WorkerOperationError("plugin invocation payload is malformed")
    module_name = payload["module"]
    if not isinstance(module_name, str) or not module_name.startswith("plugins."):
        raise WorkerOperationError("plugin module is not approved")
    module_leaf = module_name.removeprefix("plugins.")
    if not module_leaf.isidentifier() or "." in module_leaf:
        raise WorkerOperationError("plugin module is not approved")
    source = Path("/mnt/pluginroot/plugins") / f"{module_leaf}.py"
    expected_hash = payload["source_sha256"]
    if (
        not isinstance(expected_hash, str)
        or hashlib.sha256(source.read_bytes()).hexdigest() != expected_hash
    ):
        raise WorkerOperationError("plugin source hash does not match parent admission")
    if "/mnt/openagent" not in sys.path:
        sys.path.insert(0, "/mnt/openagent")
    if "/mnt/pluginroot" not in sys.path:
        sys.path.insert(0, "/mnt/pluginroot")
    module = importlib.import_module(module_name)
    manifest = getattr(module, "PLUGIN_MANIFEST", getattr(module, "MANIFEST", None))
    if (
        manifest is None
        or manifest.plugin_id != payload["plugin_id"]
        or manifest.version != payload["plugin_version"]
        or manifest.entrypoint != payload["entrypoint"]
    ):
        raise WorkerOperationError(
            "plugin manifest identity does not match parent admission"
        )
    specs = tuple(
        tool.to_tool_spec(source_module=module_name) for tool in manifest.tools
    )
    spec = next(
        (item for item in specs if item.canonical_id == payload["canonical_tool_id"]),
        None,
    )
    if spec is None:
        raise WorkerOperationError("plugin manifest does not declare requested tool")
    handlers = getattr(module, "TOOL_HANDLERS", getattr(module, "HANDLERS", None))
    handler = handlers.get(spec.canonical_id) if isinstance(handlers, Mapping) else None
    if not callable(handler):
        raise WorkerOperationError("plugin handler is not declared")
    from OpenAgentLib.PluginSDK import CapabilityCallContext, CapabilityClient
    from OpenAgentLib.ToolKernel import ToolCall, ToolContext

    context_data = payload["context"]
    if not isinstance(context_data, dict):
        raise WorkerOperationError("plugin context is malformed")
    context = ToolContext(
        correlation_id=context_data.get("correlation_id", request["call_id"]),
        actor_id=context_data.get("actor_id"),
        metadata=context_data.get("metadata", {}),
    )
    call = ToolCall(
        request["call_id"], spec, spec.canonical_id, payload["arguments"], context
    )
    capability = CapabilityClient(
        CapabilityCallContext(
            request["request_id"],
            request["call_id"],
            spec.canonical_id,
            context.actor_id or f"session:{context.correlation_id}",
            payload["grant_id"],
        ),
        _CapabilityTransport(max_frame_bytes),
    )
    result = handler(call, capability)
    if not isinstance(result, Mapping):
        raise WorkerOperationError("plugin handler result must be a JSON object")
    return dict(result)


def _run_operation(request: Mapping[str, Any], max_frame_bytes: int) -> Any:
    operation = request["operation"]
    payload = request["payload"]
    if operation == "plugin_call":
        return _plugin_call(request, max_frame_bytes)
    if operation == "ping":
        return {"pong": True}
    if operation == "echo":
        return {"value": payload.get("value")}
    if operation == "sleep":
        seconds = payload.get("seconds")
        if (
            not isinstance(seconds, (int, float))
            or isinstance(seconds, bool)
            or not isfinite(seconds)
        ):
            raise WorkerOperationError("sleep seconds must be finite")
        if seconds < 0 or seconds > 60:
            raise WorkerOperationError("sleep seconds are outside the test range")
        time.sleep(seconds)
        return {"slept": seconds}
    if operation == "crash":
        os._exit(91)
    if operation == "environment":
        return {"keys": sorted(os.environ)}
    if operation == "runtime_paths":
        return {"paths": list(sys.path)}
    if operation == "read_file":
        path = _path_argument(payload)
        try:
            return {"content": path.read_text(encoding="utf-8")}
        except OSError as exc:
            raise WorkerOperationError(f"read denied: {exc.strerror or exc}") from exc
    if operation == "write_file":
        path = _path_argument(payload)
        content = payload.get("content")
        if not isinstance(content, str) or len(content.encode("utf-8")) > 4096:
            raise WorkerOperationError("write content is invalid or too large")
        try:
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise WorkerOperationError(f"write denied: {exc.strerror or exc}") from exc
        return {"written": len(content)}
    if operation == "network_probe":
        try:
            with socket.create_connection(("198.51.100.1", 9), timeout=0.2):
                return {"connected": True}
        except OSError:
            return {"connected": False}
    if operation == "malformed_response":
        sys.stdout.buffer.write(b"not-json\n")
        sys.stdout.buffer.flush()
        return None
    if operation == "oversized_response":
        sys.stdout.buffer.write(b"x" * (max_frame_bytes + 1) + b"\n")
        sys.stdout.buffer.flush()
        return None
    if operation == "wrong_request_id":
        response = _response(request, result={"unexpected": True})
        response["request_id"] = "wrong-request-id"
        response["trace"]["request_id"] = "wrong-request-id"
        _write_response(response, max_frame_bytes)
        return None
    if operation == "duplicate_response":
        response = _response(request, result={"duplicate": True})
        _write_response(response, max_frame_bytes)
        _write_response(response, max_frame_bytes)
        return None
    raise WorkerOperationError("operation is not available in the deterministic worker")


def _write_response(response: Mapping[str, Any], max_frame_bytes: int) -> None:
    encoded = (
        json.dumps(
            response, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        + b"\n"
    )
    if len(encoded) > max_frame_bytes:
        raise WorkerOperationError("response frame is too large")
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def _capability_probe(
    request: Mapping[str, Any], max_frame_bytes: int
) -> Mapping[str, Any]:
    """Exercise the bounded capability exchange without loading plugin code."""

    frame = request["payload"].get("frame")
    expected = {
        "version",
        "kind",
        "host_request_id",
        "call_id",
        "canonical_tool_id",
        "actor_scope",
        "grant_id",
        "capability",
        "operation",
        "capability_request_id",
        "payload",
    }
    if (
        not isinstance(frame, dict)
        or set(frame) != expected
        or frame.get("version") != "2"
        or frame.get("kind") != "capability-request"
    ):
        raise WorkerOperationError("capability probe frame is invalid")
    if (frame["host_request_id"], frame["call_id"]) != (
        request["request_id"],
        request["call_id"],
    ):
        raise WorkerOperationError("capability probe identity is invalid")
    _write_response(frame, max_frame_bytes)
    response = json.loads(_read_frame(max_frame_bytes)[:-1].decode("utf-8"))
    response_fields = {
        "version",
        "kind",
        "host_request_id",
        "call_id",
        "capability_request_id",
        "ok",
        "data",
        "error",
    }
    if (
        not isinstance(response, dict)
        or set(response) != response_fields
        or response.get("version") != "2"
        or response.get("kind") != "capability-response"
    ):
        raise WorkerProtocolError("capability response is malformed")
    if (
        response["host_request_id"],
        response["call_id"],
        response["capability_request_id"],
    ) != (request["request_id"], request["call_id"], frame["capability_request_id"]):
        raise WorkerProtocolError("capability response identity is invalid")
    return {"capability_ok": response["ok"], "data": response["data"]}


def main() -> int:
    args = _parse_args()
    if (
        args.protocol_version != PLUGIN_HOST_PROTOCOL_VERSION
        or args.max_frame_bytes < 256
    ):
        return 64
    try:
        apply_resource_limits(args)
        request = _decode_request(
            _read_frame(args.max_frame_bytes), args.protocol_version
        )
        result = (
            _capability_probe(request, args.max_frame_bytes)
            if request["operation"] == "capability_probe"
            else _run_operation(request, args.max_frame_bytes)
        )
        if request["operation"] not in {
            "malformed_response",
            "oversized_response",
            "wrong_request_id",
            "duplicate_response",
        }:
            _write_response(_response(request, result=result), args.max_frame_bytes)
    except (WorkerProtocolError, WorkerOperationError, OSError, ValueError) as exc:
        if "request" in locals():
            _write_response(_response(request, error=str(exc)), args.max_frame_bytes)
            return 0
        return 64
    except Exception as exc:
        if "request" in locals():
            # Never surface worker paths, tracebacks, or arbitrary plugin data.
            detail = type(exc).__name__
            if isinstance(exc, ModuleNotFoundError) and exc.name:
                detail = f"ModuleNotFoundError({exc.name})"
            _write_response(
                _response(request, error=f"plugin worker failed: {detail}"),
                args.max_frame_bytes,
            )
            return 0
        return 64
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
