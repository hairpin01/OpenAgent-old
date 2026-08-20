from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping
from typing import Any

import pytest

from conftest import ROOT
from tool_testkit import (
    FIXED_NOW,
    build_confirmation_grant,
    build_policy_request,
    build_policy_rule,
)


sys.path.insert(0, str(ROOT / "Src"))

from OpenAgentLib.SystemPlugins.native import build_native_system_tools  # noqa: E402
from OpenAgentLib.ToolCompatibility import TOOL_COMPATIBILITY_MATRIX  # noqa: E402
from OpenAgentLib.ToolExecutor import ToolExecutor  # noqa: E402
from OpenAgentLib.ToolKernel import (  # noqa: E402
    ConfirmationRequirement,
    ToolArgumentError,
    ToolContext,
    ToolErrorCode,
    ToolResultStatus,
)
from OpenAgentLib.ToolPolicy import (  # noqa: E402
    ConfirmationState,
    ToolPolicyCatalog,
    ToolPolicyEngine,
)


class RecordingServices:
    """Deterministic fake whose operation name and JSON input are observable."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, Any]]] = []
        self.failure: Exception | None = None

    def __getattr__(self, operation: str) -> Any:
        async def execute(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
            self.calls.append((operation, arguments))
            if self.failure is not None:
                raise self.failure
            return {"result": operation}

        return execute


SYSTEM_ENTRIES = tuple(entry for entry in TOOL_COMPATIBILITY_MATRIX if entry.source_family == "system")


def _value(schema: Mapping[str, Any]) -> Any:
    schema_type = schema["type"]
    if "enum" in schema:
        return schema["enum"][0]
    if schema_type == "string":
        return "sample"
    if schema_type == "integer":
        return 1
    if schema_type == "boolean":
        return True
    if schema_type == "array":
        return [_value(schema["items"])]
    if schema_type == "object":
        return {key: _value(value) for key, value in schema.get("properties", {}).items() if key in schema.get("required", ())}
    raise AssertionError(f"unsupported sample schema {schema_type!r}")


def _arguments(spec: Any) -> dict[str, Any]:
    return _value(spec.input_schema)


def _invalid_arguments(spec: Any) -> dict[str, Any]:
    required = tuple(spec.input_schema.get("required", ()))
    if required:
        return {}
    properties = spec.input_schema.get("properties", {})
    if properties:
        return {next(iter(properties)): object()}
    return {"unknown": "value"}


def _executor(native: Any) -> ToolExecutor:
    policy = ToolPolicyEngine(ToolPolicyCatalog(tuple(build_policy_rule(spec) for spec in native.registry.specs())))
    return ToolExecutor(native.registry, policy, native_handlers=native.handlers)


@pytest.mark.parametrize("entry", SYSTEM_ENTRIES, ids=lambda entry: entry.canonical_id)
def test_every_system_matrix_entry_executes_natively(entry: Any) -> None:
    services = RecordingServices()
    native = build_native_system_tools(services)
    spec = native.registry.resolve(entry.canonical_id)
    call = native.registry.create_call(
        call_id=f"call-{entry.canonical_id}",
        requested_name=entry.canonical_id,
        arguments=_arguments(spec),
        context=ToolContext("correlation", "actor"),
    )
    request = build_policy_request(call)
    if spec.confirmation is ConfirmationRequirement.REQUIRED:
        request = build_policy_request(
            call,
            confirmation=ConfirmationState.APPROVED,
            confirmation_grant=build_confirmation_grant(call),
            now=FIXED_NOW,
        )

    result, _trace = asyncio.run(_executor(native).execute(call, request))

    assert result.status is ToolResultStatus.SUCCESS
    assert result.output == {"result": entry.canonical_id.replace(".", "_")}
    assert services.calls == [(entry.canonical_id.replace(".", "_"), call.arguments)]


@pytest.mark.parametrize("entry", SYSTEM_ENTRIES, ids=lambda entry: entry.canonical_id)
def test_invalid_arguments_are_rejected_before_service_execution(entry: Any) -> None:
    services = RecordingServices()
    native = build_native_system_tools(services)
    spec = native.registry.resolve(entry.canonical_id)

    with pytest.raises(ToolArgumentError):
        native.registry.create_call(
            call_id=f"invalid-{entry.canonical_id}",
            requested_name=entry.canonical_id,
            arguments=_invalid_arguments(spec),
        )

    assert services.calls == []


def test_aliases_resolve_uniquely_and_registry_has_exact_matrix_coverage() -> None:
    native = build_native_system_tools(RecordingServices())
    expected = {entry.canonical_id for entry in SYSTEM_ENTRIES}

    assert len(native.registry.specs()) == 39
    assert {spec.canonical_id for spec in native.registry.specs()} == expected
    assert len(native.handlers) == len(expected)
    for entry in SYSTEM_ENTRIES:
        for alias in entry.aliases:
            assert native.registry.resolve(alias).canonical_id == entry.canonical_id


def test_mutating_tool_requires_confirmation_and_service_failure_is_normalized() -> None:
    services = RecordingServices()
    native = build_native_system_tools(services)
    spec = native.registry.resolve("todo.add")
    call = native.registry.create_call(
        call_id="confirmation",
        requested_name=spec.canonical_id,
        arguments={"text": "task"},
        context=ToolContext("correlation", "actor"),
    )
    denied, _trace = asyncio.run(_executor(native).execute(call, build_policy_request(call)))
    assert denied.error.code is ToolErrorCode.CONFIRMATION_REQUIRED
    assert services.calls == []

    services.failure = RuntimeError("service unavailable")
    request = build_policy_request(
        call,
        confirmation=ConfirmationState.APPROVED,
        confirmation_grant=build_confirmation_grant(call),
        now=FIXED_NOW,
    )
    failed, _trace = asyncio.run(_executor(native).execute(call, request))
    assert failed.status is ToolResultStatus.ERROR
    assert failed.error.code is ToolErrorCode.HANDLER_FAILED


def test_native_handlers_have_no_legacy_dispatch_dependency() -> None:
    source = (ROOT / "Src" / "OpenAgentLib" / "SystemPlugins" / "native.py").read_text()
    assert "ToolDispatch" not in source
    assert "lazy_handler" not in source
    assert "getattr(" not in source
