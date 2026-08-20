from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "Src"))
sys.path.insert(0, str(ROOT.parent / "repo-MCUB-fork" / "OpenAgent"))

from OpenAgentLib.PluginCapabilities import (
    CapabilityBroker,
    CapabilityErrorCode,
    CapabilityGrant,
    CapabilityRequest,
)  # noqa: E402
from OpenAgentLib.PluginSDK import (
    CapabilityCallContext,
    CapabilityClient,
    CapabilityFamily,
)  # noqa: E402
from OpenAgentLib.ToolCompatibility import TOOL_COMPATIBILITY_MATRIX  # noqa: E402
from OpenAgentLib.ToolKernel import ToolCall  # noqa: E402
from OpenAgentLib.ToolPolicy import (
    PolicyDecisionKind,
    ToolPolicyRequest,
    tool_scope_for,
)  # noqa: E402

TARGET_MODULES = ("eval", "mcub", "task")


class RecordingTransport:
    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        self.frames: list[Mapping[str, Any]] = []
        self.data = dict(data or {"accepted": True})

    def request(self, frame: Mapping[str, Any]) -> Mapping[str, Any]:
        self.frames.append(frame)
        return {"ok": True, "data": self.data}


class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def invoke(
        self, operation: str, payload: Mapping[str, Any], _grant: CapabilityGrant
    ) -> Mapping[str, Any]:
        self.calls.append((operation, payload))
        return {"operation": operation}


class AllowPolicy:
    def evaluate(self, _call: ToolCall, _request: ToolPolicyRequest) -> SimpleNamespace:
        return SimpleNamespace(kind=PolicyDecisionKind.ALLOW)


def _modules() -> tuple[object, ...]:
    return tuple(importlib.import_module(f"plugins.{name}") for name in TARGET_MODULES)


def _declaration(module: object, tool_id: str) -> object:
    return next(tool for tool in module.MANIFEST.tools if tool.canonical_id == tool_id)


def _call(
    declaration: object, arguments: Mapping[str, Any], call_id: str = "parent-1"
) -> ToolCall:
    return ToolCall(
        call_id, declaration.to_tool_spec(), declaration.canonical_id, arguments
    )


def _capability(call: ToolCall, transport: RecordingTransport) -> CapabilityClient:
    return CapabilityClient(
        CapabilityCallContext(
            "host-1", call.call_id, call.spec.canonical_id, "actor:test", "grant-1"
        ),
        transport,
    )


def _policy_request(call: ToolCall) -> ToolPolicyRequest:
    return ToolPolicyRequest(
        frozenset({call.spec.canonical_id}),
        call.spec.capabilities,
        remaining_calls=5,
        remaining_token_budget=500,
    )


def _request(
    call: ToolCall,
    capability: CapabilityFamily,
    operation: str,
    payload: Mapping[str, Any],
    request_id: str = "request-1",
) -> CapabilityRequest:
    return CapabilityRequest(
        "host-1",
        call.call_id,
        call.spec.canonical_id,
        tool_scope_for(call),
        "grant-1",
        capability,
        operation,
        request_id,
        payload,
    )


def test_canonical_runtime_ids_and_aliases_match_the_frozen_matrix() -> None:
    expected = {
        entry.canonical_id
        for entry in TOOL_COMPATIBILITY_MATRIX
        if entry.source_module in TARGET_MODULES
        and entry.migration_disposition == "migrate"
    }
    declared = {
        tool.canonical_id for module in _modules() for tool in module.MANIFEST.tools
    }
    assert declared == expected
    for module in _modules():
        for tool in module.MANIFEST.tools:
            entry = next(
                item
                for item in TOOL_COMPATIBILITY_MATRIX
                if item.canonical_id == tool.canonical_id
            )
            assert tool.aliases == entry.aliases
    assert "eval.python.telegram" not in declared
    assert "eval.python.telegram.help" not in declared


def test_eval_is_safe_json_only_and_has_no_ambient_objects() -> None:
    module = importlib.import_module("plugins.eval")
    declaration = _declaration(module, "eval.python")
    result = module.HANDLERS["eval.python"](
        _call(
            declaration,
            {
                "code": "sum(values) + offset",
                "variables": {"values": [1, 2, 3], "offset": 4},
            },
        ),
        None,
    )
    assert result == {"ok": True, "value": {"json": "10"}, "steps": result["steps"]}
    for code in (
        "__import__('os')",
        "value.__class__",
        "__builtins__",
        "while True: pass",
        "[item for item in values]",
    ):
        with pytest.raises(ValueError):
            module.HANDLERS["eval.python"](
                _call(declaration, {"code": code, "variables": {"values": [1]}}), None
            )
    with pytest.raises(ValueError):
        module.HANDLERS["eval.python"](
            _call(declaration, {"code": "'x' * 10000"}), None
        )
    source = (
        ROOT.parent / "repo-MCUB-fork" / "OpenAgent" / "plugins" / "eval.py"
    ).read_text()
    for forbidden in (
        "client.on",
        "process_command",
        "_MCUBEvent",
        "eval(",
        "exec(",
        "compile(",
    ):
        assert forbidden not in source


def test_mcub_uses_named_control_operations_and_exact_config_keys() -> None:
    module = importlib.import_module("plugins.mcub")
    declaration = _declaration(module, "mcub.command")
    transport = RecordingTransport()
    result = module.HANDLERS["mcub.command"](
        _call(declaration, {"operation": "module-list"}),
        _capability(_call(declaration, {"operation": "module-list"}), transport),
    )
    assert result["operation"] == "module-list"
    assert transport.frames[0]["capability"] == "mcub-control"
    assert transport.frames[0]["payload"] == {}
    with pytest.raises(Exception):
        _call(declaration, {"operation": "arbitrary-command"})
    config = _declaration(module, "mcub.config")
    with pytest.raises(Exception):
        _call(config, {"operation": "get", "key": "kernel.prefix"})


def test_mcub_broker_denies_unknown_operations_and_config_namespace_escape() -> None:
    module = importlib.import_module("plugins.mcub")
    declaration = _declaration(module, "mcub.modules")
    call = _call(declaration, {})
    backend = RecordingBackend()
    broker = CapabilityBroker(AllowPolicy(), {CapabilityFamily.MCUB_CONTROL: backend})
    grant = CapabilityGrant.for_call(
        "grant-1",
        "host-1",
        call,
        CapabilityFamily.MCUB_CONTROL,
        frozenset({"module-list", "config-get"}),
        {"keys": ["openagent.system_prompt"]},
    )
    allowed = broker.dispatch(
        call,
        _policy_request(call),
        grant,
        _request(call, CapabilityFamily.MCUB_CONTROL, "module-list", {}),
    )
    assert allowed.ok and backend.calls == [("module-list", {})]
    unknown = broker.dispatch(
        call,
        _policy_request(call),
        grant,
        _request(call, CapabilityFamily.MCUB_CONTROL, "shell", {}, "request-2"),
    )
    assert unknown.error is CapabilityErrorCode.UNKNOWN_OPERATION
    escaped = broker.dispatch(
        call,
        _policy_request(call),
        grant,
        _request(
            call,
            CapabilityFamily.MCUB_CONTROL,
            "config-get",
            {"key": "kernel.prefix"},
            "request-3",
        ),
    )
    assert escaped.error is CapabilityErrorCode.INVALID_REQUEST


def _task_arguments(**overrides: Any) -> dict[str, Any]:
    arguments = {
        "canonical_tool_id": "web.fetch",
        "arguments": {"url": "https://example.test"},
        "remaining_calls": 2,
        "remaining_token_budget": 20,
        "remaining_depth": 1,
        "cancellation_parent_id": "cancel-1",
    }
    arguments.update(overrides)
    return arguments


def test_task_schedules_a_normalized_bounded_child_call() -> None:
    module = importlib.import_module("plugins.task")
    declaration = _declaration(module, "task.background")
    call = _call(declaration, _task_arguments())
    transport = RecordingTransport({"scheduled": True})
    result = module.HANDLERS["task.background"](call, _capability(call, transport))
    child = transport.frames[0]["payload"]["child_call"]
    assert result["child_call_id"] == child["call_id"]
    assert child["call_id"].startswith("child-")
    assert child["parent_call_id"] == call.call_id
    assert child["cancellation_parent_id"] == "cancel-1"
    assert {
        "call_id",
        "canonical_tool_id",
        "arguments",
        "remaining_calls",
        "remaining_token_budget",
        "remaining_depth",
        "parent_call_id",
        "cancellation_parent_id",
    } == set(child)
    with pytest.raises(ValueError):
        module.HANDLERS["task.background"](
            _call(
                declaration, _task_arguments(canonical_tool_id="task.run_background")
            ),
            RecordingTransport(),
        )


@pytest.mark.parametrize(
    ("child", "constraints"),
    [
        (
            {
                "call_id": "parent-1",
                "canonical_tool_id": "web.fetch",
                "arguments": {},
                "remaining_calls": 1,
                "remaining_token_budget": 10,
                "remaining_depth": 1,
                "parent_call_id": "parent-1",
                "cancellation_parent_id": "cancel-1",
            },
            {},
        ),
        (
            {
                "call_id": "child-1",
                "canonical_tool_id": "web.fetch",
                "arguments": {},
                "remaining_calls": 3,
                "remaining_token_budget": 10,
                "remaining_depth": 1,
                "parent_call_id": "parent-1",
                "cancellation_parent_id": "cancel-1",
            },
            {},
        ),
        (
            {
                "call_id": "child-1",
                "canonical_tool_id": "web.fetch",
                "arguments": {},
                "remaining_calls": 1,
                "remaining_token_budget": 10,
                "remaining_depth": 1,
                "parent_call_id": "parent-1",
                "cancellation_parent_id": "wrong",
            },
            {},
        ),
        (
            {
                "call_id": "ancestor-1",
                "canonical_tool_id": "web.fetch",
                "arguments": {},
                "remaining_calls": 1,
                "remaining_token_budget": 10,
                "remaining_depth": 1,
                "parent_call_id": "parent-1",
                "cancellation_parent_id": "cancel-1",
            },
            {"ancestor_call_ids": ["ancestor-1"]},
        ),
        (
            {
                "call_id": "child-1",
                "canonical_tool_id": "task.background",
                "arguments": {},
                "remaining_calls": 1,
                "remaining_token_budget": 10,
                "remaining_depth": 1,
                "parent_call_id": "parent-1",
                "cancellation_parent_id": "cancel-1",
            },
            {"ancestor_tool_ids": ["task.background"]},
        ),
    ],
)
def test_scheduling_broker_rejects_self_cycle_budget_and_cancellation_violations(
    child: Mapping[str, Any], constraints: Mapping[str, Any]
) -> None:
    module = importlib.import_module("plugins.task")
    declaration = _declaration(module, "task.background")
    call = _call(declaration, _task_arguments())
    backend = RecordingBackend()
    broker = CapabilityBroker(AllowPolicy(), {CapabilityFamily.SCHEDULING: backend})
    grant_constraints = {
        "remaining_calls": 3,
        "remaining_token_budget": 30,
        "remaining_depth": 2,
        "cancellation_parent_id": "cancel-1",
        **constraints,
    }
    grant = CapabilityGrant.for_call(
        "grant-1",
        "host-1",
        call,
        CapabilityFamily.SCHEDULING,
        frozenset({"schedule"}),
        grant_constraints,
    )
    response = broker.dispatch(
        call,
        _policy_request(call),
        grant,
        _request(call, CapabilityFamily.SCHEDULING, "schedule", {"child_call": child}),
    )
    assert response.error is CapabilityErrorCode.INVALID_REQUEST
    assert backend.calls == []
