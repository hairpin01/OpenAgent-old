from __future__ import annotations

import importlib.util
import shutil
import socket
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable

import pytest

from tool_testkit import (
    CancellationHarness,
    FakeTelegramBackend,
    FixedClock,
    JsonRecordingBackend,
    StrictTransportProcess,
    TraceCollector,
    build_confirmation_grant,
    build_policy_request,
    build_policy_rule,
    build_tool_call,
    build_tool_registry,
    build_tool_spec,
    install_fake_process_signals,
)

from OpenAgentLib.PluginCapabilities import CapabilityBroker
from OpenAgentLib.PluginHost import PluginHost, PluginHostConfig
from OpenAgentLib.PluginSDK import CapabilityFamily
from OpenAgentLib.ToolCompatibility import TOOL_COMPATIBILITY_MATRIX
from OpenAgentLib.ToolKernel import ToolCall, ToolSpec
from OpenAgentLib.ToolPolicy import (
    DEFAULT_TOOL_POLICY_CATALOG,
    ToolConcurrencyGate,
    ToolPolicyCatalog,
    ToolPolicyEngine,
)


ROOT = Path(__file__).resolve().parents[1]


def load_source_module(name: str, relative_path: str) -> ModuleType:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixed_clock() -> FixedClock:
    return FixedClock()


@pytest.fixture
def tool_spec_builder() -> Callable[..., ToolSpec]:
    return build_tool_spec


@pytest.fixture
def tool_call_builder() -> Callable[..., ToolCall]:
    return build_tool_call


@pytest.fixture
def tool_registry_builder() -> Callable[[Iterable[ToolSpec] | None], Any]:
    return build_tool_registry


@pytest.fixture
def policy_rule_builder() -> Callable[..., Any]:
    return build_policy_rule


@pytest.fixture
def policy_request_builder() -> Callable[..., Any]:
    return build_policy_request


@pytest.fixture
def confirmation_grant_builder() -> Callable[..., Any]:
    return build_confirmation_grant


@pytest.fixture(scope="session")
def authoritative_policy_catalog() -> ToolPolicyCatalog:
    return DEFAULT_TOOL_POLICY_CATALOG


@pytest.fixture
def authoritative_policy_engine(
    authoritative_policy_catalog: ToolPolicyCatalog,
) -> ToolPolicyEngine:
    return ToolPolicyEngine(authoritative_policy_catalog)


@pytest.fixture
def trace_collector(fixed_clock: FixedClock) -> TraceCollector:
    return TraceCollector(fixed_clock)


@pytest.fixture
def fake_telegram_backend() -> FakeTelegramBackend:
    return FakeTelegramBackend()


@pytest.fixture
def fake_capability_backend() -> JsonRecordingBackend:
    return JsonRecordingBackend()


@pytest.fixture
def capability_broker_builder() -> Callable[..., CapabilityBroker]:
    def public_resolver(*_args: Any, **_kwargs: Any) -> list[tuple[Any, ...]]:
        return [(None, None, None, None, ("93.184.216.34", 443))]

    def build(
        policy: ToolPolicyEngine,
        backends: dict[CapabilityFamily, Any],
    ) -> CapabilityBroker:
        return CapabilityBroker(policy, backends, resolver=public_resolver)

    return build


@pytest.fixture(params=TOOL_COMPATIBILITY_MATRIX, ids=lambda entry: entry.canonical_id)
def compatibility_case(request: pytest.FixtureRequest) -> Any:
    return request.param


@pytest.fixture
def offline_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("unexpected network access in offline test")

    original_socket = socket.socket

    class OfflineSocket(original_socket):
        def connect(self, *_args: Any, **_kwargs: Any) -> Any:
            return blocked()

        def connect_ex(self, *_args: Any, **_kwargs: Any) -> Any:
            return blocked()

    monkeypatch.setattr(socket, "socket", OfflineSocket)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


@pytest.fixture
def isolated_host_config() -> PluginHostConfig:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        pytest.fail("sandbox-required test needs a real bwrap executable")
    return PluginHostConfig(bwrap_path=Path(bwrap))


@pytest.fixture
def sandboxed_host(isolated_host_config: PluginHostConfig) -> PluginHost:
    return PluginHost(isolated_host_config)


@pytest.fixture
def strict_transport_host(monkeypatch: pytest.MonkeyPatch) -> CancellationHarness:
    process = StrictTransportProcess()
    install_fake_process_signals(monkeypatch, process)
    return CancellationHarness(PluginHost(), process)


@pytest.fixture
def serial_cancellation_gate(
    tool_spec_builder: Callable[..., ToolSpec],
) -> tuple[ToolConcurrencyGate, ToolCall]:
    spec = tool_spec_builder(
        "sample.mutate",
        aliases=(),
        capabilities=frozenset({"write"}),
        confirmation="required",
        concurrency="serial",
        idempotency="non-idempotent",
    )
    call = build_tool_call(spec)
    policy = ToolPolicyEngine(ToolPolicyCatalog((build_policy_rule(spec),)))
    return ToolConcurrencyGate(policy), call
