from __future__ import annotations

import sys
from types import SimpleNamespace
from types import MappingProxyType

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "Src"))

from OpenAgentLib.PluginCapabilities import (  # noqa: E402
    CapabilityBroker,
    CapabilityErrorCode,
    CapabilityGrant,
    CapabilityRequest,
)
from OpenAgentLib.PluginHost import PluginHost, PluginHostRequest  # noqa: E402
from OpenAgentLib.PluginSDK import (  # noqa: E402
    CapabilityFamily,
    LegacyPluginMigrationError,
    PLUGIN_SDK_API_VERSION,
    PluginManifest,
    PluginManifestError,
    PluginToolDeclaration,
    manifest_from_legacy_declarations,
)
from OpenAgentLib.ToolKernel import (  # noqa: E402
    ToolCall,
    ToolContext,
    ToolSpec,
)
from OpenAgentLib.ToolPolicy import (
    ToolPolicyCatalog,
    ToolPolicyEngine,
    ToolPolicyRequest,
    ToolPolicyRule,
)  # noqa: E402
from OpenAgentLib.ToolCompatibility import TOOL_COMPATIBILITY_MATRIX  # noqa: E402


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def invoke(
        self, operation: str, payload: object, grant: object
    ) -> dict[str, object]:
        self.calls.append((operation, payload))
        return {"operation": operation}


def _call() -> tuple[ToolCall, ToolPolicyEngine, ToolPolicyRequest]:
    capabilities = frozenset({"plugin-capability"})
    rule = ToolPolicyRule(
        "demo.tool", capabilities, "none", "serial", "idempotent", "migrate"
    )
    spec = ToolSpec(
        "demo.tool",
        (),
        {"type": "object"},
        {"type": "object"},
        "2",
        "2",
        capabilities,
        "none",
        "serial",
        "idempotent",
        "migrate",
    )
    call = ToolCall(
        "call-1", spec, "demo.tool", {}, ToolContext("session-1", "actor-1")
    )
    request = ToolPolicyRequest(
        frozenset({"demo.tool"}),
        capabilities,
        requested_timeout=1,
        maximum_timeout=1,
        remaining_calls=1,
        remaining_token_budget=1,
        estimated_tokens=0,
    )
    return call, ToolPolicyEngine(ToolPolicyCatalog([rule])), request


def _request(
    capability: CapabilityFamily,
    operation: str,
    payload: dict[str, object],
    grant: CapabilityGrant,
) -> CapabilityRequest:
    return CapabilityRequest(
        "host-1",
        "call-1",
        "demo.tool",
        "actor:actor-1",
        grant.grant_id,
        capability,
        operation,
        "cap-1",
        payload,
    )


def test_manifest_is_versioned_immutable_and_declares_only_known_capabilities() -> None:
    tool = PluginToolDeclaration("demo.tool", capabilities=frozenset({"telegram"}))
    manifest = PluginManifest(
        "demo.plugin",
        "1.0.0",
        "2",
        "demo_plugin.main",
        (tool,),
        frozenset({"telegram"}),
        metadata={"nested": {"value": 1}},
    )
    assert manifest.plugin_id == "demo.plugin"
    with pytest.raises(TypeError):
        manifest.metadata["x"] = 1  # type: ignore[index]
    with pytest.raises(PluginManifestError, match="unknown capability"):
        PluginManifest(
            "demo.plugin", "1", "2", "demo_plugin.main", (tool,), frozenset({"ambient"})
        )
    with pytest.raises(PluginManifestError, match="unsupported"):
        PluginManifest(
            "demo.plugin",
            "1",
            "1",
            "demo_plugin.main",
            (tool,),
            frozenset({"telegram"}),
        )


def test_legacy_declarations_convert_without_importing_and_reject_unmatched_aliases() -> (
    None
):
    matrix = (
        SimpleNamespace(
            source_module="demo", canonical_id="demo.tool", aliases=("demo.alias",)
        ),
    )
    manifest = manifest_from_legacy_declarations(
        {
            "tool_registry": ("demo.tool",),
            "tool_map": {"demo.tool": "run", "demo.alias": "run"},
            "tool_schemas": {"demo.tool": {"type": "object"}},
        },
        plugin_id="demo.plugin",
        entrypoint="demo_plugin.main",
        source_module="demo",
        compatibility_matrix=matrix,
    )
    assert manifest.tools[0].aliases == ("demo.alias",)
    with pytest.raises(LegacyPluginMigrationError, match="alias"):
        manifest_from_legacy_declarations(
            {"tool_registry": ("demo.tool",), "tool_map": {"demo.alias": "run"}},
            plugin_id="demo.plugin",
            entrypoint="demo_plugin.main",
            source_module="demo",
            compatibility_matrix=matrix,
        )


def test_committed_aliases_convert_and_rejected_chat_search_fails() -> None:
    entries = tuple(
        entry
        for entry in TOOL_COMPATIBILITY_MATRIX
        if entry.source_module == "ast_grep"
    )
    registry = tuple(entry.canonical_id for entry in entries)
    tool_map = {
        name: f"handler-{entry.canonical_id}"
        for entry in entries
        for name in (entry.canonical_id, *entry.aliases)
    }
    manifest = manifest_from_legacy_declarations(
        {"tool_registry": registry, "tool_map": tool_map},
        plugin_id="legacy.ast_grep",
        entrypoint="legacy_ast_grep.main",
        source_module="ast_grep",
    )
    assert {alias for tool in manifest.tools for alias in tool.aliases} == {
        alias for entry in entries for alias in entry.aliases
    }
    chat = SimpleNamespace(
        source_module="chat", canonical_id="chat.info", aliases=("chat.search",)
    )
    with pytest.raises(LegacyPluginMigrationError, match="explicitly rejected"):
        manifest_from_legacy_declarations(
            {
                "tool_registry": ("chat.info",),
                "tool_map": {"chat.info": "info", "chat.search": "info"},
            },
            plugin_id="legacy.chat",
            entrypoint="legacy_chat.main",
            source_module="chat",
            compatibility_matrix=(chat,),
        )


@pytest.mark.parametrize(
    ("capability", "operation", "payload", "constraints"),
    [
        (
            CapabilityFamily.TELEGRAM,
            "send-message",
            {"data": {"peer_id": "opaque", "text": "hello"}},
            {},
        ),
        (
            CapabilityFamily.WORKSPACE_FS,
            "read",
            {"path": "notes/a.txt"},
            {"root": "/tmp"},
        ),
        (
            CapabilityFamily.PROCESS,
            "run",
            {
                "argv": ["echo", "ok"],
                "cwd": "bin",
                "timeout_seconds": 1,
                "max_output_bytes": 10,
            },
            {
                "executables": ["echo"],
                "cwd_root": "/tmp",
                "max_timeout_seconds": 2,
                "max_output_bytes": 20,
                "max_args": 4,
                "max_arg_length": 10,
                "env_allowlist": [],
            },
        ),
        (
            CapabilityFamily.HTTPS_FETCH,
            "fetch",
            {"url": "https://example.com/a", "timeout_seconds": 1, "max_bytes": 10},
            {"max_timeout_seconds": 2, "max_bytes": 20},
        ),
        (
            CapabilityFamily.SCHEDULING,
            "schedule",
            {
                "child_call": {
                    "call_id": "child-1",
                    "canonical_tool_id": "demo.tool",
                    "arguments": {},
                    "remaining_calls": 0,
                    "remaining_token_budget": 0,
                    "remaining_depth": 0,
                    "parent_call_id": "call-1",
                    "cancellation_parent_id": "cancel-1",
                }
            },
            {
                "remaining_calls": 1,
                "remaining_token_budget": 1,
                "remaining_depth": 1,
                "cancellation_parent_id": "cancel-1",
            },
        ),
        (
            CapabilityFamily.CONFIGURATION,
            "get",
            {"key": "demo.value"},
            {"namespace": "demo"},
        ),
    ],
)
def test_declared_capability_families_are_narrow_and_return_json(
    capability: CapabilityFamily,
    operation: str,
    payload: dict[str, object],
    constraints: dict[str, object],
) -> None:
    call, policy, policy_request = _call()
    grant = CapabilityGrant.for_call(
        "grant-1", "host-1", call, capability, frozenset({operation}), constraints
    )
    backend = FakeBackend()
    response = CapabilityBroker(
        policy,
        {capability: backend},
        resolver=lambda *_args, **_kwargs: [
            (None, None, None, None, ("93.184.216.34", 443))
        ],
    ).dispatch(
        call, policy_request, grant, _request(capability, operation, payload, grant)
    )
    assert response.ok and response.data == MappingProxyType({"operation": operation})
    assert len(backend.calls) == 1


def test_denial_replay_malformed_and_ambient_objects_fail_closed() -> None:
    call, policy, policy_request = _call()
    grant = CapabilityGrant.for_call(
        "grant-1",
        "host-1",
        call,
        CapabilityFamily.TELEGRAM,
        frozenset({"send-message"}),
    )
    broker = CapabilityBroker(policy, {CapabilityFamily.TELEGRAM: FakeBackend()})
    request = _request(
        CapabilityFamily.TELEGRAM,
        "send-message",
        {"data": {"peer_id": "x", "text": "hi"}},
        grant,
    )
    assert broker.dispatch(call, policy_request, grant, request).ok
    assert (
        broker.dispatch(call, policy_request, grant, request).error
        is CapabilityErrorCode.REPLAYED
    )
    ambient = CapabilityRequest(
        "host-1",
        "call-1",
        "demo.tool",
        "actor:actor-1",
        grant.grant_id,
        CapabilityFamily.TELEGRAM,
        "send-message",
        "cap-ambient",
        {"data": {"client": "forbidden"}},
    )
    assert (
        broker.dispatch(call, policy_request, grant, ambient).error
        is CapabilityErrorCode.INVALID_REQUEST
    )
    forged = CapabilityRequest(
        "host-1",
        "call-2",
        "demo.tool",
        "actor:actor-1",
        grant.grant_id,
        CapabilityFamily.TELEGRAM,
        "send-message",
        "cap-2",
        {"data": {}},
    )
    assert (
        broker.dispatch(call, policy_request, grant, forged).error
        is CapabilityErrorCode.INVALID_GRANT
    )
    with pytest.raises(Exception):
        CapabilityRequest.from_envelope(
            {"kind": "capability-request", "version": PLUGIN_SDK_API_VERSION}
        )


def test_host_capability_frame_is_correlated_and_not_terminal() -> None:
    call, policy, policy_request = _call()
    grant = CapabilityGrant.for_call(
        "grant-1",
        "host-1",
        call,
        CapabilityFamily.CONFIGURATION,
        frozenset({"get"}),
        {"namespace": "demo"},
    )
    capability_frame = _request(
        CapabilityFamily.CONFIGURATION, "get", {"key": "demo.value"}, grant
    ).to_envelope()
    broker = CapabilityBroker(policy, {CapabilityFamily.CONFIGURATION: FakeBackend()})

    async def handler(request: CapabilityRequest):
        return broker.dispatch(call, policy_request, grant, request)

    outcome = __import__("asyncio").run(
        PluginHost().call(
            PluginHostRequest(
                "host-1", "call-1", "capability_probe", {"frame": capability_frame}
            ),
            capability_handler=handler,
        )
    )
    assert outcome.response.result == {
        "capability_ok": True,
        "data": {"operation": "get"},
    }


def test_broker_normalizes_paths_and_rejects_boundary_escapes(tmp_path) -> None:
    call, policy, policy_request = _call()
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)
    fs_grant = CapabilityGrant.for_call(
        "fs",
        "host-1",
        call,
        CapabilityFamily.WORKSPACE_FS,
        frozenset({"read"}),
        {"root": str(tmp_path)},
    )
    fs_request = CapabilityRequest(
        "host-1",
        "call-1",
        "demo.tool",
        "actor:actor-1",
        "fs",
        CapabilityFamily.WORKSPACE_FS,
        "read",
        "fs-1",
        {"path": "escape/file"},
    )
    assert (
        CapabilityBroker(policy, {CapabilityFamily.WORKSPACE_FS: FakeBackend()})
        .dispatch(call, policy_request, fs_grant, fs_request)
        .error
        is CapabilityErrorCode.INVALID_REQUEST
    )

    process_grant = CapabilityGrant.for_call(
        "proc",
        "host-1",
        call,
        CapabilityFamily.PROCESS,
        frozenset({"run"}),
        {
            "executables": ["echo"],
            "cwd_root": str(tmp_path),
            "max_timeout_seconds": 1,
            "max_output_bytes": 10,
            "max_args": 2,
            "max_arg_length": 5,
            "env_allowlist": [],
        },
    )
    process_request = CapabilityRequest(
        "host-1",
        "call-1",
        "demo.tool",
        "actor:actor-1",
        "proc",
        CapabilityFamily.PROCESS,
        "run",
        "proc-1",
        {"argv": ["sh"], "cwd": "run", "timeout_seconds": 1, "max_output_bytes": 1},
    )
    assert (
        CapabilityBroker(policy, {CapabilityFamily.PROCESS: FakeBackend()})
        .dispatch(call, policy_request, process_grant, process_request)
        .error
        is CapabilityErrorCode.INVALID_REQUEST
    )

    network_grant = CapabilityGrant.for_call(
        "net",
        "host-1",
        call,
        CapabilityFamily.HTTPS_FETCH,
        frozenset({"fetch"}),
        {"max_timeout_seconds": 1, "max_bytes": 1},
    )
    network_request = CapabilityRequest(
        "host-1",
        "call-1",
        "demo.tool",
        "actor:actor-1",
        "net",
        CapabilityFamily.HTTPS_FETCH,
        "fetch",
        "net-1",
        {"url": "https://public.invalid", "timeout_seconds": 1, "max_bytes": 1},
    )

    def private_dns(*_args, **_kwargs):
        return [(None, None, None, None, ("127.0.0.1", 443))]

    assert (
        CapabilityBroker(
            policy, {CapabilityFamily.HTTPS_FETCH: FakeBackend()}, resolver=private_dns
        )
        .dispatch(call, policy_request, network_grant, network_request)
        .error
        is CapabilityErrorCode.INVALID_REQUEST
    )


def test_nested_ambient_and_child_or_config_scope_escape_are_denied() -> None:
    call, policy, policy_request = _call()
    telegram_grant = CapabilityGrant.for_call(
        "tg", "host-1", call, CapabilityFamily.TELEGRAM, frozenset({"send-message"})
    )
    nested = CapabilityRequest(
        "host-1",
        "call-1",
        "demo.tool",
        "actor:actor-1",
        "tg",
        CapabilityFamily.TELEGRAM,
        "send-message",
        "tg-1",
        {"data": {"peer_id": "x", "text": "ok", "nested": {"session": "bad"}}},
    )
    assert (
        CapabilityBroker(policy, {CapabilityFamily.TELEGRAM: FakeBackend()})
        .dispatch(call, policy_request, telegram_grant, nested)
        .error
        is CapabilityErrorCode.INVALID_REQUEST
    )

    schedule_grant = CapabilityGrant.for_call(
        "schedule",
        "host-1",
        call,
        CapabilityFamily.SCHEDULING,
        frozenset({"schedule"}),
        {
            "remaining_calls": 1,
            "remaining_token_budget": 1,
            "remaining_depth": 1,
            "cancellation_parent_id": "cancel",
        },
    )
    child = {
        "call_id": "child",
        "canonical_tool_id": "demo.tool",
        "arguments": {},
        "remaining_calls": 1,
        "remaining_token_budget": 0,
        "remaining_depth": 0,
        "parent_call_id": "call-1",
        "cancellation_parent_id": "wrong",
    }
    schedule_request = CapabilityRequest(
        "host-1",
        "call-1",
        "demo.tool",
        "actor:actor-1",
        "schedule",
        CapabilityFamily.SCHEDULING,
        "schedule",
        "schedule-1",
        {"child_call": child},
    )
    assert (
        CapabilityBroker(policy, {CapabilityFamily.SCHEDULING: FakeBackend()})
        .dispatch(call, policy_request, schedule_grant, schedule_request)
        .error
        is CapabilityErrorCode.INVALID_REQUEST
    )

    config_grant = CapabilityGrant.for_call(
        "config",
        "host-1",
        call,
        CapabilityFamily.CONFIGURATION,
        frozenset({"set"}),
        {"namespace": "safe"},
    )
    config_request = CapabilityRequest(
        "host-1",
        "call-1",
        "demo.tool",
        "actor:actor-1",
        "config",
        CapabilityFamily.CONFIGURATION,
        "set",
        "config-1",
        {"key": "unsafe.value", "value": {}},
    )
    assert (
        CapabilityBroker(policy, {CapabilityFamily.CONFIGURATION: FakeBackend()})
        .dispatch(call, policy_request, config_grant, config_request)
        .error
        is CapabilityErrorCode.INVALID_REQUEST
    )


def test_broker_revalidates_backend_redirects() -> None:
    class RedirectBackend:
        def invoke(self, operation, payload, grant):
            return {"redirect_urls": ["https://redirect.invalid"]}

    call, policy, policy_request = _call()
    grant = CapabilityGrant.for_call(
        "net",
        "host-1",
        call,
        CapabilityFamily.HTTPS_FETCH,
        frozenset({"fetch"}),
        {"max_timeout_seconds": 1, "max_bytes": 1},
    )
    request = CapabilityRequest(
        "host-1",
        "call-1",
        "demo.tool",
        "actor:actor-1",
        "net",
        CapabilityFamily.HTTPS_FETCH,
        "fetch",
        "net-1",
        {"url": "https://initial.invalid", "timeout_seconds": 1, "max_bytes": 1},
    )

    def resolver(host, *_args, **_kwargs):
        address = "127.0.0.1" if host == "redirect.invalid" else "93.184.216.34"
        return [(None, None, None, None, (address, 443))]

    assert (
        CapabilityBroker(
            policy, {CapabilityFamily.HTTPS_FETCH: RedirectBackend()}, resolver=resolver
        )
        .dispatch(call, policy_request, grant, request)
        .error
        is CapabilityErrorCode.BACKEND_ERROR
    )
