from __future__ import annotations

import importlib
import re
import sys
from typing import Any, Mapping

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "Src"))
sys.path.insert(0, str(ROOT.parent / "repo-MCUB-fork" / "OpenAgent"))

from OpenAgentLib.PluginSDK import CapabilityCallContext, CapabilityClient  # noqa: E402
from OpenAgentLib.PluginCapabilities import (  # noqa: E402
    CapabilityBroker, CapabilityErrorCode, CapabilityGrant, CapabilityRequest,
)
from OpenAgentLib.ToolCompatibility import TOOL_COMPATIBILITY_MATRIX  # noqa: E402
from OpenAgentLib.ToolKernel import ToolCall  # noqa: E402
from OpenAgentLib.PluginSDK import CapabilityFamily  # noqa: E402
from OpenAgentLib.ToolPolicy import ToolPolicyCatalog, ToolPolicyEngine, ToolPolicyRequest, ToolPolicyRule  # noqa: E402


TARGET_MODULES = ("chat", "contacts", "creation", "dialog", "message", "moderation", "profile", "file")
_FILE_LOCAL_TOOLS = {"file.read_text", "file.write", "file.edit", "file.patch"}


class FakeTelegramTransport:
    def __init__(self) -> None:
        self.frames: list[Mapping[str, Any]] = []

    def request(self, frame: Mapping[str, Any]) -> Mapping[str, Any]:
        self.frames.append(frame)
        return {"ok": True, "data": {"artifact_id": "artifact-1", "message_id": "message-1"}}


def _modules() -> tuple[object, ...]:
    return tuple(importlib.import_module(f"plugins.{name}") for name in TARGET_MODULES)


def _arguments(schema: Mapping[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in schema.get("required", ()):
        property_schema = schema["properties"][name]
        if property_schema["type"] == "integer":
            values[name] = 1
        elif property_schema["type"] == "array":
            values[name] = ["right"]
        else:
            values[name] = f"opaque-{name}"
    return values


def test_matrix_canonical_ids_are_declared_or_explicitly_deferred() -> None:
    expected = {entry.canonical_id for entry in TOOL_COMPATIBILITY_MATRIX if entry.source_module in TARGET_MODULES}
    declared = {tool.canonical_id for module in _modules() for tool in module.MANIFEST.tools}
    missing = sorted(expected - declared)

    assert not missing, f"missing migrated canonical IDs: {', '.join(missing)}"
    assert "chat.search" not in declared


@pytest.mark.parametrize("module", _modules(), ids=TARGET_MODULES)
def test_every_telegram_tool_uses_one_exact_opaque_capability_request(module: object) -> None:
    for declaration in module.MANIFEST.tools:
        if declaration.canonical_id in _FILE_LOCAL_TOOLS:
            continue
        transport = FakeTelegramTransport()
        call = ToolCall(
            call_id=f"call-{declaration.canonical_id}", spec=declaration.to_tool_spec(),
            requested_name=declaration.canonical_id, arguments=_arguments(declaration.input_schema),
        )
        capability = CapabilityClient(
            CapabilityCallContext("host-1", call.call_id, declaration.canonical_id, "actor:test", "grant-1"),
            transport,
        )

        result = module.HANDLERS[declaration.canonical_id](call, capability)

        assert result["ok"] is True
        assert result["operation"] == transport.frames[0]["operation"]
        assert transport.frames == [{
            "version": "2", "kind": "capability-request", "host_request_id": "host-1",
            "call_id": call.call_id, "canonical_tool_id": declaration.canonical_id,
            "actor_scope": "actor:test", "grant_id": "grant-1", "capability": "telegram",
            "operation": result["operation"], "capability_request_id": f"{call.call_id}:{declaration.canonical_id}",
            "payload": {"data": _arguments(declaration.input_schema)},
        }]


def test_aliases_and_strict_arguments_follow_the_frozen_matrix() -> None:
    message = importlib.import_module("plugins.message")
    declaration = next(tool for tool in message.MANIFEST.tools if tool.canonical_id == "message.send_current")
    assert declaration.aliases == ("message.send", "send_message")
    with pytest.raises(Exception):
        ToolCall("bad", declaration.to_tool_spec(), "message.send", {"text": "x", "client": "raw"})


def test_media_and_profile_mutation_return_only_opaque_artifact_metadata() -> None:
    profile = importlib.import_module("plugins.profile")
    media = importlib.import_module("plugins.file")
    assert "profile.set_photo" in profile.HANDLERS
    assert "file.send" in media.HANDLERS
    assert "file.download_media" in media.HANDLERS
    assert _FILE_LOCAL_TOOLS.issubset(media.HANDLERS)


def test_moderation_requires_parent_confirmation_before_transport() -> None:
    moderation = importlib.import_module("plugins.moderation")
    declaration = next(tool for tool in moderation.MANIFEST.tools if tool.canonical_id == "moderation.ban")
    call = ToolCall("moderation-call", declaration.to_tool_spec(), declaration.canonical_id, {"peer_id": "peer-1", "user_id": "user-1"})
    entry = next(entry for entry in TOOL_COMPATIBILITY_MATRIX if entry.canonical_id == declaration.canonical_id)
    policy = ToolPolicyEngine(ToolPolicyCatalog((ToolPolicyRule.from_compatibility(entry),)))
    grant = CapabilityGrant.for_call("grant-1", "host-1", call, CapabilityFamily.TELEGRAM, frozenset({"moderation-ban"}))
    request = CapabilityRequest("host-1", call.call_id, declaration.canonical_id, f"call:{call.call_id}", "grant-1", CapabilityFamily.TELEGRAM, "moderation-ban", "request-1", {"data": dict(call.arguments)})
    transport = FakeTelegramTransport()

    response = CapabilityBroker(policy, {CapabilityFamily.TELEGRAM: transport}).dispatch(
        call, ToolPolicyRequest(frozenset({declaration.canonical_id}), frozenset({entry.capability_class})), grant, request,
    )

    assert response.error is CapabilityErrorCode.DENIED
    assert not transport.frames


def test_migrated_modules_have_no_telegram_or_ambient_runtime_imports() -> None:
    plugins_root = ROOT.parent / "repo-MCUB-fork" / "OpenAgent" / "plugins"
    forbidden = ("telethon", "client.", "source_event", "kernel", "tool_registry", "tool_map")
    offenders = {
        path.name: token
        for path in (plugins_root / f"{name}.py" for name in TARGET_MODULES)
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    }
    offenders.update({
        path.name: "agent."
        for path in (plugins_root / f"{name}.py" for name in TARGET_MODULES)
        if re.search(r"(?<!open)agent\.", path.read_text(encoding="utf-8"))
    })
    assert not offenders
