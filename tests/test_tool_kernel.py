from __future__ import annotations

import importlib
import sys
from datetime import datetime, timezone
from types import MappingProxyType

import pytest

from conftest import ROOT, load_source_module


sys.path.insert(0, str(ROOT / "Src"))
sys.path.insert(0, str(ROOT / "Src/OpenAgentLib"))
kernel = load_source_module(
    "openagent_tool_kernel_test",
    "Src/OpenAgentLib/ToolKernel.py",
)
compatibility = load_source_module(
    "openagent_tool_compatibility_for_kernel_test",
    "Src/OpenAgentLib/ToolCompatibility.py",
)
system_tools = load_source_module(
    "openagent_system_tools_for_kernel_test",
    "Src/OpenAgentLib/SystemPlugins/base.py",
)
packaged_system_tools = importlib.import_module("OpenAgentLib.SystemPlugins.base")


def _spec(
    canonical_id: str = "sample.inspect",
    *,
    aliases: tuple[str, ...] = ("inspect",),
    input_schema: dict | None = None,
    api_version: str = kernel.TOOL_API_VERSION,
    schema_version: str = kernel.TOOL_SCHEMA_VERSION,
) -> object:
    return kernel.ToolSpec(
        canonical_id=canonical_id,
        aliases=aliases,
        input_schema=input_schema
        or {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        output_schema={"type": "object", "additionalProperties": True},
        api_version=api_version,
        schema_version=schema_version,
        capabilities={"read-only"},
        confirmation="none",
        concurrency="parallel-read",
        idempotency="idempotent",
        migration_disposition="migrate",
    )


def _discover_real_system_tools() -> dict:
    prefix = "OpenAgentLib.SystemPlugins._loaded."
    cached = {
        name: sys.modules.pop(name)
        for name in tuple(sys.modules)
        if name.startswith(prefix)
    }
    try:
        return packaged_system_tools.discover_system_tools()
    finally:
        for name in tuple(sys.modules):
            if name.startswith(prefix):
                sys.modules.pop(name)
        sys.modules.update(cached)


def test_valid_nested_schema_creates_a_frozen_call() -> None:
    spec = _spec(
        input_schema={
            "type": "object",
            "properties": {
                "request": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "labels": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["fast", "safe"]},
                        },
                        "limit": {"type": "number"},
                        "optional": {"type": "null"},
                    },
                    "required": ["enabled", "labels"],
                    "additionalProperties": False,
                }
            },
            "required": ["request"],
            "additionalProperties": False,
        }
    )
    registry = kernel.ToolRegistry([spec])
    arguments = {
        "request": {
            "enabled": True,
            "labels": ["fast", "safe"],
            "limit": 1.5,
            "optional": None,
        }
    }

    call = registry.create_call(
        call_id="call-1",
        requested_name=" INSPECT ",
        arguments=arguments,
        context=kernel.ToolContext("trace-1", metadata={"source": ["test"]}),
    )

    arguments["request"]["labels"].append("changed")
    assert call.canonical_id == "sample.inspect"
    assert call.arguments["request"]["labels"] == ("fast", "safe")
    assert isinstance(call.arguments, MappingProxyType)
    with pytest.raises(TypeError):
        call.arguments["request"] = {}


def test_boolean_is_not_accepted_as_an_integer() -> None:
    registry = kernel.ToolRegistry([_spec()])

    with pytest.raises(kernel.ToolArgumentError) as error:
        registry.create_call(
            call_id="call-1",
            requested_name="sample.inspect",
            arguments={"name": "one", "count": True},
        )

    assert error.value.code is kernel.ToolErrorCode.INVALID_ARGUMENT
    assert error.value.field_path == ("count",)


@pytest.mark.parametrize(
    ("arguments", "path"),
    [({"name": "one", "unknown": "value"}, ("unknown",)), ({}, ("name",))],
)
def test_schema_rejects_unknown_and_missing_fields(arguments: dict, path: tuple[str, ...]) -> None:
    registry = kernel.ToolRegistry([_spec()])

    with pytest.raises(kernel.ToolArgumentError) as error:
        registry.create_call(
            call_id="call-1", requested_name="inspect", arguments=arguments
        )

    assert error.value.field_path == path
    assert error.value.canonical_id == "sample.inspect"
    assert error.value.requested_name == "inspect"


def test_registry_resolves_alias_to_the_same_spec_in_stable_order() -> None:
    zulu = _spec("zulu.read", aliases=("z",))
    alpha = _spec("alpha.read", aliases=("a",))
    registry = kernel.ToolRegistry([zulu, alpha])

    assert [spec.canonical_id for spec in registry.specs()] == ["alpha.read", "zulu.read"]
    assert registry.resolve(" z ") is zulu
    assert registry.resolve("zulu.read") is zulu
    assert registry.register(_spec("middle.read", aliases=("m",))).specs() == (
        alpha,
        _spec("middle.read", aliases=("m",)),
        zulu,
    )


def test_registry_rejects_alias_collisions_with_canonical_names() -> None:
    with pytest.raises(kernel.ToolRegistryError) as error:
        kernel.ToolRegistry(
            [_spec("alpha.read", aliases=("beta.read",)), _spec("beta.read", aliases=())]
        )

    assert error.value.code is kernel.ToolErrorCode.DUPLICATE_ALIAS
    assert error.value.requested_name == "beta.read"


def test_malformed_and_undeclared_names_are_typed_errors() -> None:
    with pytest.raises(kernel.ToolNameError) as malformed:
        _spec("not a canonical id", aliases=())
    assert malformed.value.code is kernel.ToolErrorCode.INVALID_NAME

    registry = kernel.ToolRegistry([_spec()])
    with pytest.raises(kernel.ToolUndeclaredAliasError) as undeclared:
        registry.resolve("not-declared")
    assert undeclared.value.code is kernel.ToolErrorCode.UNDECLARED_ALIAS
    assert undeclared.value.requested_name == "not-declared"


def test_public_schema_validation_rejects_incompatible_enums() -> None:
    with pytest.raises(kernel.ToolSchemaError) as enum_error:
        kernel.validate_schema({"type": "integer", "enum": [True]})
    assert enum_error.value.code is kernel.ToolErrorCode.INVALID_SCHEMA

    with pytest.raises(kernel.ToolSchemaError):
        kernel.validate_arguments({"type": "not-a-type"}, {})


def test_api_and_schema_version_mismatches_are_rejected() -> None:
    spec = _spec()
    registry = kernel.ToolRegistry([spec])

    with pytest.raises(kernel.ToolVersionError) as call_error:
        registry.create_call(
            call_id="call-1",
            requested_name="inspect",
            arguments={"name": "one"},
            api_version="1",
        )
    assert call_error.value.code is kernel.ToolErrorCode.API_VERSION_MISMATCH

    with pytest.raises(kernel.ToolVersionError) as registry_error:
        kernel.ToolRegistry([_spec(api_version="1")])
    assert registry_error.value.code is kernel.ToolErrorCode.API_VERSION_MISMATCH

    with pytest.raises(kernel.ToolVersionError) as schema_error:
        registry.create_call(
            call_id="call-1",
            requested_name="inspect",
            arguments={"name": "one"},
            schema_version="1",
        )
    assert schema_error.value.code is kernel.ToolErrorCode.SCHEMA_VERSION_MISMATCH


def test_spec_and_context_deep_copy_mutable_inputs() -> None:
    schema = {
        "type": "object",
        "properties": {"values": {"type": "array", "items": {"type": "integer"}}},
        "required": ["values"],
        "additionalProperties": False,
    }
    metadata = {"nested": {"values": [1]}}
    spec = _spec(input_schema=schema)
    context = kernel.ToolContext("trace-1", metadata=metadata)

    schema["properties"]["values"]["items"]["type"] = "string"
    metadata["nested"]["values"].append(2)

    assert spec.input_schema["properties"]["values"]["items"]["type"] == "integer"
    assert context.metadata["nested"]["values"] == (1,)
    with pytest.raises(TypeError):
        spec.input_schema["properties"] = {}
    with pytest.raises(TypeError):
        context.metadata["nested"] = {}


def test_result_and_trace_are_pure_immutable_value_objects() -> None:
    error = kernel.ToolError(kernel.ToolErrorCode.INVALID_ARGUMENT, "bad input")
    result = kernel.ToolResult("call-1", "timed_out", error=error, retryable=True)
    now = datetime(2026, 8, 21, tzinfo=timezone.utc)
    trace = kernel.ToolTrace(
        call_id="call-1",
        correlation_id="trace-1",
        state="timed_out",
        created_at=now,
        updated_at=now,
        events=(kernel.ToolTraceEvent("created", now, {"nested": ["one"]}),),
    )

    assert result.status is kernel.ToolResultStatus.TIMED_OUT
    assert result.retryable is True
    assert trace.events[0].details["nested"] == ("one",)


def test_system_tool_adapter_uses_matrix_and_rejects_descriptor_drift() -> None:
    invoked = False

    def handler(*_args, **_kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("adapting metadata must not invoke handlers")

    descriptor = system_tools.SystemTool(
        tool_class="utility",
        name="list_tools",
        handler=handler,
        docs={"desc": "List available tools"},
    )
    adapter = kernel.SystemToolAdapter(compatibility.compatibility_matrix())
    spec = adapter.to_spec(descriptor)

    entry = next(
        item
        for item in compatibility.compatibility_matrix()
        if item.canonical_id == "utility.list_tools"
    )
    assert not invoked
    assert spec.capabilities == frozenset({entry.capability_class})
    assert spec.confirmation.value == entry.confirmation_class
    assert spec.concurrency.value == entry.concurrency_class
    assert spec.idempotency.value == entry.idempotency_class
    assert spec.migration_disposition.value == entry.migration_disposition

    drifted = system_tools.SystemTool(
        tool_class="utility",
        name="list_tools",
        docs={"desc": "List available tools"},
        aliases=("unexpected",),
    )
    with pytest.raises(kernel.SystemToolAdapterError) as error:
        adapter.to_spec(drifted)
    assert error.value.code is kernel.ToolErrorCode.ADAPTER_DRIFT

    unknown = system_tools.SystemTool(
        tool_class="utility", name="not_in_matrix", docs={"desc": "Missing"}
    )
    with pytest.raises(kernel.SystemToolAdapterError):
        adapter.to_spec(unknown)


def test_system_tool_adapter_batches_real_alias_indexed_discovery() -> None:
    discovered = _discover_real_system_tools()
    adapter = kernel.SystemToolAdapter(compatibility.compatibility_matrix())

    assert len(discovered) == 42
    alias = next(key for key, tool in discovered.items() if key != tool.full_name)
    specs = adapter.to_specs(discovered)
    registry = adapter.to_registry(discovered)

    assert len(specs) == 39
    assert [spec.canonical_id for spec in specs] == sorted(
        spec.canonical_id for spec in specs
    )
    assert registry.specs() == specs
    assert registry.resolve(alias) is registry.resolve(discovered[alias].full_name)


def test_system_tool_adapter_batch_rejects_invalid_keys_and_conflicts() -> None:
    adapter = kernel.SystemToolAdapter(compatibility.compatibility_matrix())
    aliases = ("context.read_tool_output", "tool_output.read")
    first = system_tools.SystemTool(
        tool_class="context",
        name="tool_output",
        docs={"desc": "First descriptor"},
        aliases=aliases,
    )
    conflicting = system_tools.SystemTool(
        tool_class="context",
        name="tool_output",
        docs={"desc": "Conflicting descriptor"},
        aliases=aliases,
    )

    with pytest.raises(kernel.SystemToolAdapterError) as key_error:
        adapter.to_specs({"not-a-declared-name": first})
    assert key_error.value.code is kernel.ToolErrorCode.ADAPTER_DRIFT

    with pytest.raises(kernel.SystemToolAdapterError) as conflict_error:
        adapter.to_specs(
            {
                "context.tool_output": first,
                "context.read_tool_output": conflicting,
            }
        )
    assert conflict_error.value.code is kernel.ToolErrorCode.ADAPTER_DRIFT
