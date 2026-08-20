from __future__ import annotations

import json

from conftest import ROOT

import sys

sys.path.insert(0, str(ROOT / "Src"))

from OpenAgentLib.ToolKernel import ToolContext, ToolError, ToolErrorCode, ToolResult, ToolResultStatus
from OpenAgentLib.ToolModelBoundary import (
    ModelBoundaryOutput,
    ModelBoundaryLimits,
    ModelToolErrorCode,
    ModelTurnKind,
    ParsedToolCall,
    ToolModelBoundary,
)


def _boundary(tool_registry_builder, tool_spec_builder, **limits):
    spec = tool_spec_builder(
        "sample.run",
        aliases=("run",),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "count": {"type": "integer"},
                "enabled": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        output_schema={"type": "object", "additionalProperties": True},
    )
    counter = iter(range(1, 100))
    return ToolModelBoundary(
        tool_registry_builder((spec,)),
        call_id_factory=lambda: f"call-{next(counter)}",
        context=ToolContext("correlation", "actor"),
        limits=ModelBoundaryLimits(**limits),
    )


def test_legacy_forms_normalize_to_registry_tool_calls(tool_registry_builder, tool_spec_builder) -> None:
    boundary = _boundary(tool_registry_builder, tool_spec_builder)
    values = (
        "<run command=\"ls\" count=\"2\" enabled=\"true\" tags='[\"a\"]'/>",
        '```tool_call\n{"tool":"run","args":{"command":"ls","count":2,"enabled":true,"tags":["a"]}}\n```',
        '{"name":"sample.run","arguments":{"command":"ls","count":2,"enabled":true,"tags":["a"]}}',
        '<|start|>assistant to=tool.run <|message|>{"args":{"command":"ls","count":2,"enabled":true,"tags":["a"]}}<|call|>',
    )
    for value in values:
        output = boundary.parse(value)
        assert output.kind is ModelTurnKind.TOOLS
        assert len(output.calls) == 1
        assert output.calls[0].canonical_id == "sample.run"
        assert output.calls[0].arguments == {"command": "ls", "count": 2, "enabled": True, "tags": ("a",)}


def test_xml_body_uses_only_declared_body_compatibility_field(tool_registry_builder, tool_spec_builder) -> None:
    boundary = _boundary(tool_registry_builder, tool_spec_builder)
    output = boundary.parse("<run>ls</run>")
    assert output.kind is ModelTurnKind.TOOLS
    assert output.calls[0].arguments == {"command": "ls"}


def test_malformed_ambiguous_and_unknown_calls_never_produce_calls(tool_registry_builder, tool_spec_builder) -> None:
    boundary = _boundary(tool_registry_builder, tool_spec_builder)
    cases = (
        ('```tool_call\n{"tool":"run","name":"sample.run","args":{"command":"ls"}}\n```', ModelToolErrorCode.AMBIGUOUS_PAYLOAD),
        ('<run command="ls" broken/>', ModelToolErrorCode.MALFORMED_ATTRIBUTES),
        ('{"tool":"missing.tool","args":{}}', ModelToolErrorCode.UNKNOWN_TOOL),
        ('{"tool":"run","args":{"command":"ls","extra":1}}', ModelToolErrorCode.INVALID_ARGUMENT),
        ('{"tool":"run","args":{"command":"ls"},"id":"model-id"}', ModelToolErrorCode.UNKNOWN_FIELD),
        ('```tool_call\n{bad}\n```', ModelToolErrorCode.MALFORMED_JSON),
    )
    for value, code in cases:
        output = boundary.parse(value)
        assert output.kind is ModelTurnKind.INVALID
        assert not output.calls
        assert output.errors[0].code is code


def test_rejects_oversized_deep_and_duplicate_factory_ids(tool_registry_builder, tool_spec_builder) -> None:
    boundary = _boundary(tool_registry_builder, tool_spec_builder, max_input_chars=20)
    assert boundary.parse("x" * 21).errors[0].code is ModelToolErrorCode.INPUT_TOO_LARGE

    deep = _boundary(tool_registry_builder, tool_spec_builder, max_depth=2)
    assert deep.parse('{"tool":"run","args":{"command":"ls","tags":[["x"]]}}').errors[0].code is ModelToolErrorCode.NESTING_TOO_DEEP

    spec = tool_spec_builder("sample.run", aliases=("run",), input_schema={"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"], "additionalProperties": False})
    duplicate = ToolModelBoundary(tool_registry_builder((spec,)), call_id_factory=lambda: "same", context=ToolContext("c"))
    result = duplicate.parse('```tool_call\n[{"tool":"run","args":{"command":"a"}},{"tool":"run","args":{"command":"b"}}]\n```')
    assert result.errors[0].code is ModelToolErrorCode.DUPLICATE_CALL_ID


def test_legacy_scalar_coercion_rejects_boolean_as_integer(tool_registry_builder, tool_spec_builder) -> None:
    boundary = _boundary(tool_registry_builder, tool_spec_builder)
    output = boundary.parse('<run command="ls" count="true"/>')
    assert output.kind is ModelTurnKind.INVALID
    assert output.errors[0].code is ModelToolErrorCode.INVALID_ARGUMENT


def test_explicit_final_is_not_accepted_when_tool_content_is_present(tool_registry_builder, tool_spec_builder) -> None:
    boundary = _boundary(tool_registry_builder, tool_spec_builder)
    assert boundary.parse("```final\nready\n```").kind is ModelTurnKind.FINAL
    output = boundary.parse("```final\nready\n```\n<run command=\"ls\"/>")
    assert output.kind is ModelTurnKind.TOOLS
    assert output.final_answer == ""


def test_docs_and_search_are_deterministic_and_do_not_expose_source_details(tool_registry_builder, tool_spec_builder) -> None:
    first = tool_spec_builder("alpha.read", aliases=("read",), description="Read alpha")
    second = tool_spec_builder("zeta.read", aliases=("zread",), description="Read zeta")
    boundary = ToolModelBoundary(tool_registry_builder((second, first)), call_id_factory=lambda: "id", context=ToolContext("c"))
    docs = boundary.tool_docs()
    assert [doc["id"] for doc in docs] == ["alpha.read", "zeta.read"]
    assert "source_module" not in docs[0]
    assert [doc["id"] for doc in boundary.search_tool_docs("read", limit=1)] == ["alpha.read"]


def test_result_rendering_redacts_errors_and_spills(tool_registry_builder, tool_spec_builder) -> None:
    boundary = _boundary(tool_registry_builder, tool_spec_builder, max_rendered_chars=150)
    result = ToolResult("call-1", ToolResultStatus.SUCCESS, output={"token": "secret", "nested": {"password": "hidden"}})
    rendered = json.loads(boundary.render_result(result))
    assert rendered["output"] == {"nested": {"password": "[REDACTED]"}, "token": "[REDACTED]"}

    large = ToolResult("call-2", ToolResultStatus.SUCCESS, output={"text": "x" * 200})
    spilled = json.loads(boundary.render_result(large, spill_reference=lambda _result, _text: "spill://result/2"))
    assert spilled == {"call_id": "call-2", "spill_ref": "spill://result/2", "status": "success"}

    error = ToolResult("call-3", ToolResultStatus.ERROR, error=ToolError(ToolErrorCode.HANDLER_FAILED, "stack/path/secret"), retryable=True)
    envelope = json.loads(boundary.render_result(error))
    assert envelope == {"call_id": "call-3", "error_code": "handler_failed", "retryable": True, "status": "error"}


def test_parsed_calls_and_boundary_collections_are_deeply_immutable(tool_registry_builder, tool_spec_builder) -> None:
    source = {"nested": {"items": [{"value": "fixed"}]}}
    parsed = ParsedToolCall("run", source, None, None, 0, "raw_json")
    source["nested"]["items"][0]["value"] = "mutated"
    assert parsed.arguments["nested"]["items"][0]["value"] == "fixed"
    try:
        parsed.arguments["nested"]["items"][0]["value"] = "nope"
    except TypeError:
        pass
    else:
        raise AssertionError("nested parsed arguments must be immutable")

    output = ModelBoundaryOutput(ModelTurnKind.TOOLS, parsed_calls=(parsed,))
    try:
        output.parsed_calls += (parsed,)
    except (AttributeError, TypeError):
        pass
    else:
        raise AssertionError("boundary collections must be immutable")


def test_malformed_tool_markers_never_fall_back_to_prose(tool_registry_builder, tool_spec_builder) -> None:
    boundary = _boundary(tool_registry_builder, tool_spec_builder)
    malformed = (
        "```tool_call\n{\"tool\":\"run\",\"args\":{\"command\":\"ls\"}}",
        "```tool_call {\"tool\":\"run\",\"args\":{\"command\":\"ls\"}}```",
        '<tool_call name="run"',
        "</tool_call>",
        '<run command="ls">',
        '<run command="ls"/>\n<tool_call',
        '<|start|>assistant to=tool.run <|message|>{"args":{"command":"ls"}}',
        '<|message|>{"args":{"command":"ls"}}<|call|>',
        'ordinary prose {"tool":"run","args":{"command":"ls"}}',
        '```json\n{"tool":"run","args":{"command":"ls"}}',
    )
    for value in malformed:
        output = boundary.parse(value)
        assert output.kind is ModelTurnKind.INVALID
        assert output.errors[0].code is ModelToolErrorCode.MALFORMED_BLOCK


def test_non_tool_prose_and_json_examples_remain_intermediate(tool_registry_builder, tool_spec_builder) -> None:
    boundary = _boundary(tool_registry_builder, tool_spec_builder)
    values = (
        "Explain the tool registry in plain prose.",
        '{"topic":"tools","arguments":"an explanation"}',
        'Here is ordinary JSON: {"topic":"tools","arguments":"an explanation"}.',
        '```json\n{"topic":"tools","arguments":"an explanation"}\n```',
    )
    for value in values:
        assert boundary.parse(value).kind is ModelTurnKind.INTERMEDIATE


def test_invalid_boundary_limits_are_rejected() -> None:
    for value in (0, -1, True, 1.5, 1_000_001):
        try:
            ModelBoundaryLimits(max_input_chars=value)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid limit must be rejected")
