# SPDX-License-Identifier: MIT
"""Native v2 contracts for bundled system tools.

This module is intentionally independent from the legacy descriptor loader.  A
runtime adapter can implement :class:`NativeSystemToolServices` later, but the
registry and handlers here are already directly consumable by ``ToolExecutor``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from ..ToolCompatibility import TOOL_COMPATIBILITY_MATRIX, ToolCompatibility
from ..ToolKernel import (
    TOOL_API_VERSION,
    TOOL_SCHEMA_VERSION,
    ConcurrencyClass,
    ConfirmationRequirement,
    IdempotencyClass,
    MigrationDisposition,
    ToolCall,
    ToolRegistry,
    ToolSpec,
)

JsonObject = Mapping[str, Any]
ServiceMethod = Callable[[JsonObject], Awaitable[JsonObject]]
ServiceInvoker = Callable[
    ["NativeSystemToolServices", JsonObject], Awaitable[JsonObject]
]


class NativeSystemToolServices(Protocol):
    """Operation-specific services supplied by the eventual runtime adapter."""

    async def code_attach_result(self, arguments: JsonObject) -> JsonObject: ...
    async def code_choose_filename(self, arguments: JsonObject) -> JsonObject: ...
    async def code_generate_file(self, arguments: JsonObject) -> JsonObject: ...
    async def code_generate_mcub_module(self, arguments: JsonObject) -> JsonObject: ...
    async def code_read_docs(self, arguments: JsonObject) -> JsonObject: ...
    async def context_clear(self, arguments: JsonObject) -> JsonObject: ...
    async def context_discard(self, arguments: JsonObject) -> JsonObject: ...
    async def context_media_context(self, arguments: JsonObject) -> JsonObject: ...
    async def context_prune(self, arguments: JsonObject) -> JsonObject: ...
    async def context_regenerate(self, arguments: JsonObject) -> JsonObject: ...
    async def context_remember(self, arguments: JsonObject) -> JsonObject: ...
    async def context_reply_context(self, arguments: JsonObject) -> JsonObject: ...
    async def context_tool_output(self, arguments: JsonObject) -> JsonObject: ...
    async def skill_save(self, arguments: JsonObject) -> JsonObject: ...
    async def skills_activate(self, arguments: JsonObject) -> JsonObject: ...
    async def skills_export_md(self, arguments: JsonObject) -> JsonObject: ...
    async def skills_import_md(self, arguments: JsonObject) -> JsonObject: ...
    async def skills_install(self, arguments: JsonObject) -> JsonObject: ...
    async def skills_list(self, arguments: JsonObject) -> JsonObject: ...
    async def skills_read(self, arguments: JsonObject) -> JsonObject: ...
    async def skills_repo_list(self, arguments: JsonObject) -> JsonObject: ...
    async def skills_save_from_ai(self, arguments: JsonObject) -> JsonObject: ...
    async def thinking_note(self, arguments: JsonObject) -> JsonObject: ...
    async def todo_add(self, arguments: JsonObject) -> JsonObject: ...
    async def todo_clear(self, arguments: JsonObject) -> JsonObject: ...
    async def todo_close(self, arguments: JsonObject) -> JsonObject: ...
    async def todo_closeall(self, arguments: JsonObject) -> JsonObject: ...
    async def todo_current(self, arguments: JsonObject) -> JsonObject: ...
    async def todo_delete(self, arguments: JsonObject) -> JsonObject: ...
    async def todo_edit(self, arguments: JsonObject) -> JsonObject: ...
    async def utility_agent_log(self, arguments: JsonObject) -> JsonObject: ...
    async def utility_error_file(self, arguments: JsonObject) -> JsonObject: ...
    async def utility_list_tools(self, arguments: JsonObject) -> JsonObject: ...
    async def utility_placeholders(self, arguments: JsonObject) -> JsonObject: ...
    async def utility_plugin_docs(self, arguments: JsonObject) -> JsonObject: ...
    async def utility_random_template(self, arguments: JsonObject) -> JsonObject: ...
    async def utility_search_tool(self, arguments: JsonObject) -> JsonObject: ...
    async def utility_token_usage(self, arguments: JsonObject) -> JsonObject: ...
    async def utility_tool_help(self, arguments: JsonObject) -> JsonObject: ...


def _object(
    properties: Mapping[str, JsonObject] | None = None,
    required: tuple[str, ...] = (),
) -> JsonObject:
    return {
        "type": "object",
        "properties": dict(properties or {}),
        "required": list(required),
        "additionalProperties": False,
    }


_STRING = {"type": "string"}
_INTEGER = {"type": "integer"}
_BOOLEAN = {"type": "boolean"}
_STRING_LIST = {"type": "array", "items": _STRING}
_RESULT_SCHEMA = _object({"result": _STRING}, ("result",))
_EMPTY_SCHEMA = _object()

# Canonical v2 inputs replace the historical attrs/body grammar at the model
# boundary.  No handler parses legacy attributes or accepts undeclared fields.
_INPUT_SCHEMAS: Mapping[str, JsonObject] = {
    "code.attach_result": _EMPTY_SCHEMA,
    "code.choose_filename": _object({"name": _STRING}, ("name",)),
    "code.generate_file": _object(
        {"path": _STRING, "content": _STRING}, ("path", "content")
    ),
    "code.generate_mcub_module": _object(
        {"name": _STRING, "content": _STRING}, ("name", "content")
    ),
    "code.read_docs": _EMPTY_SCHEMA,
    "context.clear": _EMPTY_SCHEMA,
    "context.discard": _object({"targets": _STRING_LIST, "keep": _INTEGER}),
    "context.media_context": _EMPTY_SCHEMA,
    "context.prune": _object({"targets": _STRING_LIST, "keep": _INTEGER}),
    "context.regenerate": _EMPTY_SCHEMA,
    "context.remember": _object({"note": _STRING}, ("note",)),
    "context.reply_context": _EMPTY_SCHEMA,
    "context.tool_output": _object(
        {
            "path": _STRING,
            "latest": _BOOLEAN,
            "mode": {"type": "string", "enum": ["head", "tail", "all"]},
            "limit": _INTEGER,
            "offset": _INTEGER,
        }
    ),
    "skill.save": _object({"name": _STRING, "content": _STRING}, ("name", "content")),
    "skills.activate": _object({"query": _STRING}),
    "skills.export_md": _object({"name": _STRING}),
    "skills.import_md": _object(
        {"name": _STRING, "content": _STRING}, ("name", "content")
    ),
    "skills.install": _object({"name": _STRING}),
    "skills.list": _EMPTY_SCHEMA,
    "skills.read": _object({"name": _STRING}),
    "skills.repo_list": _EMPTY_SCHEMA,
    "skills.save_from_ai": _object(
        {"name": _STRING, "content": _STRING}, ("name", "content")
    ),
    "thinking.note": _object({"text": _STRING}, ("text",)),
    "todo.add": _object({"text": _STRING}, ("text",)),
    "todo.clear": _EMPTY_SCHEMA,
    "todo.close": _object({"id": _STRING}, ("id",)),
    "todo.closeall": _EMPTY_SCHEMA,
    "todo.current": _EMPTY_SCHEMA,
    "todo.delete": _object({"id": _STRING}, ("id",)),
    "todo.edit": _object(
        {
            "id": _STRING,
            "text": _STRING,
            "status": {"type": "string", "enum": ["open", "closed"]},
        },
        ("id", "text"),
    ),
    "utility.agent_log": _EMPTY_SCHEMA,
    "utility.error_file": _EMPTY_SCHEMA,
    "utility.list_tools": _EMPTY_SCHEMA,
    "utility.placeholders": _EMPTY_SCHEMA,
    "utility.plugin_docs": _object({"plugin": _STRING}),
    "utility.random_template": _EMPTY_SCHEMA,
    "utility.search_tool": _object({"query": _STRING}),
    "utility.token_usage": _EMPTY_SCHEMA,
    "utility.tool_help": _object({"tool": _STRING}, ("tool",)),
}


async def _invoke(
    services: NativeSystemToolServices, method: ServiceMethod, arguments: JsonObject
) -> JsonObject:
    return await method(arguments)


def _service_handler(
    method: Callable[[NativeSystemToolServices], ServiceMethod],
) -> ServiceInvoker:
    async def invoke(
        services: NativeSystemToolServices, arguments: JsonObject
    ) -> JsonObject:
        return await _invoke(services, method(services), arguments)

    return invoke


# These direct method references deliberately prevent legacy class/method-name
# dispatch or a dynamic getattr fallback from entering native execution.
_INVOKERS: Mapping[str, ServiceInvoker] = {
    "code.attach_result": _service_handler(lambda service: service.code_attach_result),
    "code.choose_filename": _service_handler(
        lambda service: service.code_choose_filename
    ),
    "code.generate_file": _service_handler(lambda service: service.code_generate_file),
    "code.generate_mcub_module": _service_handler(
        lambda service: service.code_generate_mcub_module
    ),
    "code.read_docs": _service_handler(lambda service: service.code_read_docs),
    "context.clear": _service_handler(lambda service: service.context_clear),
    "context.discard": _service_handler(lambda service: service.context_discard),
    "context.media_context": _service_handler(
        lambda service: service.context_media_context
    ),
    "context.prune": _service_handler(lambda service: service.context_prune),
    "context.regenerate": _service_handler(lambda service: service.context_regenerate),
    "context.remember": _service_handler(lambda service: service.context_remember),
    "context.reply_context": _service_handler(
        lambda service: service.context_reply_context
    ),
    "context.tool_output": _service_handler(
        lambda service: service.context_tool_output
    ),
    "skill.save": _service_handler(lambda service: service.skill_save),
    "skills.activate": _service_handler(lambda service: service.skills_activate),
    "skills.export_md": _service_handler(lambda service: service.skills_export_md),
    "skills.import_md": _service_handler(lambda service: service.skills_import_md),
    "skills.install": _service_handler(lambda service: service.skills_install),
    "skills.list": _service_handler(lambda service: service.skills_list),
    "skills.read": _service_handler(lambda service: service.skills_read),
    "skills.repo_list": _service_handler(lambda service: service.skills_repo_list),
    "skills.save_from_ai": _service_handler(
        lambda service: service.skills_save_from_ai
    ),
    "thinking.note": _service_handler(lambda service: service.thinking_note),
    "todo.add": _service_handler(lambda service: service.todo_add),
    "todo.clear": _service_handler(lambda service: service.todo_clear),
    "todo.close": _service_handler(lambda service: service.todo_close),
    "todo.closeall": _service_handler(lambda service: service.todo_closeall),
    "todo.current": _service_handler(lambda service: service.todo_current),
    "todo.delete": _service_handler(lambda service: service.todo_delete),
    "todo.edit": _service_handler(lambda service: service.todo_edit),
    "utility.agent_log": _service_handler(lambda service: service.utility_agent_log),
    "utility.error_file": _service_handler(lambda service: service.utility_error_file),
    "utility.list_tools": _service_handler(lambda service: service.utility_list_tools),
    "utility.placeholders": _service_handler(
        lambda service: service.utility_placeholders
    ),
    "utility.plugin_docs": _service_handler(
        lambda service: service.utility_plugin_docs
    ),
    "utility.random_template": _service_handler(
        lambda service: service.utility_random_template
    ),
    "utility.search_tool": _service_handler(
        lambda service: service.utility_search_tool
    ),
    "utility.token_usage": _service_handler(
        lambda service: service.utility_token_usage
    ),
    "utility.tool_help": _service_handler(lambda service: service.utility_tool_help),
}


@dataclass(frozen=True)
class NativeSystemTools:
    """The registry plus the canonical-only native handler map for an executor."""

    registry: ToolRegistry
    handlers: Mapping[str, Callable[[ToolCall], Awaitable[JsonObject]]]


def _system_entries() -> Mapping[str, ToolCompatibility]:
    entries = {
        entry.canonical_id: entry
        for entry in TOOL_COMPATIBILITY_MATRIX
        if entry.source_family == "system" and entry.migration_disposition == "migrate"
    }
    if set(entries) != set(_INPUT_SCHEMAS) or set(entries) != set(_INVOKERS):
        raise RuntimeError(
            "native system definitions must cover the frozen system matrix exactly"
        )
    return entries


def build_native_system_tools(services: NativeSystemToolServices) -> NativeSystemTools:
    """Build all bundled v2 specs and their direct executor handlers."""

    entries = _system_entries()
    specs: list[ToolSpec] = []
    handlers: dict[str, Callable[[ToolCall], Awaitable[JsonObject]]] = {}
    for canonical_id, entry in entries.items():
        specs.append(
            ToolSpec(
                canonical_id=canonical_id,
                aliases=entry.aliases,
                input_schema=_INPUT_SCHEMAS[canonical_id],
                output_schema=_RESULT_SCHEMA,
                api_version=TOOL_API_VERSION,
                schema_version=TOOL_SCHEMA_VERSION,
                capabilities=frozenset({entry.capability_class}),
                confirmation=ConfirmationRequirement(entry.confirmation_class),
                concurrency=ConcurrencyClass(entry.concurrency_class),
                idempotency=IdempotencyClass(entry.idempotency_class),
                migration_disposition=MigrationDisposition(entry.migration_disposition),
                description=canonical_id,
                source_family=entry.source_family,
                source_module=entry.source_module,
            )
        )
        invoker = _INVOKERS[canonical_id]

        async def handler(
            call: ToolCall, invoke: ServiceInvoker = invoker
        ) -> JsonObject:
            return await invoke(services, call.arguments)

        handlers[canonical_id] = handler
    return NativeSystemTools(ToolRegistry(specs), handlers)


__all__ = ["NativeSystemToolServices", "NativeSystemTools", "build_native_system_tools"]
