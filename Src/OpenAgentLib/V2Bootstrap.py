# SPDX-License-Identifier: MIT
"""One authoritative v2 registry, policy, model boundary, and executor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable
from uuid import uuid4

from .PluginSDK import PluginManifest
from .RuntimeNativeSystemServices import RuntimeNativeSystemServices
from .SystemPlugins.native import build_native_system_tools
from .ToolCompatibility import TOOL_COMPATIBILITY_MATRIX
from .ToolExecutor import ToolExecutor, ToolHostInvoker
from .ToolKernel import (
    TOOL_API_VERSION,
    TOOL_SCHEMA_VERSION,
    ConcurrencyClass,
    ConfirmationRequirement,
    IdempotencyClass,
    MigrationDisposition,
    ToolContext,
    ToolRegistry,
    ToolSpec,
)
from .ToolModelBoundary import ModelBoundaryOutput, ToolModelBoundary
from .ToolPolicy import DEFAULT_TOOL_POLICY_CATALOG, ToolPolicyEngine, ToolPolicyRequest

_OPEN_OBJECT = {"type": "object", "additionalProperties": True}


def _sibling_specs() -> tuple[ToolSpec, ...]:
    """Reserve every frozen sibling v2 ID for isolated-host dispatch.

    The worker revalidates arguments and output against its loaded manifest. The
    parent deliberately derives only stable policy metadata from the frozen
    contract; it never imports sibling source merely to obtain declarations.
    """

    return tuple(
        ToolSpec(
            canonical_id=entry.canonical_id,
            aliases=entry.aliases,
            input_schema=_OPEN_OBJECT,
            output_schema=_OPEN_OBJECT,
            api_version=TOOL_API_VERSION,
            schema_version=TOOL_SCHEMA_VERSION,
            capabilities=frozenset({entry.capability_class}),
            confirmation=ConfirmationRequirement(entry.confirmation_class),
            concurrency=ConcurrencyClass(entry.concurrency_class),
            idempotency=IdempotencyClass(entry.idempotency_class),
            migration_disposition=MigrationDisposition(entry.migration_disposition),
            description=entry.canonical_id,
            source_family="plugin-v2",
            source_module=entry.source_module,
        )
        for entry in TOOL_COMPATIBILITY_MATRIX
        if entry.source_family == "sibling-plugin"
        and entry.migration_disposition == MigrationDisposition.MIGRATE.value
    )


@dataclass(frozen=True)
class V2ToolRuntime:
    """The sole runtime dependency for model tool calls after bootstrap."""

    registry: ToolRegistry
    policy: ToolPolicyEngine
    executor: ToolExecutor
    manifests: tuple[PluginManifest, ...]

    def boundary(self, context: ToolContext) -> ToolModelBoundary:
        return ToolModelBoundary(
            self.registry, call_id_factory=lambda: uuid4().hex, context=context
        )

    async def on_unload(self) -> None:
        """Release executor-owned work during module unload."""

        await self.executor.on_unload()

    async def execute_model_output(
        self,
        value: str,
        *,
        context: ToolContext,
        request_for: Callable[[object], ToolPolicyRequest],
    ) -> tuple[ModelBoundaryOutput, tuple[object, ...], tuple[object, ...]]:
        output = self.boundary(context).parse(value)
        if not output.calls:
            return output, (), ()
        requests = tuple(request_for(call) for call in output.calls)
        results, traces = await self.executor.execute_batch(output.calls, requests)
        return output, results, traces


def build_v2_tool_runtime(
    app: Any,
    manifests: Iterable[PluginManifest] = (),
    *,
    host_invoker: ToolHostInvoker | None = None,
) -> V2ToolRuntime:
    """Build one deterministic registry without legacy discovery or dispatch."""

    services = RuntimeNativeSystemServices(app)
    native = build_native_system_tools(services)
    manifest_items = tuple(manifests)
    declared_specs = tuple(
        spec
        for manifest in manifest_items
        for spec in (
            tool.to_tool_spec(source_module=manifest.plugin_id)
            for tool in manifest.tools
        )
    )
    specs = (*native.registry.specs(), *(declared_specs or _sibling_specs()))
    registry = ToolRegistry(specs)
    policy = ToolPolicyEngine(DEFAULT_TOOL_POLICY_CATALOG)
    runtime = V2ToolRuntime(
        registry,
        policy,
        ToolExecutor(
            registry, policy, native_handlers=native.handlers, host_invoker=host_invoker
        ),
        manifest_items,
    )
    services.bind_registry(registry)
    return runtime
