# SPDX-License-Identifier: MIT
"""Focused bootstrap for the native v2 tool runtime.

This module composes the v2 primitives without coupling them to the legacy
agent loop.  Callers remain responsible for supplying the policy request that
describes the execution environment for each model turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .SystemPlugins.native import NativeSystemToolServices, build_native_system_tools
from .ToolExecutor import ToolExecutor
from .ToolKernel import ToolCall, ToolContext, ToolRegistry, ToolResult, ToolTrace
from .ToolModelBoundary import (
    ModelBoundaryLimits,
    ModelBoundaryOutput,
    ModelTurnKind,
    ToolModelBoundary,
)
from .ToolPolicy import (
    DEFAULT_TOOL_POLICY_CATALOG,
    ToolPolicyCatalog,
    ToolPolicyEngine,
    ToolPolicyRequest,
)

PolicyRequestProvider = ToolPolicyRequest | Callable[[ToolCall], ToolPolicyRequest]
SpillReference = Callable[[ToolResult, str], str]


@dataclass(frozen=True)
class V2RuntimeExecution:
    """One parsed model turn and any terminal native execution artifacts."""

    boundary_output: ModelBoundaryOutput
    results: tuple[ToolResult, ...] = ()
    traces: tuple[ToolTrace, ...] = ()
    rendered_results: tuple[str, ...] = ()


class ToolRuntimeV2:
    """Composition root for registry, policy, executor, and model boundary."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: ToolPolicyEngine,
        executor: ToolExecutor,
        boundary: ToolModelBoundary,
    ) -> None:
        if not isinstance(registry, ToolRegistry):
            raise TypeError("registry must be a ToolRegistry")
        if not isinstance(policy, ToolPolicyEngine):
            raise TypeError("policy must be a ToolPolicyEngine")
        if not isinstance(executor, ToolExecutor):
            raise TypeError("executor must be a ToolExecutor")
        if not isinstance(boundary, ToolModelBoundary):
            raise TypeError("boundary must be a ToolModelBoundary")
        if executor.registry is not registry or executor.policy is not policy:
            raise ValueError("executor must use the supplied registry and policy")
        self.registry = registry
        self.policy = policy
        self.executor = executor
        self.boundary = boundary

    @classmethod
    def from_native(
        cls,
        services: NativeSystemToolServices,
        *,
        context: ToolContext,
        call_id_factory: Callable[[], str],
        policy_catalog: ToolPolicyCatalog = DEFAULT_TOOL_POLICY_CATALOG,
        boundary_limits: ModelBoundaryLimits = ModelBoundaryLimits(),
        **executor_options: Any,
    ) -> "ToolRuntimeV2":
        """Build a complete runtime over the bundled native system tools."""
        native = build_native_system_tools(services)
        policy = ToolPolicyEngine(policy_catalog)
        executor = ToolExecutor(
            native.registry,
            policy,
            native_handlers=native.handlers,
            **executor_options,
        )
        boundary = ToolModelBoundary(
            native.registry,
            call_id_factory=call_id_factory,
            context=context,
            limits=boundary_limits,
        )
        return cls(native.registry, policy, executor, boundary)

    async def execute_model_output(
        self,
        model_output: Any,
        policy_request: PolicyRequestProvider | None = None,
        *,
        spill_reference: SpillReference | None = None,
    ) -> V2RuntimeExecution:
        """Parse a model turn, execute its calls, and render terminal results."""
        parsed = self.boundary.parse(model_output)
        if parsed.kind is not ModelTurnKind.TOOLS:
            return V2RuntimeExecution(parsed)
        if policy_request is None:
            raise ValueError("policy_request is required for tool calls")

        requests = tuple(
            policy_request(call) if callable(policy_request) else policy_request
            for call in parsed.calls
        )
        if any(not isinstance(request, ToolPolicyRequest) for request in requests):
            raise TypeError(
                "policy request provider must return ToolPolicyRequest values"
            )

        results, traces = await self.executor.execute_batch(parsed.calls, requests)
        rendered = tuple(
            self.boundary.render_result(result, spill_reference=spill_reference)
            for result in results
        )
        return V2RuntimeExecution(parsed, results, traces, rendered)


__all__ = ["PolicyRequestProvider", "ToolRuntimeV2", "V2RuntimeExecution"]
