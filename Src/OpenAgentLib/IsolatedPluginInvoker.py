# SPDX-License-Identifier: MIT
"""Executor host adapter for the reviewed sibling v2 plugin package."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import uuid4

from .InstalledPluginRegistry import (
    InstalledPluginRecord,
    InstalledPluginRegistry,
    InstalledPluginStatus,
)
from .PluginCapabilities import (
    CapabilityErrorCode,
    CapabilityRequest,
    CapabilityResponse,
)
from .PluginDiscovery import StaticPluginSource
from .PluginHost import PluginHost, PluginHostOutcome, PluginHostRequest, SandboxMount
from .ToolKernel import ToolCall
from .ToolPolicy import ToolPolicyRequest


class _InvocationStageError(RuntimeError):
    """Marks an adapter fault for executor-only internal diagnostics."""

    def __init__(self, stage: str, message: str) -> None:
        self.tool_failure_stage = stage
        super().__init__(message)


class IsolatedPluginInvoker:
    """Run a statically admitted sibling handler in Bubblewrap only."""

    def __init__(
        self,
        host: PluginHost,
        sources: dict[str, StaticPluginSource],
        *,
        plugin_root: Path,
        openagent_source: Path,
        capability_handler: Callable[
            [ToolCall, ToolPolicyRequest, CapabilityRequest],
            CapabilityResponse | Awaitable[CapabilityResponse],
        ],
        registry: InstalledPluginRegistry | None = None,
    ) -> None:
        self._host = host
        self._plugin_root = Path(plugin_root).resolve()
        self._sources: dict[str, StaticPluginSource] = {}
        self._module_paths: dict[str, str] = {}
        self._records: dict[str, InstalledPluginRecord | None] = {}
        for module, source in sources.items():
            self.register_source(module, source)
        self._openagent_source = Path(openagent_source).resolve()
        self._capability_handler = capability_handler
        self._registry = registry
        self._active_calls: dict[str, tuple[ToolCall, ToolPolicyRequest]] = {}
        self._invocation_tasks: dict[tuple[str, int, str], asyncio.Task[object]] = {}

    def validate_source(
        self,
        module: str,
        source: StaticPluginSource,
        *,
        record: InstalledPluginRecord | None = None,
    ) -> tuple[str, str]:
        """Validate an isolated source binding without changing live state."""

        module = str(module or "").strip()
        source_path = Path(source.path).resolve()
        if not module or source_path.suffix != ".py" or source_path.stem != module:
            raise ValueError("plugin module must match the admitted source filename")
        if source_path.stem.startswith("_"):
            raise ValueError(
                "plugin helper modules cannot be admitted as callable sources"
            )
        try:
            relative = source_path.relative_to(self._plugin_root)
        except ValueError as exc:
            raise ValueError(
                f"plugin source {source_path} is outside {self._plugin_root}"
            ) from exc
        parts = relative.with_suffix("").parts
        if len(parts) < 2:
            raise ValueError("plugin source must be inside an importable package")
        if any(not part.isidentifier() for part in parts):
            raise ValueError("plugin source path contains an unsafe module component")
        package = self._plugin_root
        for part in parts[:-1]:
            package /= part
            if not (package / "__init__.py").is_file():
                raise ValueError("plugin source package is missing __init__.py")
        if record is not None and (
            record.source.digest != source.digest or not record.enabled
        ):
            raise ValueError("plugin record does not admit this executable source")
        return module, ".".join(parts)

    def register_source(
        self,
        module: str,
        source: StaticPluginSource,
        *,
        record: InstalledPluginRecord | None = None,
    ) -> None:
        """Admit one already-validated source for isolated execution."""

        validated = self.validate_source(module, source, record=record)
        self.register_validated_source(validated, source, record=record)

    def register_validated_source(
        self,
        validated: tuple[str, str],
        source: StaticPluginSource,
        *,
        record: InstalledPluginRecord | None = None,
    ) -> None:
        """Publish a source that ``validate_source`` already accepted."""

        module, module_path = validated
        self._sources[module] = source
        self._module_paths[module] = module_path
        self._records[module] = record

    def unregister_source(self, module: str) -> StaticPluginSource | None:
        """Remove a source that is no longer statically admitted."""
        module = str(module or "").strip()
        self._module_paths.pop(module, None)
        self._records.pop(module, None)
        return self._sources.pop(module, None)

    def snapshot_source(
        self, module: str
    ) -> tuple[StaticPluginSource, InstalledPluginRecord | None] | None:
        """Return an opaque registration snapshot for transactional replacement."""

        module = str(module or "").strip()
        source = self._sources.get(module)
        return (source, self._records.get(module)) if source is not None else None

    def restore_source(
        self,
        module: str,
        snapshot: tuple[StaticPluginSource, InstalledPluginRecord | None],
    ) -> None:
        """Restore a registration removed after a failed replacement attempt."""

        source, record = snapshot
        self.register_source(module, source, record=record)

    async def invoke(
        self,
        call: ToolCall,
        policy_request: ToolPolicyRequest,
        *,
        retryable: bool,
    ) -> PluginHostOutcome:
        module = call.spec.source_module
        source = self._sources.get(module)
        if source is None:
            raise _InvocationStageError("admission", "plugin source was not admitted")
        module_path = self._module_paths[module]
        record = self._records.get(module)
        task_key: tuple[str, int, str] | None = None
        if record is not None and self._registry is not None:
            current = self._registry.get(record.plugin_id)
            if (
                current.generation != record.generation
                or current.status is not InstalledPluginStatus.ACTIVE
            ):
                raise _InvocationStageError("admission", "plugin is not active")
            self._registry.own_task(
                record.plugin_id,
                call.call_id,
                expected_generation=record.generation,
            )
            current_task = asyncio.current_task()
            if current_task is None:
                self._registry.release_task(
                    record.plugin_id,
                    call.call_id,
                    expected_generation=record.generation,
                )
                raise _InvocationStageError(
                    "admission", "plugin invocation requires an asyncio task"
                )
            task_key = (record.plugin_id, record.generation, call.call_id)
            self._invocation_tasks[task_key] = current_task
        request = PluginHostRequest(
            request_id=f"plugin-{uuid4().hex}",
            call_id=call.call_id,
            operation="plugin_call",
            payload={
                "module": module_path,
                "plugin_id": (
                    record.plugin_id if record is not None else f"openagent.{module}"
                ),
                "plugin_version": (
                    record.manifest.version if record is not None else "2.0.0"
                ),
                "canonical_tool_id": call.canonical_id,
                "entrypoint": f"{module_path}.HANDLERS",
                "source_sha256": source.digest,
                "arguments": dict(call.arguments),
                "context": {
                    "correlation_id": (
                        call.context.correlation_id if call.context else call.call_id
                    ),
                    "actor_id": call.context.actor_id if call.context else None,
                    "metadata": dict(call.context.metadata) if call.context else {},
                },
                "grant_id": f"grant-{call.call_id}",
            },
            retryable=retryable,
        )
        self._active_calls[call.call_id] = (call, policy_request)
        try:
            return await self._host.call(
                request,
                mounts=(
                    SandboxMount(self._plugin_root, "/mnt/pluginroot", True),
                    SandboxMount(self._openagent_source, "/mnt/openagent", True),
                ),
                capability_handler=self._handle_capability_request,
            )
        finally:
            self._active_calls.pop(call.call_id, None)
            if (
                task_key is not None
                and record is not None
                and self._registry is not None
            ):
                self._invocation_tasks.pop(task_key, None)
                self._registry.release_task(
                    record.plugin_id,
                    call.call_id,
                    expected_generation=record.generation,
                )

    async def quiesce(self, plugin_id: str, generation: int) -> None:
        """Cancel and reap every live invocation for one plugin generation."""

        tasks = tuple(
            task
            for (
                owned_plugin_id,
                owned_generation,
                _call_id,
            ), task in self._invocation_tasks.items()
            if owned_plugin_id == plugin_id and owned_generation == generation
        )
        current_task = asyncio.current_task()
        for task in tasks:
            if task is not current_task:
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _handle_capability_request(
        self, request: CapabilityRequest
    ) -> CapabilityResponse:
        active = self._active_calls.get(request.call_id)
        if active is None:
            return CapabilityResponse.denied(request, CapabilityErrorCode.DENIED)
        call, policy_request = active
        try:
            response = self._capability_handler(call, policy_request, request)
            if hasattr(response, "__await__"):
                response = await response
            return response
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise _InvocationStageError(
                "capability", "capability handler failed"
            ) from error
