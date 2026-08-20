# SPDX-License-Identifier: MIT
"""Executor host adapter for the reviewed sibling v2 plugin package."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import uuid4

from .PluginCapabilities import CapabilityRequest, CapabilityResponse
from .PluginDiscovery import StaticPluginSource
from .PluginHost import PluginHost, PluginHostOutcome, PluginHostRequest, SandboxMount
from .ToolKernel import ToolCall


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
            [CapabilityRequest], CapabilityResponse | Awaitable[CapabilityResponse]
        ],
    ) -> None:
        self._host = host
        self._sources = dict(sources)
        self._plugin_root = Path(plugin_root).resolve()
        self._openagent_source = Path(openagent_source).resolve()
        self._capability_handler = capability_handler

    async def invoke(self, call: ToolCall, *, retryable: bool) -> PluginHostOutcome:
        module = call.spec.source_module
        source = self._sources.get(module)
        if source is None:
            raise RuntimeError(f"plugin source {module!r} was not admitted")
        request = PluginHostRequest(
            request_id=f"plugin-{uuid4().hex}",
            call_id=call.call_id,
            operation="plugin_call",
            payload={
                "module": f"plugins.{module}",
                "plugin_id": f"openagent.{module}",
                "plugin_version": "2.0.0",
                "canonical_tool_id": call.canonical_id,
                "entrypoint": f"plugins.{module}.HANDLERS",
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
        return await self._host.call(
            request,
            mounts=(
                SandboxMount(self._plugin_root, "/mnt/pluginroot", True),
                SandboxMount(self._openagent_source, "/mnt/openagent", True),
            ),
            capability_handler=self._capability_handler,
        )
