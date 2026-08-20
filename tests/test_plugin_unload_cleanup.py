from __future__ import annotations

import asyncio

from OpenAgentLib.Plugin.PluginsEngine import _OpenAgentPluginSkillMixin


class _Harness(_OpenAgentPluginSkillMixin):
    def __init__(self) -> None:
        self._plugin_unload_tasks: set[asyncio.Task[object]] = set()


def test_plugin_unload_tasks_are_cancelled_reaped_and_cleanup_is_idempotent() -> None:
    async def scenario() -> None:
        harness = _Harness()
        cancelled = asyncio.Event()

        async def pending() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(pending())
        harness._plugin_unload_tasks.add(task)
        await asyncio.sleep(0)

        await harness._cancel_plugin_unload_tasks()
        assert cancelled.is_set()
        assert harness._plugin_unload_tasks == set()

        await harness._cancel_plugin_unload_tasks()
        assert harness._plugin_unload_tasks == set()

    asyncio.run(scenario())
