from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier, Thread

import pytest

from OpenAgentLib.InstalledPluginActions import (
    InstalledPluginActionError,
    InstalledPluginActionErrorCode,
    InstalledPluginActionStore,
)
from OpenAgentLib.InstalledPluginRegistry import (
    InstalledPluginManifest,
    InstalledPluginRecord,
    InstalledPluginRegistry,
    InstalledPluginSource,
    InstalledPluginStatus,
    InstalledPluginTool,
)


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2025, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value


def _active_record(tmp_path: Path):
    registry = InstalledPluginRegistry()
    record = registry.install(
        InstalledPluginRecord(
                plugin_id="example.plugin",
                source=InstalledPluginSource((tmp_path / "plugin.py").resolve(), "a" * 64),
                manifest=InstalledPluginManifest(
                manifest_version="2",
                api_version="2",
                version="1",
                entrypoint="plugin.HANDLERS",
                capabilities=frozenset({"network"}),
                tools=(
                    InstalledPluginTool(
                        canonical_id="example.run", capabilities=frozenset({"network"})
                    ),
                ),
                ),
        )
    )
    return registry, registry.activate(record.plugin_id, expected_generation=record.generation)


def test_actions_are_frozen_actor_bound_and_one_time(tmp_path: Path) -> None:
    registry, record = _active_record(tmp_path)
    store = InstalledPluginActionStore(clock=_Clock())
    action = store.issue(registry, record, actor_id=7, kind="disable", payload={"page": [2]})

    assert action.payload == {"page": (2,)}
    with pytest.raises(InstalledPluginActionError) as caught:
        store.consume(registry, action.token, actor_id=8, kind="disable")
    assert caught.value.code is InstalledPluginActionErrorCode.ACTOR_MISMATCH

    consumed, current = store.consume(registry, action.token, actor_id=7, kind="disable")
    assert consumed is action
    assert current.action_ids == frozenset()
    with pytest.raises(InstalledPluginActionError) as caught:
        store.consume(registry, action.token, actor_id=7, kind="disable")
    assert caught.value.code is InstalledPluginActionErrorCode.CONSUMED


def test_action_rejects_expiry_and_stale_generation(tmp_path: Path) -> None:
    registry, record = _active_record(tmp_path)
    clock = _Clock()
    store = InstalledPluginActionStore(clock=clock)
    expired = store.issue(registry, record, actor_id=7, kind="delete", ttl=timedelta(seconds=1))
    clock.value += timedelta(seconds=1)
    with pytest.raises(InstalledPluginActionError) as caught:
        store.consume(registry, expired.token, actor_id=7, kind="delete")
    assert caught.value.code is InstalledPluginActionErrorCode.EXPIRED

    fresh = store.issue(registry, record, actor_id=7, kind="delete")
    registry.begin_unload(record.plugin_id, expected_generation=record.generation)
    with pytest.raises(InstalledPluginActionError) as caught:
        store.consume(registry, fresh.token, actor_id=7, kind="delete")
    assert caught.value.code is InstalledPluginActionErrorCode.STALE_GENERATION


def test_concurrent_consumption_has_one_winner(tmp_path: Path) -> None:
    registry, record = _active_record(tmp_path)
    store = InstalledPluginActionStore(clock=_Clock())
    action = store.issue(registry, record, actor_id=7, kind="delete")
    barrier = Barrier(2)
    results: list[str] = []

    def consume() -> None:
        barrier.wait()
        try:
            store.consume(registry, action.token, actor_id=7, kind="delete")
            results.append("winner")
        except InstalledPluginActionError as exc:
            results.append(exc.code.value)

    threads = [Thread(target=consume), Thread(target=consume)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == ["consumed", "winner"]


def test_disabled_action_can_be_explicitly_authorized(tmp_path: Path) -> None:
    registry, record = _active_record(tmp_path)
    unloading = registry.set_enabled(record.plugin_id, False, expected_generation=record.generation)
    disabled = registry.complete_unload(
        record.plugin_id, expected_generation=unloading.generation
    )
    store = InstalledPluginActionStore(clock=_Clock())
    action = store.issue(
        registry,
        disabled,
        actor_id=7,
        kind="enable",
        statuses=frozenset({InstalledPluginStatus.DISABLED}),
    )
    _action, current = store.consume(registry, action.token, actor_id=7, kind="enable")
    assert current.status is InstalledPluginStatus.DISABLED
