from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
from threading import Event, Thread

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "Src"))

from OpenAgentLib.InstalledPluginRegistry import (  # noqa: E402
    InstalledPluginCollisionError,
    InstalledPluginManifest,
    InstalledPluginMissingError,
    InstalledPluginRecord,
    InstalledPluginRecordError,
    InstalledPluginRegistry,
    InstalledPluginStaleGenerationError,
    InstalledPluginStatus,
    InstalledPluginTool,
    InstalledPluginTransitionError,
    InstalledPluginSource,
)


def _record(
    tmp_path: Path,
    *,
    plugin_id: str = "example.plugin",
    source_name: str = "example_plugin.py",
    digest: str = "a" * 64,
    tool_id: str = "example.run",
    aliases: tuple[str, ...] = ("example",),
    metadata: object | None = None,
) -> InstalledPluginRecord:
    return InstalledPluginRecord(
        plugin_id=plugin_id,
        source=InstalledPluginSource((tmp_path / source_name).resolve(), digest),
        manifest=InstalledPluginManifest(
            manifest_version="2",
            api_version="2",
            version="1.0.0",
            entrypoint="example_plugin.HANDLERS",
            display_name="Example plugin",
            metadata=(
                metadata
                if metadata is not None
                else {"author": "Ada", "presentation": {"tags": ["demo"]}}
            ),
            capabilities=frozenset({"network"}),
            tools=(
                InstalledPluginTool(
                    canonical_id=tool_id,
                    aliases=aliases,
                    capabilities=frozenset({"network"}),
                ),
            ),
        ),
    )


def test_lookup_normalizes_and_returns_immutable_snapshots(tmp_path: Path) -> None:
    registry = InstalledPluginRegistry()
    record = _record(
        tmp_path,
        plugin_id=" Example.Plugin ",
        tool_id=" Example.Run ",
        aliases=(" Example-Alias ",),
    )

    installed = registry.install(record)

    assert installed.plugin_id == "example.plugin"
    assert installed.source_path == str((tmp_path / "example_plugin.py").resolve())
    assert installed.source_digest == "a" * 64
    assert installed.tools[0].canonical_id == "example.run"
    assert installed.aliases == ("example-alias",)
    assert registry.get("EXAMPLE.PLUGIN") is installed
    assert registry.snapshot() == (installed,)
    assert registry.snapshot(state="installed") == (installed,)
    assert registry.snapshot(state="enabled") == (installed,)

    with pytest.raises(TypeError):
        installed.metadata["author"] = "Grace"  # type: ignore[index]
    with pytest.raises(TypeError):
        installed.metadata["presentation"]["tags"] = ()  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        installed.enabled = False  # type: ignore[misc]


def test_record_rejects_non_json_metadata_and_noncanonical_source(
    tmp_path: Path,
) -> None:
    with pytest.raises(InstalledPluginRecordError) as non_json:
        _record(tmp_path, metadata={"unsupported": object()})
    assert non_json.value.code.value == "invalid_record"

    with pytest.raises(InstalledPluginRecordError) as relative:
        InstalledPluginSource("relative_plugin.py", "a" * 64)
    assert relative.value.code.value == "invalid_record"

    unnormalized_path = tmp_path / "nested" / ".." / "plugin.py"
    with pytest.raises(InstalledPluginRecordError) as unnormalized:
        InstalledPluginSource(str(unnormalized_path), "a" * 64)
    assert unnormalized.value.code.value == "invalid_record"


def test_record_rejects_cyclic_json_metadata_and_schemas(tmp_path: Path) -> None:
    metadata: dict[str, object] = {}
    metadata["self"] = metadata
    with pytest.raises(InstalledPluginRecordError):
        _record(tmp_path, metadata=metadata)

    schema: dict[str, object] = {}
    schema["self"] = schema
    with pytest.raises(InstalledPluginRecordError):
        InstalledPluginTool(
            canonical_id="example.run",
            capabilities=frozenset({"network"}),
            input_schema=schema,
        )


def test_registry_rejects_source_tool_and_alias_collisions(tmp_path: Path) -> None:
    registry = InstalledPluginRegistry()
    first = registry.install(_record(tmp_path))

    with pytest.raises(InstalledPluginCollisionError):
        registry.install(
            _record(
                tmp_path,
                plugin_id="other.plugin",
                digest="b" * 64,
                tool_id="other.run",
                aliases=("other",),
            )
        )
    with pytest.raises(InstalledPluginCollisionError):
        registry.install(
            _record(
                tmp_path,
                plugin_id="other.plugin",
                source_name="other_plugin.py",
                tool_id="example.run",
                aliases=("other",),
            )
        )
    with pytest.raises(InstalledPluginCollisionError):
        registry.install(
            _record(
                tmp_path,
                plugin_id="other.plugin",
                source_name="other_plugin.py",
                tool_id="other.run",
                aliases=("example",),
            )
        )
    with pytest.raises(InstalledPluginCollisionError):
        registry.install(
            _record(
                tmp_path,
                plugin_id="example.run",
                source_name="plugin_id_collision.py",
                tool_id="other.run",
                aliases=("other",),
            )
        )
    with pytest.raises(InstalledPluginCollisionError):
        registry.install(
            _record(
                tmp_path,
                plugin_id="other.plugin",
                source_name="tool_id_collision.py",
                tool_id="example.plugin",
                aliases=("other",),
            )
        )

    assert registry.get(first.plugin_id) is first
    assert registry.snapshot() == (first,)


def test_valid_and_invalid_lifecycle_transitions(tmp_path: Path) -> None:
    registry = InstalledPluginRegistry()
    installed = registry.install(_record(tmp_path))

    with pytest.raises(InstalledPluginTransitionError):
        registry.begin_unload(
            installed.plugin_id, expected_generation=installed.generation
        )

    active = registry.activate(
        installed.plugin_id, expected_generation=installed.generation
    )
    unloading = registry.begin_unload(
        active.plugin_id, expected_generation=active.generation
    )
    restored = registry.complete_unload(
        unloading.plugin_id, expected_generation=unloading.generation
    )

    assert [installed.status, active.status, unloading.status, restored.status] == [
        InstalledPluginStatus.INSTALLED,
        InstalledPluginStatus.ACTIVE,
        InstalledPluginStatus.UNLOADING,
        InstalledPluginStatus.INSTALLED,
    ]
    assert [
        installed.generation,
        active.generation,
        unloading.generation,
        restored.generation,
    ] == sorted(
        [
            installed.generation,
            active.generation,
            unloading.generation,
            restored.generation,
        ]
    )


def test_admit_install_and_replace_reuses_only_its_own_names(tmp_path: Path) -> None:
    registry = InstalledPluginRegistry()
    candidate = _record(tmp_path)
    admitted = registry.admit(candidate)
    installed = registry.install(candidate, expected_generation=admitted.generation)
    replacement = _record(
        tmp_path,
        source_name="replacement_plugin.py",
        digest="b" * 64,
    )

    replaced = registry.replace(replacement, expected_generation=installed.generation)

    assert admitted.status is InstalledPluginStatus.ADMITTED
    assert replaced.source_path.endswith("replacement_plugin.py")
    assert replaced.tool_names == installed.tool_names
    assert replaced.generation > installed.generation

    active = registry.activate(
        replaced.plugin_id, expected_generation=replaced.generation
    )
    with pytest.raises(InstalledPluginTransitionError):
        registry.replace(replacement, expected_generation=active.generation)
    assert registry.get(active.plugin_id) is active


def test_install_cannot_swap_an_admitted_declaration(tmp_path: Path) -> None:
    registry = InstalledPluginRegistry()
    admitted = registry.admit(_record(tmp_path))
    swapped = _record(
        tmp_path,
        source_name="unadmitted_plugin.py",
        digest="b" * 64,
    )

    with pytest.raises(InstalledPluginRecordError):
        registry.install(swapped, expected_generation=admitted.generation)

    assert registry.get(admitted.plugin_id) is admitted


def test_stale_generation_cannot_change_newer_record(tmp_path: Path) -> None:
    registry = InstalledPluginRegistry()
    installed = registry.install(_record(tmp_path))
    active = registry.activate(
        installed.plugin_id, expected_generation=installed.generation
    )

    with pytest.raises(InstalledPluginStaleGenerationError):
        registry.set_enabled(
            active.plugin_id, False, expected_generation=installed.generation
        )

    assert registry.get(active.plugin_id) is active


def test_disable_enable_and_fail_are_explicit_state_changes(tmp_path: Path) -> None:
    registry = InstalledPluginRegistry()
    installed = registry.install(_record(tmp_path))
    disabled = registry.set_enabled(
        installed.plugin_id, False, expected_generation=installed.generation
    )
    assert registry.snapshot(state="enabled") == ()
    reenabled = registry.set_enabled(
        disabled.plugin_id, True, expected_generation=disabled.generation
    )
    failed = registry.fail(
        reenabled.plugin_id,
        "host_failed",
        "isolated host exited before registration",
        expected_generation=reenabled.generation,
    )

    assert disabled.enabled is False
    assert disabled.status is InstalledPluginStatus.DISABLED
    assert reenabled.enabled is True
    assert reenabled.status is InstalledPluginStatus.INSTALLED
    assert failed.status is InstalledPluginStatus.FAILED
    assert failed.failure_code == "host_failed"
    assert failed.failure_reason == "isolated host exited before registration"


def test_remove_reinstall_allocates_a_new_generation(tmp_path: Path) -> None:
    registry = InstalledPluginRegistry()
    candidate = _record(tmp_path)
    installed = registry.install(candidate)
    removed = registry.remove(
        installed.plugin_id, expected_generation=installed.generation
    )
    reinstalled = registry.install(candidate)

    assert removed.status is InstalledPluginStatus.REMOVED
    assert removed.generation > installed.generation
    assert reinstalled.generation > removed.generation
    assert registry.get(reinstalled.plugin_id) is reinstalled

    with pytest.raises(InstalledPluginMissingError):
        InstalledPluginRegistry().get("example.plugin")


def test_replace_active_publishes_one_generation_and_clears_actions(
    tmp_path: Path,
) -> None:
    registry = InstalledPluginRegistry()
    installed = registry.install(_record(tmp_path))
    active = registry.activate(
        installed.plugin_id, expected_generation=installed.generation
    )
    action_owner = registry.bind_action(
        active.plugin_id, "replace-action", expected_generation=active.generation
    )
    replacement = registry.replace_active(
        _record(tmp_path, source_name="replacement.py", digest="b" * 64),
        expected_generation=action_owner.generation,
    )

    assert replacement.status is InstalledPluginStatus.ACTIVE
    assert replacement.generation == active.generation + 1
    assert replacement.action_ids == frozenset()
    with pytest.raises(InstalledPluginMissingError):
        registry.get_action_owner(
            "replace-action", expected_generation=active.generation
        )


def test_remove_active_requires_quiescence_and_returns_retired_tombstone(
    tmp_path: Path,
) -> None:
    registry = InstalledPluginRegistry()
    installed = registry.install(_record(tmp_path))
    active = registry.activate(
        installed.plugin_id, expected_generation=installed.generation
    )
    owned = registry.own_task(
        active.plugin_id, "active-task", expected_generation=active.generation
    )

    with pytest.raises(InstalledPluginTransitionError):
        registry.remove_active(owned.plugin_id, expected_generation=owned.generation)
    assert registry.get(owned.plugin_id) is owned

    released = registry.release_task(
        owned.plugin_id, "active-task", expected_generation=owned.generation
    )
    removed = registry.remove_active(
        released.plugin_id, expected_generation=released.generation
    )

    assert removed.status is InstalledPluginStatus.REMOVED
    assert removed.generation > released.generation
    with pytest.raises(InstalledPluginMissingError):
        registry.get(released.plugin_id)


def test_action_and_task_ownership_is_generation_safe(tmp_path: Path) -> None:
    registry = InstalledPluginRegistry()
    installed = registry.install(_record(tmp_path))
    active = registry.activate(
        installed.plugin_id, expected_generation=installed.generation
    )
    action_bound = registry.bind_action(
        active.plugin_id, "callback-1", expected_generation=active.generation
    )
    task_owned = registry.own_task(
        action_bound.plugin_id, "task-1", expected_generation=action_bound.generation
    )

    assert task_owned.callback_ids == frozenset({"callback-1"})
    assert task_owned.task_ids == frozenset({"task-1"})
    assert (
        registry.get_action_owner(
            "callback-1", expected_generation=task_owned.generation
        )
        is task_owned
    )
    assert (
        registry.get_task_owner("task-1", expected_generation=task_owned.generation)
        is task_owned
    )
    with pytest.raises(InstalledPluginTransitionError):
        registry.begin_unload(
            task_owned.plugin_id, expected_generation=task_owned.generation
        )

    task_released = registry.release_task(
        task_owned.plugin_id, "task-1", expected_generation=task_owned.generation
    )
    action_released = registry.release_action(
        task_released.plugin_id,
        "callback-1",
        expected_generation=task_released.generation,
    )

    assert action_released.task_ids == frozenset()
    assert action_released.callback_ids == frozenset()
    with pytest.raises(InstalledPluginMissingError):
        registry.get_action_owner(
            "callback-1", expected_generation=action_released.generation
        )


def test_concurrent_mutation_and_snapshot_reads_are_consistent(tmp_path: Path) -> None:
    registry = InstalledPluginRegistry()
    registry.install(_record(tmp_path))
    done = Event()
    errors: list[BaseException] = []

    def writer() -> None:
        try:
            for _ in range(100):
                current = registry.get("example.plugin")
                registry.set_enabled(
                    current.plugin_id,
                    not current.enabled,
                    expected_generation=current.generation,
                )
        except BaseException as exc:  # pragma: no cover - asserted after join
            errors.append(exc)
        finally:
            done.set()

    def reader() -> None:
        try:
            while not done.is_set():
                snapshot = registry.snapshot()
                if len(snapshot) != 1:
                    raise RuntimeError("snapshot exposed a partial registry")
                record = snapshot[0]
                if record.status not in {
                    InstalledPluginStatus.INSTALLED,
                    InstalledPluginStatus.DISABLED,
                }:
                    raise RuntimeError("snapshot exposed an invalid lifecycle state")
        except BaseException as exc:  # pragma: no cover - asserted after join
            errors.append(exc)

    writer_thread = Thread(target=writer)
    reader_thread = Thread(target=reader)
    reader_thread.start()
    writer_thread.start()
    writer_thread.join()
    reader_thread.join()

    assert errors == []
    assert len(registry.snapshot()) == 1
