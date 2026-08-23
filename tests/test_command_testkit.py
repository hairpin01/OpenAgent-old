from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from command_testkit import (
    CommandTaskHarness,
    DeterministicClock,
    FakeCommandEvent,
    FakeInstalledPluginActionError,
    FakeInstalledPluginActionErrorCode,
    FakeInstalledPluginActionStore,
    build_installed_plugin_action,
    validate_installed_plugin_action,
)
from OpenAgentLib.InstalledPluginRegistry import (
    InstalledPluginManifest,
    InstalledPluginRecord,
    InstalledPluginRegistry,
    InstalledPluginSource,
    InstalledPluginTool,
)


def _record(tmp_path: Path, *, digest: str = "a" * 64) -> InstalledPluginRecord:
    return InstalledPluginRecord(
        plugin_id="example.plugin",
        source=InstalledPluginSource((tmp_path / "example_plugin.py").resolve(), digest),
        manifest=InstalledPluginManifest(
            manifest_version="2",
            api_version="2",
            version="1.0.0",
            entrypoint="example_plugin.HANDLERS",
            display_name="Example plugin",
            capabilities=frozenset({"network"}),
            tools=(
                InstalledPluginTool(
                    canonical_id="example.run",
                    capabilities=frozenset({"network"}),
                ),
            ),
        ),
    )


def _installed(tmp_path: Path) -> tuple[InstalledPluginRegistry, InstalledPluginRecord]:
    registry = InstalledPluginRegistry()
    return registry, registry.install(_record(tmp_path))


def _assert_action_error(
    code: FakeInstalledPluginActionErrorCode,
    action,
    registry: InstalledPluginRegistry,
    clock: DeterministicClock,
    *,
    actor_id: int = 42,
) -> None:
    with pytest.raises(FakeInstalledPluginActionError) as caught:
        validate_installed_plugin_action(
            action, registry, actor_id=actor_id, now=clock()
        )
    assert caught.value.code is code
    assert caught.value.action is action


@pytest.mark.asyncio
async def test_event_records_output_in_call_order_and_exposes_identity() -> None:
    event = FakeCommandEvent(
        ".plugins",
        chat_id=11,
        sender_id=22,
        msg_id=33,
        out=False,
        admin=False,
    )

    edited = await event.edit("first", parse_mode="html")
    replied = await event.reply("second", link_preview=False)
    responded = await event.respond("third")

    assert edited is replied is responded is event
    assert [call.operation for call in event.calls] == ["edit", "reply", "respond"]
    assert event.calls[0].args == ("first",)
    assert event.calls[0].kwargs == {"parse_mode": "html"}
    assert event.edits == (event.calls[0],)
    assert event.replies == (event.calls[1],)
    assert event.responses == (event.calls[2],)
    assert event.output == "first\n\nsecond\n\nthird"
    assert (event.chat_id, event.sender_id, event.user_id) == (11, 22, 22)
    assert (event.msg_id, event.id) == (33, 33)
    assert (event.raw_text, event.text) == (".plugins", ".plugins")
    assert event.message.id == 33
    assert event.message.text == ".plugins"
    assert event.message is event
    assert event.out is False
    assert event.admin is False
    assert event.is_admin is False


@pytest.mark.asyncio
async def test_event_delegates_reply_chat_and_sender_access() -> None:
    reply = object()
    chat = object()
    sender = object()
    event = FakeCommandEvent(reply_message=reply, chat=chat, sender=sender)

    assert await event.get_reply_message() is reply
    assert await event.get_chat() is chat
    assert await event.get_sender() is sender
    assert [call.operation for call in event.calls] == [
        "get_reply_message",
        "get_chat",
        "get_sender",
    ]


@pytest.mark.asyncio
async def test_event_records_answer_alert_edit_and_delete_calls() -> None:
    event = FakeCommandEvent()

    answered = await event.answer("Not allowed", alert=True)
    await event.edit("Updated", buttons=(("OK", b"ok"),))
    deleted = await event.delete(revoke=True)

    assert answered is event.answer_result
    assert deleted is None
    assert event.answers == (event.calls[0],)
    assert event.deletes == (event.calls[2],)
    assert [(call.operation, call.args, dict(call.kwargs)) for call in event.calls] == [
        ("answer", ("Not allowed",), {"alert": True}),
        ("edit", ("Updated",), {"buttons": (("OK", b"ok"),)}),
        ("delete", (), {"revoke": True}),
    ]


def test_event_rejects_unknown_api() -> None:
    event = FakeCommandEvent()

    with pytest.raises(AttributeError):
        event.forward_to


def test_valid_action_returns_current_immutable_record(tmp_path: Path) -> None:
    registry, installed = _installed(tmp_path)
    clock = DeterministicClock()
    action = build_installed_plugin_action(
        installed,
        actor_id=42,
        clock=clock,
        token="valid-action",
        payload={"page": 2, "filters": ["active"]},
    )

    validated = validate_installed_plugin_action(
        action, registry, actor_id=42, now=clock()
    )

    assert validated is installed
    assert action.generation == installed.generation
    assert action.payload == {"page": 2, "filters": ("active",)}
    assert action.consumed is False


def test_action_payload_rejects_cycles_and_non_finite_numbers(tmp_path: Path) -> None:
    _, installed = _installed(tmp_path)
    clock = DeterministicClock()
    cyclic: list[object] = []
    cyclic.append(cyclic)

    with pytest.raises(TypeError, match="cycles"):
        build_installed_plugin_action(
            installed,
            actor_id=42,
            clock=clock,
            token="cyclic-action",
            payload={"value": cyclic},
        )
    with pytest.raises(TypeError, match="finite"):
        build_installed_plugin_action(
            installed,
            actor_id=42,
            clock=clock,
            token="non-finite-action",
            payload={"value": float("nan")},
        )


def test_expired_action_is_rejected_without_registry_mutation(tmp_path: Path) -> None:
    registry, installed = _installed(tmp_path)
    clock = DeterministicClock()
    action = build_installed_plugin_action(
        installed,
        actor_id=42,
        clock=clock,
        token="expired-action",
        ttl=timedelta(seconds=30),
    )
    clock.advance(seconds=30)

    _assert_action_error(
        FakeInstalledPluginActionErrorCode.EXPIRED, action, registry, clock
    )
    assert registry.get(installed.plugin_id) is installed


def test_cross_actor_action_is_rejected(tmp_path: Path) -> None:
    registry, installed = _installed(tmp_path)
    clock = DeterministicClock()
    action = build_installed_plugin_action(
        installed, actor_id=42, clock=clock, token="actor-action"
    )

    _assert_action_error(
        FakeInstalledPluginActionErrorCode.ACTOR_MISMATCH,
        action,
        registry,
        clock,
        actor_id=99,
    )


def test_replaced_generation_makes_action_stale(tmp_path: Path) -> None:
    registry, installed = _installed(tmp_path)
    clock = DeterministicClock()
    action = build_installed_plugin_action(
        installed, actor_id=42, clock=clock, token="stale-action"
    )
    replacement = _record(tmp_path, digest="b" * 64)
    replaced = registry.replace(
        replacement, expected_generation=installed.generation
    )

    _assert_action_error(
        FakeInstalledPluginActionErrorCode.STALE_GENERATION,
        action,
        registry,
        clock,
    )
    assert replaced.generation > action.generation


def test_consumed_action_cannot_be_replayed(tmp_path: Path) -> None:
    registry, installed = _installed(tmp_path)
    clock = DeterministicClock()
    action = build_installed_plugin_action(
        installed, actor_id=42, clock=clock, token="consumed-action"
    )
    store = FakeInstalledPluginActionStore()
    store.issue(action, registry)

    record, consumed = store.consume(
        action, registry, actor_id=42, now=clock()
    )

    assert record.generation == installed.generation
    assert action.token not in record.action_ids
    assert action.consumed is False
    assert consumed.consumed is True
    for replay in (action, consumed):
        with pytest.raises(FakeInstalledPluginActionError) as caught:
            store.validate(replay, registry, actor_id=42, now=clock())
        assert caught.value.code is FakeInstalledPluginActionErrorCode.CONSUMED


def test_action_store_rejects_cross_registry_substitution(tmp_path: Path) -> None:
    registry, installed = _installed(tmp_path)
    other_registry, _ = _installed(tmp_path)
    clock = DeterministicClock()
    action = build_installed_plugin_action(
        installed, actor_id=42, clock=clock, token="registry-action"
    )
    store = FakeInstalledPluginActionStore()
    store.issue(action, registry)

    with pytest.raises(FakeInstalledPluginActionError) as caught:
        store.validate(action, other_registry, actor_id=42, now=clock())
    assert caught.value.code is FakeInstalledPluginActionErrorCode.MISSING


@pytest.mark.parametrize(
    ("state", "expected_code"),
    [
        ("missing", FakeInstalledPluginActionErrorCode.MISSING),
        ("removed", FakeInstalledPluginActionErrorCode.MISSING),
        ("disabled", FakeInstalledPluginActionErrorCode.DISABLED),
        ("unloading", FakeInstalledPluginActionErrorCode.UNLOADING),
    ],
)
def test_unavailable_registry_record_is_rejected(
    tmp_path: Path,
    state: str,
    expected_code: FakeInstalledPluginActionErrorCode,
) -> None:
    registry, installed = _installed(tmp_path)
    clock = DeterministicClock()

    if state == "missing":
        registry = InstalledPluginRegistry()
        current = installed
    elif state == "removed":
        current = registry.remove(
            installed.plugin_id, expected_generation=installed.generation
        )
    elif state == "disabled":
        current = registry.set_enabled(
            installed.plugin_id,
            False,
            expected_generation=installed.generation,
        )
    else:
        active = registry.activate(
            installed.plugin_id, expected_generation=installed.generation
        )
        current = registry.begin_unload(
            active.plugin_id, expected_generation=active.generation
        )

    action = build_installed_plugin_action(
        current, actor_id=42, clock=clock, token=f"{state}-action"
    )
    _assert_action_error(expected_code, action, registry, clock)


@pytest.mark.asyncio
async def test_task_barrier_cancellation_reaps_task_without_sleep() -> None:
    harness = CommandTaskHarness()
    task = await harness.start()

    assert harness.started.is_set()
    assert not task.done()
    await harness.cancel_and_reap(task)

    assert task.cancelled()
    assert harness.reaped.is_set()

    second = await harness.start()
    assert not second.done()
    await harness.cancel_and_reap(second)
    assert second.cancelled()
