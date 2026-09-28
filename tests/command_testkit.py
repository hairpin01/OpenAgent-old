from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping

from OpenAgentLib.InstalledPluginRegistry import (
    InstalledPluginMissingError,
    InstalledPluginRecord,
    InstalledPluginRegistry,
    InstalledPluginStatus,
)

FIXED_NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


class DeterministicClock:
    def __init__(self, now: datetime = FIXED_NOW) -> None:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock time must be timezone-aware")
        self._now = now

    def __call__(self) -> datetime:
        return self._now

    def advance(self, **delta: float) -> datetime:
        self._now += timedelta(**delta)
        return self._now


@dataclass(frozen=True)
class FakeCommandChat:
    id: int


@dataclass(frozen=True)
class FakeCommandSender:
    id: int


@dataclass(frozen=True)
class FakeCommandResult:
    operation: str
    id: int


@dataclass(frozen=True)
class FakeCommandCall:
    operation: str
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]


class FakeCommandEvent:
    def __init__(
        self,
        text: str = "",
        *,
        chat_id: int = 100,
        sender_id: int = 200,
        msg_id: int = 300,
        out: bool = True,
        admin: bool = True,
        reply_message: Any = None,
        chat: Any = None,
        sender: Any = None,
    ) -> None:
        self.chat_id = chat_id
        self.sender_id = sender_id
        self.user_id = sender_id
        self.msg_id = msg_id
        self.id = msg_id
        self.raw_text = text
        self.text = text
        self.out = out
        self.admin = admin
        self.is_admin = admin
        self.message = self
        self.chat = chat if chat is not None else FakeCommandChat(chat_id)
        self.sender = sender if sender is not None else FakeCommandSender(sender_id)
        self.reply_message = reply_message
        self._calls: list[FakeCommandCall] = []
        self.answer_result = FakeCommandResult("answer", msg_id)

    def _record(
        self, operation: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        self._calls.append(
            FakeCommandCall(
                operation,
                tuple(deepcopy(args)),
                MappingProxyType(deepcopy(kwargs)),
            )
        )

    @property
    def calls(self) -> tuple[FakeCommandCall, ...]:
        return tuple(self._calls)

    async def edit(self, text: Any, *args: Any, **kwargs: Any) -> FakeCommandEvent:
        self._record("edit", (text, *args), kwargs)
        return self

    async def reply(self, text: Any, *args: Any, **kwargs: Any) -> FakeCommandEvent:
        self._record("reply", (text, *args), kwargs)
        return self

    async def respond(self, text: Any, *args: Any, **kwargs: Any) -> FakeCommandEvent:
        self._record("respond", (text, *args), kwargs)
        return self

    async def answer(self, *args: Any, **kwargs: Any) -> FakeCommandResult:
        self._record("answer", args, kwargs)
        return self.answer_result

    async def delete(self, *args: Any, **kwargs: Any) -> None:
        self._record("delete", args, kwargs)
        return None

    async def get_reply_message(self) -> Any:
        self._record("get_reply_message", (), {})
        return self.reply_message

    async def get_chat(self) -> Any:
        self._record("get_chat", (), {})
        return self.chat

    async def get_sender(self) -> Any:
        self._record("get_sender", (), {})
        return self.sender

    def _calls_for(self, operation: str) -> tuple[FakeCommandCall, ...]:
        return tuple(call for call in self.calls if call.operation == operation)

    @property
    def edits(self) -> tuple[FakeCommandCall, ...]:
        return self._calls_for("edit")

    @property
    def replies(self) -> tuple[FakeCommandCall, ...]:
        return self._calls_for("reply")

    @property
    def responses(self) -> tuple[FakeCommandCall, ...]:
        return self._calls_for("respond")

    @property
    def answers(self) -> tuple[FakeCommandCall, ...]:
        return self._calls_for("answer")

    @property
    def deletes(self) -> tuple[FakeCommandCall, ...]:
        return self._calls_for("delete")

    @property
    def output(self) -> str:
        output_operations = {"edit", "reply", "respond"}
        return "\n\n".join(
            str(call.args[0])
            for call in self.calls
            if call.operation in output_operations and call.args
        ).strip()


class FakeInstalledPluginActionErrorCode(str, Enum):
    MISSING = "missing"
    DISABLED = "disabled"
    UNLOADING = "unloading"
    UNAVAILABLE = "unavailable"
    STALE_GENERATION = "stale_generation"
    ACTOR_MISMATCH = "actor_mismatch"
    EXPIRED = "expired"
    CONSUMED = "consumed"


class FakeInstalledPluginActionError(ValueError):
    def __init__(
        self,
        code: FakeInstalledPluginActionErrorCode,
        action: FakeInstalledPluginAction,
    ) -> None:
        self.code = code
        self.action = action
        super().__init__(f"{code.value}: action {action.token!r} is not valid")


def _freeze_json(value: Any, active: set[int] | None = None) -> Any:
    active = set() if active is None else active
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not isfinite(value):
            raise TypeError("action payload numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("action payload keys must be strings")
        identity = id(value)
        if identity in active:
            raise TypeError("action payload cannot contain cycles")
        active.add(identity)
        try:
            return MappingProxyType(
                {key: _freeze_json(item, active) for key, item in value.items()}
            )
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise TypeError("action payload cannot contain cycles")
        active.add(identity)
        try:
            return tuple(_freeze_json(item, active) for item in value)
        finally:
            active.remove(identity)
    raise TypeError("action payload must contain only JSON-safe values")


@dataclass(frozen=True)
class FakeInstalledPluginAction:
    token: str
    plugin_id: str
    generation: int
    actor_id: int
    expires_at: datetime
    payload: Mapping[str, Any] | None = None
    consumed: bool = False

    def __post_init__(self) -> None:
        if not self.token:
            raise ValueError("action token cannot be empty")
        if isinstance(self.generation, bool) or not isinstance(self.generation, int):
            raise TypeError("action generation must be an integer")
        if isinstance(self.actor_id, bool) or not isinstance(self.actor_id, int):
            raise TypeError("action actor ID must be an integer")
        if self.expires_at.tzinfo is None or self.expires_at.utcoffset() is None:
            raise ValueError("action expiry must be timezone-aware")
        if self.payload is not None:
            object.__setattr__(self, "payload", _freeze_json(self.payload))


def build_installed_plugin_action(
    record: InstalledPluginRecord,
    *,
    actor_id: int,
    clock: DeterministicClock,
    token: str,
    ttl: timedelta = timedelta(minutes=5),
    payload: Mapping[str, Any] | None = None,
) -> FakeInstalledPluginAction:
    return FakeInstalledPluginAction(
        token=token,
        plugin_id=record.plugin_id,
        generation=record.generation,
        actor_id=actor_id,
        expires_at=clock() + ttl,
        payload=payload,
    )


def validate_installed_plugin_action(
    action: FakeInstalledPluginAction,
    registry: InstalledPluginRegistry,
    *,
    actor_id: int,
    now: datetime,
) -> InstalledPluginRecord:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("validation time must be timezone-aware")
    try:
        record = registry.get(action.plugin_id)
    except InstalledPluginMissingError as exc:
        raise FakeInstalledPluginActionError(
            FakeInstalledPluginActionErrorCode.MISSING, action
        ) from exc

    status_errors = {
        InstalledPluginStatus.DISABLED: FakeInstalledPluginActionErrorCode.DISABLED,
        InstalledPluginStatus.UNLOADING: FakeInstalledPluginActionErrorCode.UNLOADING,
    }
    if record.status in status_errors:
        raise FakeInstalledPluginActionError(status_errors[record.status], action)
    if record.status not in {
        InstalledPluginStatus.INSTALLED,
        InstalledPluginStatus.ACTIVE,
    }:
        raise FakeInstalledPluginActionError(
            FakeInstalledPluginActionErrorCode.UNAVAILABLE, action
        )
    if record.generation != action.generation:
        raise FakeInstalledPluginActionError(
            FakeInstalledPluginActionErrorCode.STALE_GENERATION, action
        )
    if (
        isinstance(actor_id, bool)
        or not isinstance(actor_id, int)
        or actor_id != action.actor_id
    ):
        raise FakeInstalledPluginActionError(
            FakeInstalledPluginActionErrorCode.ACTOR_MISMATCH, action
        )
    if now >= action.expires_at:
        raise FakeInstalledPluginActionError(
            FakeInstalledPluginActionErrorCode.EXPIRED, action
        )
    if action.consumed:
        raise FakeInstalledPluginActionError(
            FakeInstalledPluginActionErrorCode.CONSUMED, action
        )
    return record


class FakeInstalledPluginActionStore:
    def __init__(self) -> None:
        self._actions: dict[str, FakeInstalledPluginAction] = {}
        self._registries: dict[str, InstalledPluginRegistry] = {}

    def issue(
        self,
        action: FakeInstalledPluginAction,
        registry: InstalledPluginRegistry,
    ) -> FakeInstalledPluginAction:
        if action.token in self._actions:
            raise ValueError(f"action token {action.token!r} is already issued")
        registry.bind_action(
            action.plugin_id,
            action.token,
            expected_generation=action.generation,
        )
        self._actions[action.token] = action
        self._registries[action.token] = registry
        return action

    def validate(
        self,
        action: FakeInstalledPluginAction,
        registry: InstalledPluginRegistry,
        *,
        actor_id: int,
        now: datetime,
    ) -> InstalledPluginRecord:
        current = self._actions.get(action.token)
        if current is None:
            raise FakeInstalledPluginActionError(
                FakeInstalledPluginActionErrorCode.MISSING, action
            )
        if self._registries[action.token] is not registry:
            raise FakeInstalledPluginActionError(
                FakeInstalledPluginActionErrorCode.MISSING, action
            )
        if current.consumed:
            raise FakeInstalledPluginActionError(
                FakeInstalledPluginActionErrorCode.CONSUMED, action
            )
        if current != action:
            raise FakeInstalledPluginActionError(
                FakeInstalledPluginActionErrorCode.MISSING, action
            )
        try:
            owner = registry.get_action_owner(
                action.token, expected_generation=action.generation
            )
        except InstalledPluginMissingError as exc:
            raise FakeInstalledPluginActionError(
                FakeInstalledPluginActionErrorCode.MISSING, action
            ) from exc
        if owner.plugin_id != action.plugin_id or owner.generation != action.generation:
            raise FakeInstalledPluginActionError(
                FakeInstalledPluginActionErrorCode.STALE_GENERATION, action
            )
        return validate_installed_plugin_action(
            current, registry, actor_id=actor_id, now=now
        )

    def consume(
        self,
        action: FakeInstalledPluginAction,
        registry: InstalledPluginRegistry,
        *,
        actor_id: int,
        now: datetime,
    ) -> tuple[InstalledPluginRecord, FakeInstalledPluginAction]:
        self.validate(action, registry, actor_id=actor_id, now=now)
        record = registry.release_action(
            action.plugin_id,
            action.token,
            expected_generation=action.generation,
        )
        consumed = replace(action, consumed=True)
        self._actions[action.token] = consumed
        return record, consumed


class CommandTaskHarness:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.reaped = asyncio.Event()

    async def _blocked_task(self) -> None:
        self.started.set()
        try:
            await self.release.wait()
        finally:
            self.reaped.set()

    async def start(self) -> asyncio.Task[None]:
        self.started.clear()
        self.release.clear()
        self.reaped.clear()
        task = asyncio.create_task(self._blocked_task())
        await self.started.wait()
        return task

    async def cancel_and_reap(self, task: asyncio.Task[None]) -> None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancelled command task unexpectedly completed")
        await self.reaped.wait()
