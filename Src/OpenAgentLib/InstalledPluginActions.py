"""Opaque, generation-bound actions for installed-plugin callbacks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import RLock
from typing import TypeVar
from uuid import uuid4

from .InstalledPluginRegistry import (
    InstalledPluginMissingError,
    InstalledPluginRecord,
    InstalledPluginRegistry,
    InstalledPluginStaleGenerationError,
    InstalledPluginStatus,
    InstalledPluginTransitionError,
    _freeze_json,
)

_T = TypeVar("_T")


class InstalledPluginActionErrorCode(str, Enum):
    """Reasons an installed-plugin callback cannot be used."""

    MISSING = "missing"
    CONSUMED = "consumed"
    EXPIRED = "expired"
    ACTOR_MISMATCH = "actor-mismatch"
    KIND_MISMATCH = "kind-mismatch"
    STALE_GENERATION = "stale-generation"
    STATUS_MISMATCH = "status-mismatch"


class InstalledPluginActionError(ValueError):
    def __init__(self, code: InstalledPluginActionErrorCode) -> None:
        self.code = code
        super().__init__(code.value)


@dataclass(frozen=True, slots=True)
class InstalledPluginAction:
    """Immutable callback authorization with JSON-only frozen payload data."""

    token: str
    plugin_id: str
    generation: int
    actor_id: int | str
    expires_at: datetime
    kind: str
    payload: Mapping[str, object]
    statuses: frozenset[InstalledPluginStatus]


class InstalledPluginActionStore:
    """Lock-safe, one-time callback action store backed by registry ownership."""

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._actions: dict[str, InstalledPluginAction] = {}
        self._consumed: set[str] = set()
        self._lock = RLock()

    def issue(
        self,
        registry: InstalledPluginRegistry,
        record: InstalledPluginRecord,
        *,
        actor_id: int | str,
        kind: str,
        payload: Mapping[str, object] | None = None,
        ttl: timedelta = timedelta(minutes=5),
        statuses: frozenset[InstalledPluginStatus] | None = None,
    ) -> InstalledPluginAction:
        if not isinstance(kind, str) or not kind.strip():
            raise TypeError("action kind must be a non-empty string")
        if ttl <= timedelta():
            raise ValueError("action TTL must be positive")
        now = self._now()
        action = InstalledPluginAction(
            token=uuid4().hex,
            plugin_id=record.plugin_id,
            generation=record.generation,
            actor_id=actor_id,
            expires_at=now + ttl,
            kind=kind.strip(),
            payload=_freeze_payload(payload or {}),
            statuses=statuses or frozenset({record.status}),
        )
        with self._lock:
            registry.bind_action(
                action.plugin_id,
                action.token,
                expected_generation=action.generation,
            )
            self._actions[action.token] = action
        return action

    def consume(
        self,
        registry: InstalledPluginRegistry,
        token: str,
        *,
        actor_id: int | str,
        kind: str | None = None,
    ) -> tuple[InstalledPluginAction, InstalledPluginRecord]:
        """Validate and atomically consume an action before caller-side mutation."""

        with self._lock:
            action = self._actions.get(token)
            if action is None:
                code = (
                    InstalledPluginActionErrorCode.CONSUMED
                    if token in self._consumed
                    else InstalledPluginActionErrorCode.MISSING
                )
                raise InstalledPluginActionError(code)
            if action.actor_id != actor_id:
                raise InstalledPluginActionError(
                    InstalledPluginActionErrorCode.ACTOR_MISMATCH
                )
            if kind is not None and action.kind != kind:
                raise InstalledPluginActionError(
                    InstalledPluginActionErrorCode.KIND_MISMATCH
                )
            if self._now() >= action.expires_at:
                raise InstalledPluginActionError(InstalledPluginActionErrorCode.EXPIRED)
            try:
                owner = registry.get_action_owner(
                    action.token, expected_generation=action.generation
                )
                record = registry.get(action.plugin_id)
            except (
                InstalledPluginMissingError,
                InstalledPluginStaleGenerationError,
            ) as exc:
                raise InstalledPluginActionError(
                    InstalledPluginActionErrorCode.STALE_GENERATION
                ) from exc
            if (
                owner.plugin_id != action.plugin_id
                or owner.generation != action.generation
            ):
                raise InstalledPluginActionError(
                    InstalledPluginActionErrorCode.STALE_GENERATION
                )
            if record.generation != action.generation:
                raise InstalledPluginActionError(
                    InstalledPluginActionErrorCode.STALE_GENERATION
                )
            if record.status not in action.statuses:
                raise InstalledPluginActionError(
                    InstalledPluginActionErrorCode.STATUS_MISMATCH
                )
            try:
                record = registry.release_action(
                    action.plugin_id,
                    action.token,
                    expected_generation=action.generation,
                )
            except (
                InstalledPluginMissingError,
                InstalledPluginStaleGenerationError,
                InstalledPluginTransitionError,
            ) as exc:
                raise InstalledPluginActionError(
                    InstalledPluginActionErrorCode.STALE_GENERATION
                ) from exc
            del self._actions[action.token]
            self._consumed.add(action.token)
            return action, record

    def revoke_generation(
        self,
        registry: InstalledPluginRegistry,
        plugin_id: str,
        *,
        expected_generation: int,
        transition: Callable[[], _T],
    ) -> _T:
        """Revoke one generation's callbacks before its synchronous transition."""

        with self._lock:
            actions = tuple(
                action
                for action in self._actions.values()
                if (
                    action.plugin_id == plugin_id
                    and action.generation == expected_generation
                )
            )
            registry.release_actions(
                plugin_id,
                (action.token for action in actions),
                expected_generation=expected_generation,
            )
            for action in actions:
                del self._actions[action.token]
                self._consumed.add(action.token)
            return transition()

    def revoke_all(self, registry: InstalledPluginRegistry) -> None:
        """Best-effort release of actions during lifecycle teardown."""

        with self._lock:
            actions = tuple(self._actions.values())
            self._actions.clear()
            self._consumed.clear()
            for action in actions:
                try:
                    registry.release_action(
                        action.plugin_id,
                        action.token,
                        expected_generation=action.generation,
                    )
                except (
                    InstalledPluginMissingError,
                    InstalledPluginStaleGenerationError,
                ):
                    pass

    def tracks(self, token: str) -> bool:
        """Return whether a token belongs to this store, including replays."""

        with self._lock:
            return token in self._actions or token in self._consumed

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("action clock must return a timezone-aware datetime")
        return now.astimezone(timezone.utc)


def _freeze_payload(payload: Mapping[str, object]) -> Mapping[str, object]:
    frozen = _freeze_json(payload)
    if not isinstance(frozen, Mapping):
        raise TypeError("action payload must be a JSON object")
    return frozen
