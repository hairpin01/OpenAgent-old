from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from OpenAgentLib.InstalledPluginRegistry import (
    InstalledPluginManifest,
    InstalledPluginRecord,
    InstalledPluginRegistry,
    InstalledPluginSource,
    InstalledPluginTool,
)
from OpenAgentLib.ResponseAgent import _OpenAgentResponseMixin


class _Buttons:
    @staticmethod
    def inline(text, callback, args=(), style=None):
        return {"text": text, "callback": callback, "args": args, "style": style}

    @staticmethod
    def input(text, callback, placeholder=None, data=None, **_kwargs):
        return {"text": text, "callback": callback, "data": data}


class _Event:
    def __init__(self, sender_id: int) -> None:
        self.sender_id = sender_id


class _Harness(_OpenAgentResponseMixin):
    def __init__(self, registry: InstalledPluginRegistry) -> None:
        self._installed_plugin_registry = registry
        self._regen_payloads = {}
        self._input_events = {}
        self.Button = _Buttons()
        self.log = SimpleNamespace(debug=lambda *_args, **_kwargs: None)

    @staticmethod
    def strings(key: str) -> str:
        return key

    @staticmethod
    def _open_sessions_panel(*_args):
        return None

    @staticmethod
    def _clear_context(*_args):
        return None

    @staticmethod
    def _on_follow_up_input(*_args):
        return None


def _record(tmp_path: Path, digest: str = "a" * 64) -> InstalledPluginRecord:
    return InstalledPluginRecord(
        plugin_id="demo.plugin",
        source=InstalledPluginSource((tmp_path / "plugin.py").resolve(), digest),
        manifest=InstalledPluginManifest(
            manifest_version="2",
            api_version="2",
            version="1",
            entrypoint="plugin.HANDLERS",
            capabilities=frozenset({"network"}),
            tools=(
                InstalledPluginTool(
                    canonical_id="demo.run", capabilities=frozenset({"network"})
                ),
            ),
        ),
    )


def _payload(harness: _Harness, *, actor_id: int = 7) -> tuple[str, dict]:
    harness._final_buttons(
        1, "p", "p", [], source_event=_Event(actor_id), agent_log=["demo.run"]
    )
    token, payload = next(iter(harness._regen_payloads.items()))
    return token, payload


def test_regen_tracks_only_executed_installed_owner_generation(tmp_path: Path) -> None:
    registry = InstalledPluginRegistry()
    active = registry.install_active(_record(tmp_path))
    harness = _Harness(registry)

    _token, payload = _payload(harness)

    assert payload["plugin_generations"] == ((active.plugin_id, active.generation),)
    assert harness._regen_plugin_generations(["native.tool", "answer.accepted"]) == ()


def test_regen_rejects_replaced_owner_but_not_unrelated_plugin(tmp_path: Path) -> None:
    registry = InstalledPluginRegistry()
    active = registry.install_active(_record(tmp_path))
    unrelated = registry.install_active(
        InstalledPluginRecord(
            plugin_id="other.plugin",
            source=InstalledPluginSource((tmp_path / "other.py").resolve(), "c" * 64),
            manifest=InstalledPluginManifest(
                manifest_version="2",
                api_version="2",
                version="1",
                entrypoint="other.HANDLERS",
                capabilities=frozenset({"network"}),
                tools=(
                    InstalledPluginTool(
                        canonical_id="other.run", capabilities=frozenset({"network"})
                    ),
                ),
            ),
        )
    )
    harness = _Harness(registry)
    token, _payload_value = _payload(harness)
    registry.replace_active(
        _record(tmp_path, "b" * 64), expected_generation=active.generation
    )
    assert harness._validate_regen_payload(token, _Event(7), consume=True) is None

    token, _payload_value = _payload(harness)
    registry.replace_active(
        InstalledPluginRecord(
            plugin_id="other.plugin",
            source=InstalledPluginSource((tmp_path / "other.py").resolve(), "d" * 64),
            manifest=unrelated.manifest,
        ),
        expected_generation=unrelated.generation,
    )
    assert harness._validate_regen_payload(token, _Event(7), consume=True) is not None


def test_regen_actor_expiry_and_replay_are_rejected(tmp_path: Path) -> None:
    registry = InstalledPluginRegistry()
    registry.install_active(_record(tmp_path))
    harness = _Harness(registry)
    token, payload = _payload(harness)
    assert harness._validate_regen_payload(token, _Event(8), consume=True) is None

    token, payload = _payload(harness)
    payload["expires_at"] = 0
    assert harness._validate_regen_payload(token, _Event(7), consume=True) is None

    token, _payload_value = _payload(harness)
    assert harness._validate_regen_payload(token, _Event(7), consume=True) is not None
    assert harness._validate_regen_payload(token, _Event(7), consume=True) is None
