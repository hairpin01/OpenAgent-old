from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from OpenAgentLib.InstalledPluginRegistry import (
    InstalledPluginMissingError,
    InstalledPluginRegistry,
    InstalledPluginStatus,
    InstalledPluginTransitionError,
)
from OpenAgentLib.IsolatedPluginInvoker import IsolatedPluginInvoker
from OpenAgentLib.Plugin.PluginsEngine import _OpenAgentPluginSkillMixin
from OpenAgentLib.PluginHost import PluginHost

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "repo-MCUB-fork" / "OpenAgent" / "plugins"


class _Harness(_OpenAgentPluginSkillMixin):
    def __init__(self, tmp_path: Path) -> None:
        self.kernel = SimpleNamespace(WORK_DIR=str(tmp_path))
        self.startup_disabled_ids: set[str] = set()
        self._installed_plugin_registry = InstalledPluginRegistry()
        self._plugins_cache = object()
        self._tool_map_cache = object()
        self._tool_registry_cache = object()
        self._args_raw = lambda _event: ""
        self._v2_plugin_invoker = IsolatedPluginInvoker(
            PluginHost(),
            {},
            plugin_root=tmp_path,
            openagent_source=ROOT / "OpenAgent-old" / "Src",
            capability_handler=lambda *_args: {"ok": False},
            registry=self._installed_plugin_registry,
        )

    @property
    def _v2_plugin_sources(self) -> dict[str, object]:
        invoker = self._v2_plugin_invoker
        return invoker._sources if invoker is not None else {}

    @property
    def _plugin_files(self) -> dict[str, Path]:
        return {
            Path(record.source.path).stem: Path(record.source.path)
            for record in self._installed_plugin_registry.snapshot()
        }

    @property
    def _disabled_plugins(self) -> set[str]:
        return self.startup_disabled_ids


class _RecordingHost:
    def __init__(self) -> None:
        self.requests: list[object] = []

    async def call(self, request: object, **_kwargs: object) -> object:
        self.requests.append(request)
        return SimpleNamespace()


def _chat_files(*, marker: bytes = b"") -> dict[str, bytes]:
    files = {
        name: (CANONICAL / name).read_bytes()
        for name in ("chat.py", "_telegram_v2.py", "__init__.py")
    }
    files["chat.py"] += marker
    return files


def _terminal_files() -> dict[str, bytes]:
    return {
        name: (CANONICAL / name).read_bytes()
        for name in ("terminal.py", "_resource_v2.py", "__init__.py")
    }


def _install(harness: _Harness, *, marker: bytes = b""):
    record = asyncio.run(
        harness._install_v2_plugin_files(_chat_files(marker=marker), "chat.py")
    )
    if record.plugin_id in harness.startup_disabled_ids and record.enabled:
        return asyncio.run(
            harness._set_installed_plugin_enabled(
                record.plugin_id, expected_generation=record.generation, enabled=False
            )
        )
    return record


def test_new_install_publishes_active_record_and_matching_live_bindings(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path)

    record = _install(harness)

    target = (tmp_path / "openagent_plugins" / "chat.py").resolve()
    assert record.status is InstalledPluginStatus.ACTIVE
    assert harness._installed_plugin_registry.get("openagent.chat") is record
    assert harness._v2_plugin_sources["chat"].path == target
    assert harness._plugin_files["chat"] == target
    assert (
        harness._v2_plugin_invoker._sources["chat"]
        is harness._v2_plugin_sources["chat"]
    )
    assert harness._v2_plugin_invoker._records["chat"] is record


def test_active_replace_consumes_one_generation_and_retires_actions(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path)
    host = _RecordingHost()
    harness._v2_plugin_invoker._host = host
    installed = _install(harness)
    action_owner = harness._installed_plugin_registry.bind_action(
        installed.plugin_id, "replace-action", expected_generation=installed.generation
    )

    replacement = _install(harness, marker=b"\n# replacement\n")

    assert replacement.status is InstalledPluginStatus.ACTIVE
    assert replacement.generation == installed.generation + 1
    assert replacement.action_ids == frozenset()
    assert replacement.source.digest != action_owner.source.digest
    with pytest.raises(InstalledPluginMissingError):
        harness._installed_plugin_registry.get_action_owner(
            "replace-action", expected_generation=installed.generation
        )
    assert harness._v2_plugin_invoker._records["chat"] is replacement
    asyncio.run(
        harness._v2_plugin_invoker.invoke(
            SimpleNamespace(
                spec=SimpleNamespace(source_module="chat"),
                call_id="replacement-call",
                canonical_id=replacement.manifest.tools[0].canonical_id,
                arguments={},
                context=None,
            ),
            SimpleNamespace(),
            retryable=False,
        )
    )
    assert host.requests[0].payload["source_sha256"] == replacement.source.digest


def test_disabled_update_stays_disabled_without_live_source(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    harness._disabled_plugins.add("openagent.chat")

    installed = _install(harness)
    replacement = _install(harness, marker=b"\n# disabled replacement\n")

    assert installed.status is InstalledPluginStatus.DISABLED
    assert replacement.status is InstalledPluginStatus.DISABLED
    assert replacement.generation == installed.generation + 1
    assert "chat" not in harness._v2_plugin_sources
    assert "chat" not in harness._v2_plugin_invoker._sources
    assert (
        harness._plugin_files["chat"]
        == (tmp_path / "openagent_plugins" / "chat.py").resolve()
    )


def test_real_invoker_invokes_relocated_terminal_after_disable_and_reenable(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cubkit-cache"
    cache_root.mkdir()
    harness = _Harness(cache_root)
    host = _RecordingHost()
    harness._v2_plugin_invoker._host = host
    files = _terminal_files()
    installed = asyncio.run(
        harness._install_v2_plugin_files(files, "terminal.py")
    )
    disabled = asyncio.run(
        harness._set_installed_plugin_enabled(
            installed.plugin_id, expected_generation=installed.generation, enabled=False
        )
    )

    installed_root = tmp_path / "installed"
    installed_plugins = installed_root / "openagent_plugins"
    installed_plugins.mkdir(parents=True)
    for name, content in files.items():
        (installed_plugins / name).write_bytes(content)
    harness.kernel.WORK_DIR = str(installed_root)
    harness._v2_plugin_invoker._plugin_root = installed_root.resolve()

    active = asyncio.run(
        harness._set_installed_plugin_enabled(
            disabled.plugin_id, expected_generation=disabled.generation, enabled=True
        )
    )
    call = SimpleNamespace(
        spec=SimpleNamespace(source_module="terminal"),
        call_id="terminal-reenabled",
        canonical_id="terminal.run",
        arguments={"argv": ["pwd"], "cwd": "."},
        context=None,
    )
    asyncio.run(
        harness._v2_plugin_invoker.invoke(call, SimpleNamespace(), retryable=False)
    )

    assert active.status is InstalledPluginStatus.ACTIVE
    assert harness._v2_plugin_invoker._sources["terminal"].path == (
        installed_plugins / "terminal.py"
    ).resolve()
    assert host.requests[0].payload["module"] == "openagent_plugins.terminal"
    assert host.requests[0].payload["source_sha256"] == active.source.digest


def test_enabled_install_without_invoker_rolls_back_before_publication(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path)
    harness._v2_plugin_invoker = None

    with pytest.raises(ValueError, match="requires an isolated invoker"):
        _install(harness)

    plugins = tmp_path / "openagent_plugins"
    assert not (plugins / "chat.py").exists()
    assert not (plugins / "_telegram_v2.py").exists()
    assert not (plugins / "__init__.py").exists()
    assert harness._installed_plugin_registry.snapshot() == ()
    assert harness._v2_plugin_sources == {}
    assert harness._plugin_files == {}


def test_enabled_update_without_invoker_restores_active_generation_exactly(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path)
    installed = _install(harness)
    target = tmp_path / "openagent_plugins" / "chat.py"
    snapshot = (target.read_bytes(), target.stat().st_mode)
    harness._v2_plugin_invoker = None

    with pytest.raises(ValueError, match="requires an isolated invoker"):
        _install(harness, marker=b"\n# missing invoker\n")

    assert target.read_bytes() == snapshot[0]
    assert target.stat().st_mode == snapshot[1]
    assert harness._installed_plugin_registry.get(installed.plugin_id) is installed
    assert harness._plugin_files["chat"] == target.resolve()


def test_disabled_install_succeeds_without_invoker(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    installed = _install(harness)
    record = asyncio.run(
        harness._set_installed_plugin_enabled(
            installed.plugin_id, expected_generation=installed.generation, enabled=False
        )
    )
    harness._v2_plugin_invoker = None

    assert record.status is InstalledPluginStatus.DISABLED
    assert harness._installed_plugin_registry.get(record.plugin_id) is record
    assert harness._v2_plugin_sources == {}
    assert (
        harness._plugin_files["chat"]
        == (tmp_path / "openagent_plugins" / "chat.py").resolve()
    )


def test_static_admission_and_file_write_failures_leave_prior_state_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(tmp_path)
    installed = _install(harness)
    target = tmp_path / "openagent_plugins" / "chat.py"
    snapshot = (target.read_bytes(), target.stat().st_mode)
    source = harness._v2_plugin_sources["chat"]

    with pytest.raises(Exception):
        asyncio.run(
            harness._install_v2_plugin_files(
                {"chat.py": b"not valid Python ("}, "chat.py"
            )
        )

    def reject_write(*_args: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(harness, "_write_plugin_files", reject_write)
    with pytest.raises(OSError, match="disk full"):
        _install(harness, marker=b"\n# never written\n")

    assert target.read_bytes() == snapshot[0]
    assert target.stat().st_mode == snapshot[1]
    assert harness._installed_plugin_registry.get(installed.plugin_id) is installed
    assert harness._v2_plugin_sources["chat"] is source
    assert harness._v2_plugin_invoker._sources["chat"] is source


def test_invoker_validation_and_identity_collision_roll_back_files_and_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(tmp_path)
    installed = _install(harness)
    target = tmp_path / "openagent_plugins" / "chat.py"
    snapshot = (target.read_bytes(), target.stat().st_mode)
    source = harness._v2_plugin_sources["chat"]

    def reject_source(*_args: object, **_kwargs: object) -> tuple[str, str]:
        raise ValueError("invalid live source")

    monkeypatch.setattr(harness._v2_plugin_invoker, "validate_source", reject_source)
    with pytest.raises(ValueError, match="invalid live source"):
        _install(harness, marker=b"\n# invalid invoker\n")

    assert target.read_bytes() == snapshot[0]
    assert target.stat().st_mode == snapshot[1]
    assert harness._installed_plugin_registry.get(installed.plugin_id) is installed
    assert harness._v2_plugin_sources["chat"] is source
    assert harness._v2_plugin_invoker._sources["chat"] is source

    monkeypatch.undo()
    collision = _chat_files()
    collision["other.py"] = collision.pop("chat.py")
    with pytest.raises(ValueError, match="another source"):
        asyncio.run(harness._install_v2_plugin_files(collision, "other.py"))

    assert not (tmp_path / "openagent_plugins" / "other.py").exists()
    assert harness._installed_plugin_registry.get(installed.plugin_id) is installed
    assert harness._v2_plugin_sources["chat"] is source


def test_repeated_concurrent_install_has_one_published_generation(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path)

    async def install_twice():
        return await asyncio.gather(
            harness._install_v2_plugin_files(_chat_files(), "chat.py"),
            harness._install_v2_plugin_files(_chat_files(), "chat.py"),
        )

    first, second = asyncio.run(install_twice())

    assert first is second
    assert first.status is InstalledPluginStatus.ACTIVE
    assert harness._installed_plugin_registry.next_generation == first.generation + 1


def test_active_task_blocks_replace_and_delete_until_released(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    installed = _install(harness)
    target = tmp_path / "openagent_plugins" / "chat.py"
    snapshot = (target.read_bytes(), target.stat().st_mode)
    owned = harness._installed_plugin_registry.own_task(
        installed.plugin_id, "active-task", expected_generation=installed.generation
    )

    with pytest.raises(InstalledPluginTransitionError):
        _install(harness, marker=b"\n# blocked replacement\n")
    with pytest.raises(InstalledPluginTransitionError):
        harness._unregister_plugin("chat")

    assert target.read_bytes() == snapshot[0]
    assert target.stat().st_mode == snapshot[1]
    assert harness._installed_plugin_registry.get(installed.plugin_id) is owned

    harness._installed_plugin_registry.release_task(
        owned.plugin_id, "active-task", expected_generation=owned.generation
    )
    harness._unregister_plugin("openagent.chat")

    assert not target.exists()
    with pytest.raises(InstalledPluginMissingError):
        harness._installed_plugin_registry.get("openagent.chat")


def test_delete_file_failure_and_shared_helpers_preserve_expected_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(tmp_path)
    installed = _install(harness)
    plugins = tmp_path / "openagent_plugins"
    target = plugins / "chat.py"
    original_unlink = Path.unlink

    def reject_target(path: Path, *args: object, **kwargs: object) -> None:
        if path == target:
            raise OSError("permission denied")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", reject_target)
    with pytest.raises(OSError, match="permission denied"):
        harness._unregister_plugin("chat")
    assert target.exists()
    assert harness._installed_plugin_registry.get(installed.plugin_id) is installed

    monkeypatch.undo()
    harness._unregister_plugin("chat")

    assert not target.exists()
    assert (plugins / "_telegram_v2.py").exists()
    assert (plugins / "__init__.py").exists()
    assert "chat" not in harness._v2_plugin_sources
    assert "chat" not in harness._plugin_files
    assert "chat" not in harness._v2_plugin_invoker._sources


def test_code_install_uses_the_same_registry_transaction(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    plugins = tmp_path / "openagent_plugins"
    plugins.mkdir()
    for filename in ("_telegram_v2.py", "__init__.py"):
        (plugins / filename).write_bytes((CANONICAL / filename).read_bytes())

    result = asyncio.run(
        harness._install_plugin_from_code("chat", (CANONICAL / "chat.py").read_text())
    )

    assert result == "chat"
    assert (
        harness._installed_plugin_registry.get("openagent.chat").status
        is InstalledPluginStatus.ACTIVE
    )
    assert "chat" in harness._v2_plugin_sources

    class _Reply:
        file = SimpleNamespace(name="chat.py")

        async def download_media(self, *, file: object) -> bytes:
            assert file is bytes
            return (CANONICAL / "chat.py").read_bytes()

    class _Event:
        async def get_reply_message(self) -> _Reply:
            return _Reply()

    assert asyncio.run(harness._install_plugin_from_reply(_Event())) == "chat"
