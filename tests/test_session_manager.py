from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path
from types import ModuleType

from conftest import load_source_module

package_name = "openagent_manager_test"
package = ModuleType(package_name)
package.__path__ = []  # type: ignore[attr-defined]
sys.modules[package_name] = package
load_source_module(
    f"{package_name}.OASession",
    "Src/OpenAgentLib/Manager/OASession.py",
)
session_module = load_source_module(
    f"{package_name}.Session",
    "Src/OpenAgentLib/Manager/Session.py",
)


class _Logger:
    def warning(self, _message: str) -> None:
        pass


def test_close_does_not_reschedule_debounced_save(tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = session_module.SessionManager(
            tmp_path / "sessions.json",
            logger=_Logger(),
            model_getter=lambda: "test-model",
            default_name_getter=lambda: "new",
            session_limit=5,
        )
        manager._save_debounce_seconds = 60
        manager.schedule_save()
        assert manager._save_task is not None

        await manager.close()
        await asyncio.sleep(0)

        assert manager._closing
        assert manager._save_task is None
        assert manager._saved_generation == manager._save_generation
        assert (tmp_path / "sessions.json").is_file()

    asyncio.run(scenario())


def test_close_waits_for_inflight_thread_before_latest_save(tmp_path: Path) -> None:
    async def scenario() -> None:
        path = tmp_path / "sessions.json"
        manager = session_module.SessionManager(
            path,
            logger=_Logger(),
            model_getter=lambda: "test-model",
            default_name_getter=lambda: "new",
            session_limit=5,
        )
        manager._save_debounce_seconds = 0
        started = threading.Event()
        release = threading.Event()
        original_save = manager._save_payload_sync
        calls = 0

        def blocking_save(payload: dict) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                release.wait(timeout=5)
            original_save(payload)

        manager._save_payload_sync = blocking_save
        manager.session_prefs[1] = {"state": "old"}
        manager.schedule_save()
        await asyncio.to_thread(started.wait, 5)
        manager.session_prefs[1] = {"state": "latest"}
        manager.schedule_save()

        close_task = asyncio.create_task(manager.close())
        await asyncio.sleep(0)
        assert not close_task.done()
        release.set()
        await close_task

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["prefs"]["1"]["state"] == "latest"
        assert calls >= 2

    asyncio.run(scenario())
