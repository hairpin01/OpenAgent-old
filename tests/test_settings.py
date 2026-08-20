from __future__ import annotations

import json

from conftest import load_source_module

settings = load_source_module(
    "openagent_settings_test",
    "Src/Settings.py",
)


class _Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, value: str) -> None:
        self.lines.append(value)


def test_debug_is_disabled_in_source_builds() -> None:
    assert settings.DEBUG is False


def test_debug_log_redacts_credentials_and_preserves_trace(monkeypatch) -> None:
    logger = _Logger()
    monkeypatch.setattr(settings, "DEBUG", True)

    settings.debug_log(
        logger,
        "provider.request",
        api_key="super-secret",
        headers={"Authorization": "Bearer secret", "Content-Type": "application/json"},
        messages=[{"role": "user", "content": "inspect project"}],
    )

    assert len(logger.lines) == 1
    payload = json.loads(logger.lines[0].removeprefix("[OpenAgent DEBUG] "))
    assert payload["api_key"] == "<redacted>"
    assert payload["headers"]["Authorization"] == "<redacted>"
    content = payload["messages"][0]["content"]
    assert content["length"] == len("inspect project")
    assert content["preview"] == "inspect project"
    assert len(content["sha256"]) == 64
    assert "super-secret" not in logger.lines[0]
    assert "Bearer secret" not in logger.lines[0]


def test_debug_log_redacts_secrets_embedded_in_neutral_strings(monkeypatch) -> None:
    logger = _Logger()
    monkeypatch.setattr(settings, "DEBUG", True)

    settings.debug_log(
        logger,
        "tool.result",
        result=(
            "Authorization: Bearer TOPSECRET123456789 "
            'payload={"api_key":"TOPSECRET2"} '
            "token=123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcd"
        ),
    )

    line = logger.lines[0]
    assert "TOPSECRET123456789" not in line
    assert "TOPSECRET2" not in line
    assert "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcd" not in line
    assert "<redacted>" in line


def test_debug_log_bounds_strings_collections_and_total_event(monkeypatch) -> None:
    logger = _Logger()
    monkeypatch.setattr(settings, "DEBUG", True)

    settings.debug_log(
        logger,
        "large.event",
        values=["x" * 20_000 for _ in range(200)],
    )

    assert len(logger.lines[0]) <= settings.DEBUG_MAX_EVENT_CHARS + 64
    assert "truncated" in logger.lines[0]


def test_configure_debug_changes_runtime_only() -> None:
    original = settings.DEBUG
    try:
        settings.configure_debug(True)
        assert settings.DEBUG is True
        settings.configure_debug(False)
        assert settings.DEBUG is False
    finally:
        settings.configure_debug(original)


def test_debug_profile_is_derived_from_artifact_name() -> None:
    assert settings.debug_for_artifact("/tmp/OpenDebug.py")
    assert settings.debug_for_artifact("custom-debug-build.py")
    assert not settings.debug_for_artifact("/tmp/OpenAgent-MCUB-repo.py")
    assert not settings.debug_for_artifact("OpenAgentMain.py")


def test_debug_logging_never_breaks_agent(monkeypatch) -> None:
    class RaisingLogger:
        def info(self, _value: str) -> None:
            raise RuntimeError("logger unavailable")

    class RaisingRepr:
        def __repr__(self) -> str:
            raise RuntimeError("bad repr")

    monkeypatch.setattr(settings, "DEBUG", True)
    settings.debug_log(RaisingLogger(), "broken.logger", value=RaisingRepr())


def test_disabled_debug_does_not_touch_values(monkeypatch) -> None:
    class RaisingRepr:
        def __repr__(self) -> str:
            raise AssertionError("disabled debug evaluated payload")

    monkeypatch.setattr(settings, "DEBUG", False)
    settings.debug_log(_Logger(), "disabled", value=RaisingRepr())
