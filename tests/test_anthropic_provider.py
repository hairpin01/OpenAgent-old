from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import cubkit

from OpenAgentLib.Placeholders import OpenAgentProviderService
from OpenAgentLib.Plugin.PluginsEngine import _OpenAgentAgentLoopMixin

SRC = Path(__file__).resolve().parents[1] / "Src"
sys.path.insert(0, str(SRC))
src_package = ModuleType("Src")
src_package.__path__ = [str(SRC)]
sys.modules.setdefault("Src", src_package)
original_load_strings = cubkit.load_strings
cubkit.load_strings = lambda: (lambda key, **_kwargs: key)
try:
    OpenAgent = importlib.import_module("Src.OpenAgentMain").OpenAgent
finally:
    cubkit.load_strings = original_load_strings


def test_anthropic_is_a_named_provider_and_config_choice() -> None:
    provider_schema = next(
        item for item in OpenAgent.config.schema if item["key"] == "provider"
    )

    assert "anthropic" in OpenAgent.PROVIDERS
    assert OpenAgent.PROVIDER_LABELS["anthropic"] == "Anthropic"
    assert provider_schema["choices"] == list(OpenAgent.PROVIDERS)
    assert "anthropic" in provider_schema["choices"]
    assert OpenAgent.BASE_URLS["anthropic"] == "https://api.anthropic.com"


def test_openai_api_mode_is_configured_and_selects_responses_only_for_openai() -> None:
    mode_schema = next(
        item for item in OpenAgent.config.schema if item["key"] == "openai_api_mode"
    )
    assert mode_schema["default"] == "chat"
    assert mode_schema["choices"] == ["chat", "responses"]
    assert "provider=openai" in mode_schema["description"]

    class _ConfigProbe(_OpenAgentAgentLoopMixin):
        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config

    config = {item["key"]: item["default"] for item in OpenAgent.config.schema}
    config["openai_api_mode"] = "responses"
    probe = _ConfigProbe(config)
    assert probe._uses_openai_responses_api("openai") is True
    assert probe._uses_openai_responses_api("openrouter") is False


def test_anthropic_uses_generic_key_and_default_model_resolution() -> None:
    service = OpenAgentProviderService()
    config = {"provider": "anthropic", "api_key": "  secret-key  ", "model": ""}

    assert service.provider(config, OpenAgent.PROVIDERS) == "anthropic"
    assert service.api_key(config) == "secret-key"
    assert (
        service.model(config, OpenAgent.DEFAULT_MODELS, "anthropic")
        == "claude-sonnet-4-5"
    )
