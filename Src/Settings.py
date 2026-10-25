"""Build config for OpenAgent: all global constants live here."""

from __future__ import annotations

# Build marker. Rewritten by Scripts/configure_build_profile.py
# (debug / release). Keep exactly one such line in this file.
DEBUG = False

# Default values for the user config (ModuleConfig)
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TIMEOUT = 60
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# Provider endpoints, relative to base_url (OpenAI compatible)
CHAT_PATH = "/chat/completions"
MODELS_PATH = "/models"

# Limits
MAX_REPLY_CHARS = 3500
MAX_MODELS_LISTED = 50
MAX_DETAIL_CHARS = 200

# Seconds between rich-message progress refreshes while a request is running
PROGRESS_UPDATE_INTERVAL = 1

# Sub-command that lists provider models with a GET request
MODELS_ARG = "models"

CUSTOM_EMOJI = {
    "think": '<tg-emoji emoji-id="5352865883293267245">🎲</tg-emoji>',
    "ok": '<tg-emoji emoji-id="5350552124281368665">✅</tg-emoji>',
    "bals": '<tg-emoji emoji-id="5352830479877843657">🌐</tg-emoji>',
}
