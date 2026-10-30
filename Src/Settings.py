"""Build config for OpenAgent: all global constants live here."""

from __future__ import annotations

# Build marker. Rewritten by Scripts/configure_build_profile.py
# (debug / release). Keep exactly one such line in this file.
DEBUG = True

# Context chat
CHAT_ID = 'global'

# Default values for the user config (ModuleConfig)
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TIMEOUT = 60
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_TOOL_RULES_ENABLED = True

# Provider endpoints, relative to base_url (OpenAI compatible)
CHAT_PATH = "/chat/completions"
MODELS_PATH = "/models"

# Limits
MAX_REPLY_CHARS = 6200
MAX_MODELS_LISTED = 50
MAX_DETAIL_CHARS = 200

# Tools
MAX_TOOL_ITERATIONS = 6
MAX_TOOL_OUTPUT_CHARS = 8000
DEFAULT_TOOL_TIMEOUT = 20
TOOL_LOG_JSON_CHARS = 60  # JSON chars shown per call in the tool log

# Seconds between rich-message progress refreshes while a request is running
PROGRESS_UPDATE_INTERVAL = 0.8

# Sub-command that lists provider models with a GET request
MODELS_ARG = "models"

# Premium emoji
CUSTOM_EMOJI = {
    "think": '<tg-emoji emoji-id="5352865883293267245">🎲</tg-emoji>',
    "ok": '<tg-emoji emoji-id="5350552124281368665">✅</tg-emoji>',
    "bals": '<tg-emoji emoji-id="5352830479877843657">🌐</tg-emoji>',
}

# Rich rules
RICH_HTML_RULES = """
Отвечай в HTML для Telegram Rich Messages.
Используй только эти теги:
- Абзацы: <p>...</p>
- Заголовки: <h1>…</h1> … <h6>…</h6>
- Инлайн: <b>, <i>, <u>, <s>, <code>, <mark>, <sub>, <sup>, <tg-spoiler>
- Ссылки: <a href="https://...">текст</a>
- Цитаты: <blockquote>…<cite>Автор</cite></blockquote> (или <blockquote expandable>)
- Код: <pre><code class="language-python">…</code></pre>
- Списки: <ul><li>…</li></ul> или <ol><li>…</li></ol>
- Футер: <footer>…</footer>, разделитель: <hr/>
- Таблицы: <table bordered striped>…</table>
  Внутри: <caption>, <thead>, <tbody>, <tr>, <th>, <td>.
  В ячейках — только inline-теги (<b>, <i>, <code>, <a>, <mark> и т.п.).
  Не используй внутри ячеек <p>, <pre>, <blockquote>, <ul>, <ol>.
  Максимум 20 колонок.
- Эмодзи: <tg-emoji emoji-id="...">😀</tg-emoji> (не знаешь emoji-id? — не пишешь)
- Время: <tg-time unix="..." format="wDT">...</tg-time>
- Формулы: <tg-math>x^2</tg-math>
Не используй Markdown. Не добавляй <html> или <body>. Только фрагмент.
"""

# Tool rules, appended to the system prompt when tool_rules is enabled
TOOL_RULES = """
У тебя есть инструменты (tools). Вызывай их, когда они нужны для ответа.
Не выдумывай результат инструмента и не вызывай несуществующие инструменты.
Аргументы передавай JSON-объектом строго по схеме инструмента.
Не уверен в аргументах — сначала вызови help_tool с именем инструмента.
Перед правкой файла прочитай его через read_file, итог правок смотри через diff.
Если инструмент вернул {{"ok": false}}, исправь аргументы или объясни ошибку.
Доступные инструменты:
{tools}
"""

# HTML rules for RichBoby
ALLOWED_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "cite", "pre", "footer", "hr",
    "ul", "ol", "li", "details", "summary", "aside",
    "b", "strong", "i", "em", "u", "ins", "s", "strike", "del",
    "code", "mark", "sub", "sup", "tg-spoiler",
    "a", "tg-emoji", "tg-time", "tg-math", "img",
    "table", "caption", "thead", "tbody", "tfoot",
    "tr", "th", "td",
}

VOID_TAGS = {"hr", "img"}

BLOCK_TAGS_FORBIDDEN_IN_CELLS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "pre", "footer", "hr",
    "ul", "ol", "li", "details", "summary", "aside",
    "table", "tr", "td", "th",
}

# Context
DEFAULT_CONTEXT_ENABLED = True
DEFAULT_CONTEXT_SIZE = 10
MAX_CONTEXT_MESSAGES = 50
CONTEXT_RESET_ARGS = ("reset", "clear", "forget")
