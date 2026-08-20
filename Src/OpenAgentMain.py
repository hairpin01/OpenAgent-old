# SPDX-License-Identifier: MIT
# scope: heroku_min 9.9.9
# -- repo data --
# repo: https://github.com/hairpin01/repo-MCUB-fork/
# source: https://github.com/hairpin01/OpenAgent-old/
# -- end --
# scop: kernel min v1.4.6

from __future__ import annotations

import asyncio
import contextlib
import html
import io
import re
import time
import uuid
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

from cubkit import load_strings
import Settings as OpenAgentSettings
from Settings import debug_log

from core.lib.loader.module_base import ModuleBase, bot_command, callback, command
from core.lib.loader.module_config import (
    Boolean,
    Choice,
    ConfigValue,
    Float,
    Group,
    Row,
    Answer,
    Integer,
    List,
    ModuleConfig,
    Secret,
    String,
)

if TYPE_CHECKING:
    from core.lib.types import InlineMessage, Event

OpenAgentSettings.configure_debug(OpenAgentSettings.debug_for_artifact(__file__))

try:
    from OpenAgentLib.OpenAgentMixins import (
        _OpenAgentLifecycleMixin,
        _OpenAgentProviderMixin,
        _OpenAgentTodoMixin,
        _OpenAgentToolDisplayMixin,
        _OpenAgentContextMixin,
        _OpenAgentSessionsMixin,
        _OpenAgentPluginSkillMixin,
        _OpenAgentRuntimeToolsMixin,
        _OpenAgentTelegramMediaMixin,
        _OpenAgentStatusMixin,
        _OpenAgentAgentLoopMixin,
        _OpenAgentResponseMixin,
        _OpenAgentToolRegistryMixin,
    )
except Exception as e:
    raise RuntimeError(e) from e  # debug


class OpenAgent(
    _OpenAgentLifecycleMixin,
    _OpenAgentProviderMixin,
    _OpenAgentTodoMixin,
    _OpenAgentToolDisplayMixin,
    _OpenAgentContextMixin,
    _OpenAgentSessionsMixin,
    _OpenAgentPluginSkillMixin,
    _OpenAgentRuntimeToolsMixin,
    _OpenAgentTelegramMediaMixin,
    _OpenAgentStatusMixin,
    _OpenAgentAgentLoopMixin,
    _OpenAgentResponseMixin,
    _OpenAgentToolRegistryMixin,
    ModuleBase,
):
    DEBUG = OpenAgentSettings.DEBUG
    name = "OpenAgent"
    version = "0.8.1-main.build:1052"
    author = "@dev_dolbaeb && @Hairpin00"
    description = {
        "ru": "ИИ агент в юзерботе с новой архитектурой инструментов",
        "en": "AI agent in userbot with refreshed tool architecture",
        "rofl": "ИИ агент, который делает вид, что всё контролирует",
        "linux": "AI agent daemon with tool-oriented runtime",
    }
    strings = load_strings()

    def _debug_log(self, event: str, **fields: Any) -> None:
        if not self.DEBUG:
            return
        debug_log(self.log, event, **fields)

    PROVIDERS = (
        "openai",
        "google",
        "openrouter",
        "groq",
        "deepseek",
        "xai",
        "other",
    )
    PROVIDER_LABELS = {
        "openai": "OpenAI",
        "google": "Google",
        "openrouter": "OpenRouter",
        "groq": "Groq",
        "deepseek": "DeepSeek",
        "xai": "xAI",
        "other": "Other",
    }

    async def on_unload(self) -> None:
        tasks = set(getattr(self, "_background_tool_tasks", {}).values())
        tasks.update(getattr(self, "_plugin_unload_tasks", set()))
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        for waiters_name in (
            "_inline_status_waiters",
            "_tool_confirmation_waiters",
        ):
            for waiter in getattr(self, waiters_name, {}).values():
                if not waiter.done():
                    waiter.cancel()

        session_manager = getattr(self, "session_manager", None)
        if session_manager is not None:
            await session_manager.close()
        http_client = getattr(self, "_http_client", None)
        if http_client is not None:
            await http_client.close()

        await super().on_unload()

    DEFAULT_MODELS = {
        "openai": "gpt-5.5",
        "google": "gemini-1.5-flash",
        "openrouter": "openai/gpt-4o-mini",
        "groq": "llama-3.3-70b-versatile",
        "deepseek": "deepseek-chat",
        "xai": "grok-2-latest",
        "other": "gpt-4o-mini",
    }
    BASE_URLS = {
        "openai": "https://api.openai.com/v1",
        "google": "https://generativelanguage.googleapis.com/v1beta",
        "openrouter": "https://openrouter.ai/api/v1",
        "groq": "https://api.groq.com/openai/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "xai": "https://api.x.ai/v1",
    }
    PLACEHOLDER_KEYS = (
        "{agent_version}, {provider}, {provider_key}, {model}, {reasoning_effort}, "
        "{chat_id}, {user_id}, {session_name}, {session_messages}, "
        "{runtime_comments_count}, {runtime_comments}, {tool_count}, {available_tool_count}, "
        "{elapsed}, {input_tokens}, {output_tokens}, {total_tokens}, {thinking}, "
        "{todo}, {random}, {prefix}, {time}, {date}"
    )
    WEB_SEARCH_RE = re.compile(
        r"<web_search>\s*(.*?)\s*</web_search>", re.DOTALL | re.I
    )
    SEND_RE = re.compile(
        r'<send_message(?:\s+chat=["\']([^"\']+)["\'])?\s*>(.*?)</send_message>',
        re.DOTALL | re.I,
    )
    SKILL_RE = re.compile(
        r'<skill\s+name=["\']([^"\']+)["\']\s*>(.*?)</skill>', re.DOTALL | re.I
    )
    CREATE_CHANNEL_RE = re.compile(
        r"<create_channel([^>]*)>(.*?)</create_channel>", re.DOTALL | re.I
    )
    CREATE_GROUP_RE = re.compile(
        r"<create_group([^>]*)>(.*?)</create_group>", re.DOTALL | re.I
    )
    CREATE_BOT_RE = re.compile(
        r"<create_bot([^>]*)>(.*?)</create_bot>", re.DOTALL | re.I
    )
    SEARCH_MESSAGES_RE = re.compile(
        r"<search_messages([^>]*)>(.*?)</search_messages>", re.DOTALL | re.I
    )
    UPDATE_PROFILE_RE = re.compile(
        r"<update_profile([^>]*)>(.*?)</update_profile>", re.DOTALL | re.I
    )
    SET_PROFILE_PHOTO_RE = re.compile(
        r"<set_profile_photo([^>]*)>(.*?)</set_profile_photo>", re.DOTALL | re.I
    )
    DELETE_MESSAGES_RE = re.compile(
        r"<delete_messages([^>]*)>(.*?)</delete_messages>", re.DOTALL | re.I
    )
    FORWARD_MESSAGE_RE = re.compile(
        r"<forward_message([^>]*)>(.*?)</forward_message>", re.DOTALL | re.I
    )
    DOWNLOAD_MEDIA_RE = re.compile(
        r"<download_media([^>]*)>(.*?)</download_media>", re.DOTALL | re.I
    )
    GENERATED_FILE_RE = re.compile(
        r'<file\s+name=["\']([^"\']+)["\']\s*>(.*?)</file>',
        re.DOTALL | re.I,
    )
    MCUB_DOCS_URL = "https://x0.at/y2rb.md"
    TOOL_CALL_RE = re.compile(
        r"<([a-z0-9._]+)([^>]*)>(.*?)</\1>|<([a-z0-9._]+)([^>]*)/?>", re.DOTALL | re.I
    )
    TOOL_CALL_JSON_RE = re.compile(r"```tool_call\s*(.*?)```", re.DOTALL | re.I)
    TOOL_REGISTRY = ()
    # Built-in tools are now discovered dynamically from
    # OpenAgentLib/SystemPlugins/<group>/<tool>.py.
    AGENT_MAX_STEPS = 15
    PREMIUM_EMOJIS = {
        "claude": '<tg-emoji emoji-id="5368808376694248152">💬</tg-emoji>',
        "start": '<tg-emoji emoji-id="5368434680179758177">🏁</tg-emoji>',
        "workout": '<tg-emoji emoji-id="5368387680352637360">🏋️‍♂️</tg-emoji>',
        "party": '<tg-emoji emoji-id="5368635272332352173">🎉</tg-emoji>',
        "loading_dots": '<tg-emoji emoji-id="5328311576736833844">🔴</tg-emoji>',
        "loading_wait": '<tg-emoji emoji-id="5326015457155620929">😐</tg-emoji>',
        "reconnect": '<tg-emoji emoji-id="5325872701032635449">⏳</tg-emoji>',
        "loading_squares": '<tg-emoji emoji-id="5334960765931626355">🎲</tg-emoji>',
        "loading_lava": '<tg-emoji emoji-id="5310041868191407556">🩸</tg-emoji>',
        "soon": '<tg-emoji emoji-id="5411382892850871522">🔜</tg-emoji>',
        "top": '<tg-emoji emoji-id="5411132595041765682">🔝</tg-emoji>',
        "linux": '<tg-emoji emoji-id="5300957668762987048">👩‍💻</tg-emoji>',
        "js": '<tg-emoji emoji-id="5300896259320586992">👩‍💻</tg-emoji>',
        "ts": '<tg-emoji emoji-id="5301254000031572585">👩‍💻</tg-emoji>',
        "grid": '<tg-emoji emoji-id="5294096239464295059">🔵</tg-emoji>',
        "done": '<tg-emoji emoji-id="4916036072560919511">✅</tg-emoji>',
        "warn": '<tg-emoji emoji-id="4915853119839011973">⚠️</tg-emoji>',
        "link": '<tg-emoji emoji-id="4916086774649848789">🔗</tg-emoji>',
        "web": '<tg-emoji emoji-id="4906943755644306322">🌐</tg-emoji>',
        "telegram": '<tg-emoji emoji-id="4918203446202467778">💙</tg-emoji>',
        "at": '<tg-emoji emoji-id="5082413149873767213">💙</tg-emoji>',
        "lock": '<tg-emoji emoji-id="4904500559203009298">🔒</tg-emoji>',
        "bubble": '<tg-emoji emoji-id="4918408122868958076">🖱️</tg-emoji>',
        "back": '<tg-emoji emoji-id="5352759161945867747">🔙</tg-emoji>',
        "block": '<tg-emoji emoji-id="5408830797513784663">🚫</tg-emoji>',
        "blink": '<tg-emoji emoji-id="5411528341918356895">⚪️</tg-emoji>',
        "terminal": '<tg-emoji emoji-id="5409076727341154520">⚙️</tg-emoji>',
        "num_0": '<tg-emoji emoji-id="5140999334174655345">0️⃣</tg-emoji>',
        "num_1": '<tg-emoji emoji-id="5141109049114232089">1️⃣</tg-emoji>',
        "num_2": '<tg-emoji emoji-id="5140871649091912628">2️⃣</tg-emoji>',
        "num_3": '<tg-emoji emoji-id="5141399818400170896">3️⃣</tg-emoji>',
        "num_4": '<tg-emoji emoji-id="5138822752123225428">4️⃣</tg-emoji>',
        "num_5": '<tg-emoji emoji-id="5141062672057369534">5️⃣</tg-emoji>',
        "num_6": '<tg-emoji emoji-id="5139005588881015916">6️⃣</tg-emoji>',
        "num_7": '<tg-emoji emoji-id="5140999557512954818">7️⃣</tg-emoji>',
        "num_8": '<tg-emoji emoji-id="5141013683660391172">8️⃣</tg-emoji>',
        "num_9": '<tg-emoji emoji-id="5141137309999039199">9️⃣</tg-emoji>',
    }
    config = ModuleConfig(
        Group(
            "Provider & Model 🧠",
            [
                ConfigValue(
                    "provider",
                    "openai",
                    description="Provider: openai, google, openrouter, groq, deepseek, xai, other",
                    validator=Choice(choices=list(PROVIDERS)),
                ),
                ConfigValue(
                    "api_key",
                    "",
                    description="API key for the selected provider",
                    validator=Secret(),
                ),
                ConfigValue(
                    "model",
                    "",
                    description="Model name. Empty means provider default",
                    validator=String(),
                ),
                ConfigValue(
                    "custom_base_url",
                    "",
                    description="Endpoint for provider=other, e.g. https://api.deepseek.com/v1",
                    validator=String(),
                ),
                ConfigValue(
                    "system_prompt",
                    "You are OpenAgent inside a Telegram userbot. Help the user directly. You may inspect the local workspace through terminal commands when needed.",
                    description="System prompt for the agent",
                    validator=String(),
                ),
                ConfigValue(
                    "temperature",
                    0.7,
                    description="Sampling temperature",
                    validator=Float(min=0.0, max=2.0),
                ),
                ConfigValue(
                    "max_tokens",
                    1200,
                    description="Maximum response tokens",
                    validator=Integer(min=64, max=32768),
                ),
                ConfigValue(
                    "reasoning_effort",
                    "off",
                    description="Reasoning effort for models/providers that support it: off, low, medium, high, xhigh",
                    validator=Choice(choices=["off", "low", "medium", "high", "xhigh"]),
                ),
                ConfigValue(
                    "timeout",
                    180,
                    description="HTTP timeout seconds for each provider request. Increase for slow reasoning/code tasks.",
                    validator=Integer(min=10, max=600),
                ),
                ConfigValue(
                    "provider_reconnect_attempts",
                    2,
                    description="Maximum retries for transient provider failures",
                    validator=Integer(min=0, max=5),
                ),
                ConfigValue(
                    "agent_max_steps",
                    6,
                    description="Maximum model loop iterations before forced finalization",
                    validator=Integer(min=1, max=15),
                ),
                ConfigValue(
                    "agent_max_model_calls",
                    10,
                    description="Maximum provider attempts in the main agent loop, including retries",
                    validator=Integer(min=1, max=20),
                ),
                ConfigValue(
                    "agent_deadline",
                    180,
                    description="Overall agent request deadline in seconds",
                    validator=Integer(min=15, max=900),
                ),
                ConfigValue(
                    "context_window_tokens",
                    16000,
                    description="Estimated provider context-window budget",
                    validator=Integer(min=2048, max=1000000),
                ),
                ConfigValue(
                    "context_reserve_tokens",
                    2400,
                    description="Tokens reserved for output and tool follow-ups",
                    validator=Integer(min=256, max=65536),
                ),
            ],
            description="AI provider, credentials, model and request limits",
            button_text="🧠 Provider",
            key="provider_model",
        ),
        Group(
            "Tools & Permissions 🛠",
            [
                ConfigValue(
                    "terminal_enabled",
                    True,
                    description="Allow the agent to execute terminal commands",
                    validator=Boolean(),
                ),
                ConfigValue(
                    "terminal_steps",
                    3,
                    description="Maximum terminal commands per request",
                    validator=Integer(min=0, max=10),
                ),
                ConfigValue(
                    "terminal_timeout",
                    30,
                    description="Terminal command timeout seconds",
                    validator=Integer(min=3, max=120),
                ),
                ConfigValue(
                    "web_search_enabled",
                    True,
                    description="Allow the agent to search the web",
                    validator=Boolean(),
                ),
                ConfigValue(
                    "web_search_steps",
                    3,
                    description="Maximum web searches per request",
                    validator=Integer(min=0, max=10),
                ),
                ConfigValue(
                    "mcub_use",
                    False,
                    description="Allow the agent to execute MCUB userbot commands",
                    validator=Boolean(),
                ),
                ConfigValue(
                    "mcub_steps",
                    3,
                    description="Maximum MCUB commands per request",
                    validator=Integer(min=0, max=10),
                ),
                ConfigValue(
                    "send_messages_enabled",
                    True,
                    description="Allow the agent to send messages as the userbot",
                    validator=Boolean(),
                ),
                ConfigValue(
                    "send_message_steps",
                    3,
                    description="Maximum userbot messages sent per request",
                    validator=Integer(min=0, max=10),
                ),
                ConfigValue(
                    "create_chats_enabled",
                    True,
                    description="Allow the agent to create channels/groups",
                    validator=Boolean(),
                ),
                ConfigValue(
                    "create_chat_steps",
                    2,
                    description="Maximum channels/groups created per request",
                    validator=Integer(min=0, max=5),
                ),
                ConfigValue(
                    "create_bots_enabled",
                    True,
                    description="Allow the agent to create Telegram bots via BotFather",
                    validator=Boolean(),
                ),
                ConfigValue(
                    "create_bot_steps",
                    1,
                    description="Maximum Telegram bots created per request",
                    validator=Integer(min=0, max=3),
                ),
                ConfigValue(
                    "account_tools_enabled",
                    True,
                    description="Allow the agent to edit profile/join chats/read/search messages",
                    validator=Boolean(),
                ),
                ConfigValue(
                    "account_tool_steps",
                    5,
                    description="Maximum account-level tools per request",
                    validator=Integer(min=0, max=15),
                ),
                ConfigValue(
                    "chat_management_enabled",
                    True,
                    description="Allow the agent to manage chats: mute, ban, promote, title, slowmode",
                    validator=Boolean(),
                ),
                ConfigValue(
                    "chat_management_steps",
                    5,
                    description="Maximum chat-management tools per request",
                    validator=Integer(min=0, max=15),
                ),
                ConfigValue(
                    "media_max_bytes",
                    8_000_000,
                    description="Maximum replied media bytes sent to AI",
                    validator=Integer(min=1024, max=25_000_000),
                ),
            ],
            description="Terminal, web, MCUB and Telegram action limits",
            button_text="🛠 Tools",
            key="tools_permissions",
        ),
        Group(
            "Context & Memory 🧾",
            [
                ConfigValue(
                    "context_enabled",
                    True,
                    description="Remember chat context between .oa requests",
                    validator=Boolean(),
                ),
                ConfigValue(
                    "context_turns",
                    10,
                    description="How many user/assistant turns to remember per chat",
                    validator=Integer(min=0, max=50),
                ),
                ConfigValue(
                    "context_compaction_enabled",
                    True,
                    description="Automatically summarize old chat context when it becomes too large",
                    validator=Boolean(),
                ),
                ConfigValue(
                    "context_compaction_chars",
                    18000,
                    description="Legacy character threshold used by older configurations",
                    validator=Integer(min=2000, max=200000),
                ),
                ConfigValue(
                    "context_compaction_tokens",
                    10000,
                    description="Compact remembered chat context after this estimated token count",
                    validator=Integer(min=1000, max=500000),
                ),
                ConfigValue(
                    "context_compaction_keep_turns",
                    2,
                    description="Recent user/assistant turns to keep verbatim after compaction",
                    validator=Integer(min=0, max=10),
                ),
                ConfigValue(
                    "context_compaction_max_tokens",
                    900,
                    description="Maximum tokens used for the compaction summary response",
                    validator=Integer(min=128, max=4096),
                ),
                ConfigValue(
                    "tool_memory_enabled",
                    False,
                    description="Remember concise notes from tool outputs for next requests",
                    validator=Boolean(),
                ),
                ConfigValue(
                    "tool_memory_items",
                    20,
                    description="Maximum remembered tool notes per chat",
                    validator=Integer(min=1, max=200),
                ),
                ConfigValue(
                    "tool_memory_max_chars",
                    500,
                    description="Maximum characters per remembered tool note",
                    validator=Integer(min=80, max=4000),
                ),
            ],
            description="Chat memory, compaction and tool notes",
            button_text="🧾 Context",
            key="context_memory",
        ),
        Row(),
        Group(
            "Templates & Display 🎨",
            [
                ConfigValue(
                    "response_header",
                    '<blockquote><a href="tg://emoji?id=6010179991944305029">☺️</a> <strong>OpenAgent</strong>: <a href="tg://emoji?id=5325872701032635449">⏳</a>  <em>{elapsed}</em>s\n• <u>{provider}/{model}</u>  •  <code>{reasoning_effort}</code>\n| | | | | | | | | | | | | | | | | | | | | | | | | | |\n<a href="tg://emoji?id=5408994848084624514">💸</a> <strong>in</strong> <em>{input_tokens}</em>, <strong>out</strong> <em>{output_tokens}</em> | <b>total</b>\n<i>{total_tokens}</i> | <strong>tool use:</strong> <em>{tool_count}</em></blockquote>\n<blockquote expandable><i>{thinking}</i></blockquote>',
                    description="Final response header template. Placeholders: "
                    + PLACEHOLDER_KEYS,
                    validator=String(),
                ),
                ConfigValue(
                    "request_label",
                    '<a href="tg://emoji?id=6010352868672936598"><strong>🐈‍⬛</strong></a><strong></strong><strong> Prompt:</strong>',
                    description="Request block label template. Placeholders: "
                    + PLACEHOLDER_KEYS,
                    validator=String(),
                ),
                ConfigValue(
                    "response_label",
                    '<a href="tg://emoji?id=6010286885090368072"><strong>❌</strong></a><strong></strong><strong> Answer:</strong>',
                    description="Response block label template. Placeholders: "
                    + PLACEHOLDER_KEYS,
                    validator=String(),
                ),
                ConfigValue(
                    "thinking_template",
                    '<blockquote><a href="tg://emoji?id=6010292571627069263">😎</a> <u>{provider}/{model}</u> • <em>prepares the response...</em></blockquote >\n<blockquote><a href="tg://emoji?id=5404857686477015710">🔄</a><strong><em> {random}</em></strong><em></em></blockquote>',
                    description="Initial loading/thinking message template. Placeholders: "
                    + PLACEHOLDER_KEYS,
                    validator=String(),
                ),
                ConfigValue(
                    "tool_display_template",
                    '<blockquote expandable><i>{thinking_line}</i></blockquote>\n<blockquote expandable><strong>┌|</strong> {tool_state_emoji_html} {status_emoji_html} <em>{status_text}</em> <code>{tool}</code>\n<strong>└|</strong> <a href="tg://emoji?id=6010570945637392851">🥳</a>  <b>Round:</b> <code>{round}/{round_total}</code> • <b>Reasoning:</b>\n<code>{reasoning_effort}</code>\n</blockquote><blockquote><a href="tg://emoji?id=5310041868191407556">🩸</a> <strong>{activity_line}</strong></blockquote>\n<blockquote expandable><a href="tg://emoji?id=6012361831035705571">😪</a> <strong>Log tools</strong>\n<code>{log_lines}</code></blockquote>',
                    description="Tool execution status template. Raw: {tool}, {title}, {value}, {log}, {step}. Semantic: {round}, {round_total}, {progress_bar}, {progress_percent}, {status_emoji}, {status_icon}, {status_emoji_html}, {status_icon_html}, {status_text}, {tool_state}, {tool_state_emoji}, {tool_state_icon}, {tool_state_emoji_html}, {tool_state_icon_html}, {tool_running_emoji}, {tool_running_icon}, {tool_running_emoji_html}, {tool_running_icon_html}, {tool_done_emoji}, {tool_done_icon}, {tool_done_emoji_html}, {tool_done_icon_html}, {tool_group}, {tool_short}, {tool_input}, {tool_input_block}, {thinking_line}, {thinking_block}, {log_lines}, {log_block}, {log_count}, {elapsed_line}, {token_line}, {model_line}, {activity_line}. General placeholders: "
                    + PLACEHOLDER_KEYS,
                    validator=String(),
                ),
                ConfigValue(
                    "tool_status_emojis",
                    "thinking=❔\nterminal=🖥\nweb=🌐\nfile=📦\nmcub=🧲\nmessage=💬\ndialog=🗂\nchat=🐈‍⬛\nmoderation=🛡\nprofile=👤\ncontacts=👥\ncreation=✨\nskills=🧠\ncode=🧬\ncontext=🧾\nutility=🛠\ndefault=🛠",
                    description="Custom emoji/icon map for {status_emoji}/{status_icon}. Format: group_or_tool=emoji per line. Tool-specific keys like terminal.run or thinking.note override groups like terminal/thinking. Premium emoji HTML is allowed via {status_emoji_html}/{status_icon_html}.",
                    validator=String(),
                ),
                ConfigValue(
                    "tool_display_max_chars",
                    1200,
                    description="Maximum chars from current tool input shown in status form",
                    validator=Integer(min=0, max=4000),
                ),
                ConfigValue(
                    "tool_trace_inline_max_chars",
                    6000,
                    description="Maximum chars of a tool call kept inline before the full output is saved to openagent_tool_outputs and replaced by a file path plus preview",
                    validator=Integer(min=0, max=50000),
                ),
                ConfigValue(
                    "tool_display_log_lines",
                    8,
                    description="How many recent tool names to show in status form",
                    validator=Integer(min=0, max=30),
                ),
                ConfigValue(
                    "thinking_display_limit",
                    3,
                    description="How many recent thinking.note entries to show in {thinking}",
                    validator=Integer(min=0, max=20),
                ),
                ConfigValue(
                    "thinking_empty_text",
                    "Модель ещё не думала.",
                    description="Text for {thinking} when no thinking.note entries exist",
                    validator=String(),
                ),
                ConfigValue(
                    "thinking_bullet",
                    "•",
                    description="Prefix marker for each thinking.note line in {thinking}. Empty disables the marker",
                    validator=String(),
                ),
                ConfigValue(
                    "random_strings",
                    ["Thinking...", "Думаю...", "Генерирую..."],
                    description="Random lines for {random}",
                    validator=List(
                        item_type=str,
                    ),
                ),
                ConfigValue(
                    "todo_status_emojis",
                    "pending=...\nopen=>>>\nclosed=---",
                    description="State markers for {todo}. Format: pending=..., open=>>>, closed=---",
                    validator=String(),
                ),
                ConfigValue(
                    "placeholders",
                    "",
                    description="Available OpenAgent placeholders (auto-generated)",
                    validator=String(),
                ),
            ],
            description="Response headers, labels, thinking and tool status templates",
            button_text="🎨 Display",
            key="templates_display",
        ),
        Group(
            "Repo Context & Skills 📚",
            [
                ConfigValue(
                    "repo_context_enabled",
                    True,
                    description="Inject local workspace snapshot into system prompt",
                    validator=Boolean(),
                ),
                ConfigValue(
                    "repo_context_max_chars",
                    7000,
                    description="Maximum chars used for repo context in system prompt",
                    validator=Integer(min=500, max=30000),
                ),
                ConfigValue(
                    "skills_enabled",
                    True,
                    description="Enable loading OpenAgent skills into the system prompt",
                    validator=Boolean(),
                ),
                ConfigValue(
                    "skills_trigger_mode",
                    "auto",
                    description="When to load skills: auto = only on keyword match, always = every request, off = never",
                    validator=String(),
                ),
                ConfigValue(
                    "skill_repo_url",
                    "https://raw.githubusercontent.com/hairpin01/repo-MCUB-fork/main/OpenAgent/skills",
                    description="Base URL for installable OpenAgent skills repository",
                    validator=String(),
                ),
            ],
            description="Workspace context and OpenAgent skills loading",
            button_text="📚 Skills",
            key="repo_skills",
        ),
        Group(
            "Tool Confirmations 🛡",
            [
                ConfigValue(
                    "tool_confirmation_enabled",
                    True,
                    description="Ask for confirmation before tools that can change files, chats, account state, or run commands",
                    validator=Boolean(),
                ),
                ConfigValue(
                    "tool_confirmation_mode",
                    "medium",
                    description="How often to ask before tools: low = only critical/destructive, medium = write/actions, high = almost every non-read tool",
                    validator=Choice(choices=["low", "medium", "high"]),
                ),
                ConfigValue(
                    "tool_confirmation_template",
                    '<blockquote><a href="tg://emoji?id=6010201728773790293">😈</a> Continue?\n<a href="tg://emoji?id=6012317326584583729">😐</a> Tool: {tool} • {elapsed}s</blockquote>\n<blockquote expandable><a href="tg://emoji?id=6010394680179562842">😶</a> <b>What will be completed</b>\n<a href="tg://emoji?id=6010292550152230657">☀️</a> <code>{value}</code></blockquote>',
                    description="Confirmation form template. Placeholders: {tool}, {value}, {elapsed}, {elapsed_line}",
                    validator=String(),
                ),
                ConfigValue(
                    "tool_confirmation_yes_text",
                    "Выполнить",
                    description="Confirm button text for dangerous tools",
                    validator=String(),
                ),
                ConfigValue(
                    "tool_confirmation_no_text",
                    "Не сейчас",
                    description="Cancel button text for dangerous tools",
                    validator=String(),
                ),
                ConfigValue(
                    "tool_confirmation_timeout",
                    900,
                    description="Seconds to wait for dangerous tool confirmation",
                    validator=Integer(min=10, max=3600),
                ),
            ],
            description="Confirmation policy and prompt/button templates",
            button_text="🛡 Confirm",
            key="confirmations",
        ),
        Row(),
        Answer("❔ About", "AI agent in userbot with refreshed tool architecture"),
    )
    SESSION_LIMIT = 20
    from .MCUBEvent import _MCUBEvent

    @callback(ttl=900)
    async def _open_sessions_panel(
        self, call: InlineMessage, chat_id: int | None = None
    ) -> None:
        cid = int(
            chat_id
            or getattr(call, "chat_id", 0)
            or getattr(call, "_openagent_source_chat_id", 0)
            or 0
        )
        if not cid:
            await call.answer(
                self.strings("error", error="chat_id is missing"), alert=True
            )
            return
        await self._show_sessions_panel(call, cid)

    @callback(ttl=900)
    async def _return_to_last_response(self, call: InlineMessage, chat_id: int) -> None:
        cid = int(chat_id)
        saved_turn = self._last_saved_assistant_turn(cid)
        if not saved_turn:
            await call.answer(self.strings("saved_response_missing"), alert=True)
            return
        prompt, answer, thinking_notes = saved_turn
        with contextlib.suppress(Exception):
            setattr(call, "_openagent_source_chat_id", cid)
        self._set_placeholder_context(call)
        await self._reply_text(
            call,
            answer,
            title=self._response_title(
                0.0, tool_count=0, thinking_notes=thinking_notes
            ),
            prompt=prompt,
            thinking_notes=thinking_notes,
            buttons=self._final_buttons(
                cid,
                prompt,
                prompt,
                [],
                source_event=call,
            ),
            edit_current=True,
        )
        self._store_last_loading(cid, call)

    @callback(ttl=900)
    async def _switch_session(self, call: InlineMessage, session_id: str) -> None:
        session = self._sessions.get(str(session_id))
        if session is None:
            await call.answer(self.strings("skill_not_found"), alert=True)
            return
        self._set_active_session(session.chat_id, session.id)
        self.session_manager.set_preference(session.chat_id, "continue")
        await self._show_sessions_panel(
            call,
            session.chat_id,
            alert=self.strings("chat_switched", name=session.name),
        )

    @callback(ttl=900)
    async def _remember_session_choice(self, call: InlineMessage, chat_id: int) -> None:
        self.session_manager.set_preference(int(chat_id), "continue")
        await self._save_sessions()
        await call.answer(self.strings("chat_choice_saved"), alert=True)

    @callback(ttl=900)
    async def _delete_active_session(self, call: InlineMessage, chat_id: int) -> None:
        cid = int(chat_id)
        sessions = self._get_chat_sessions(cid)
        if len(sessions) <= 1:
            await call.answer(self.strings("chat_delete_last"), alert=True)
            return
        active = self._get_active_session(cid)
        self._sessions.pop(active.id, None)
        remaining = self._get_chat_sessions(cid)
        self._active_session[cid] = remaining[0].id
        await self._save_sessions()
        await self._show_sessions_panel(call, cid, alert=self.strings("chat_deleted"))

    @callback(ttl=900)
    async def _run_pending_here(self, call: InlineMessage, prompt_token: str) -> None:
        """Run pending prompt in the current active session."""
        chat_id = self._pending_prompts.get(prompt_token, {}).get("chat_id")
        if chat_id:
            self.session_manager.set_preference(int(chat_id), "continue")
        await self._execute_pending(call, prompt_token)

    @callback(ttl=900)
    async def _run_pending_in(
        self,
        call: InlineMessage,
        prompt_token: str,
        session_id: str,
    ) -> None:
        """Switch to another session, then run the pending prompt."""
        session = self._sessions.get(str(session_id))
        if session is None:
            with contextlib.suppress(Exception):
                await call.answer(self.strings("chat_delete_last"), alert=True)
            return
        self._set_active_session(session.chat_id, session.id)
        self.session_manager.set_preference(session.chat_id, "continue")
        await self._execute_pending(call, prompt_token)

    @callback(ttl=900)
    async def _remember_pref_continue(
        self,
        call: InlineMessage,
        prompt_token: str,
        chat_id: int,
    ) -> None:
        """Save 'always continue here' pref then run pending in current session."""
        self.session_manager.set_preference(int(chat_id), "continue")
        with contextlib.suppress(Exception):
            await call.answer(self.strings("pref_saved"), alert=False)
        await self._execute_pending(call, prompt_token)

    @callback(ttl=900)
    async def _remember_pref_new(
        self,
        call: InlineMessage,
        prompt_token: str,
        chat_id: int,
    ) -> None:
        """Save 'always create new' pref, create new session, then run."""
        cid = int(chat_id)
        self.session_manager.set_preference(cid, "new")
        self._fresh_session(cid)
        with contextlib.suppress(Exception):
            await call.answer(self.strings("pref_saved"), alert=False)
        await self._execute_pending(call, prompt_token)

    @callback(ttl=900)
    async def _confirm_tool_action(
        self,
        call: InlineMessage,
        token: str | None = None,
        approved: bool = False,
    ) -> None:
        if token:
            future = self._tool_confirmation_waiters.get(token)
            if future is not None and not future.done():
                future.set_result(bool(approved))
        with contextlib.suppress(Exception):
            await call.answer(
                (
                    self.strings("tool_confirmation_approved")
                    if approved
                    else self.strings("cancelled")
                ),
                alert=False,
            )

    @callback(ttl=900)
    async def _activate_inline_status(
        self, call: InlineMessage, token: str | None = None
    ) -> None:
        if token:
            future = self._inline_status_waiters.get(token)
            if future is not None and not future.done():
                future.set_result(call)
        with contextlib.suppress(Exception):
            await call.answer()

    def _oa_arg_parser(self, event: Event) -> Any | None:
        with contextlib.suppress(Exception):
            return self.args(event)
        return None

    def _oa_prompt_from_parser(self, parser: Any | None) -> str:
        if parser is None:
            return ""
        raw = str(getattr(parser, "raw_args", "") or "")
        raw = re.sub(r"(?<!\S)--test(?:=\S+|\s+\S+)?", "", raw)
        raw = re.sub(
            r"(?<!\S)--new(?:=(?:\{[^}]*\}|\"[^\"]*\"|'[^']*'|\S*))?(?=\s|$)", "", raw
        )
        raw = re.sub(r"(?<!\S)(?:--flash|-f)(?=\s|$)", "", raw)
        return re.sub(r"\s+", " ", raw).strip()

    def _oa_flash_arg(self, parser: Any | None) -> bool:
        if parser is None:
            return False
        with contextlib.suppress(Exception):
            if bool(parser.get_flag("flash")) or bool(parser.get_flag("f")):
                return True
        raw = str(getattr(parser, "raw_args", "") or "")
        return bool(re.search(r"(?<!\S)(?:--flash|-f)(?=\s|$)", raw))

    def _oa_new_chat_arg(self, parser: Any | None) -> tuple[bool, str]:
        if parser is None:
            return False, ""
        raw = str(getattr(parser, "raw_args", "") or "")
        match = re.search(
            r"(?<!\S)--new(?:=(?:\{[^}]*\}|\"[^\"]*\"|'[^']*'|\S*))?(?=\s|$)", raw
        )
        if not match:
            return False, ""
        token = match.group(0)
        if "=" not in token:
            return True, ""
        name = token.split("=", 1)[1].strip()
        if len(name) >= 2 and (
            (name[0] == name[-1] and name[0] in {'"', "'"})
            or (name[0] == "{" and name[-1] == "}")
        ):
            name = name[1:-1]
        return True, name.strip()[:64]

    def _oa_test_name(self, parser: Any | None) -> str:
        if parser is None or not hasattr(parser, "get_kwarg"):
            return ""
        return str(parser.get_kwarg("test", "") or "").strip().lower()

    async def _run_oa_test(self, event: Event, name: str) -> None:
        """Run internal OpenAgent smoke tests without hitting real provider APIs."""
        name = (name or "").strip().lower()
        old_once = self._ask_provider_once
        old_show = self._show_agent_action
        old_sleep = asyncio.sleep
        calls: list[int] = []
        statuses: list[str] = []
        log: list[str] = []

        async def no_sleep(_delay: float) -> None:
            return None

        async def fake_show(
            _event: Any,
            title: str,
            value: str,
            _log: list[str],
            tool_name: str = "",
            **_kwargs: Any,
        ) -> None:
            statuses.append(f"{title}:{tool_name}:{value}")

        try:
            asyncio.sleep = no_sleep
            self._show_agent_action = fake_show  # type: ignore[method-assign]
            if name == "reconnect":

                async def fake_once(
                    _provider: str,
                    _messages: list[dict[str, Any]],
                    _api_key: str,
                    *,
                    max_tokens_override: int | None = None,
                ) -> str:
                    calls.append(1)
                    if len(calls) <= 5:
                        raise RuntimeError("Provider request timed out after 1s")
                    return "ok"

                self._ask_provider_once = fake_once  # type: ignore[method-assign]
                result = await self._ask_provider_with_reconnect(
                    "openai",
                    [],
                    "test-key",
                    status_event=event,
                    agent_log=log,
                    started_at=time.monotonic(),
                    thinking_notes=[],
                )
                text = (
                    "Reconnect test OK\n"
                    f"result={result}\n"
                    f"calls={len(calls)}\n"
                    f"statuses={len(statuses)}\n"
                    f"log={', '.join(log)}"
                )
            elif name == "timeout_provider":
                max_reconnects = max(
                    0,
                    min(int(self.config.get("provider_reconnect_attempts", 5) or 0), 5),
                )

                async def fake_once_timeout(
                    _provider: str,
                    _messages: list[dict[str, Any]],
                    _api_key: str,
                    *,
                    max_tokens_override: int | None = None,
                ) -> str:
                    calls.append(1)
                    raise RuntimeError("Provider request timed out after 1s")

                self._ask_provider_once = fake_once_timeout  # type: ignore[method-assign]
                try:
                    await self._ask_provider_with_reconnect(
                        "openai",
                        [],
                        "test-key",
                        status_event=event,
                        agent_log=log,
                        started_at=time.monotonic(),
                        thinking_notes=[],
                    )
                except Exception as exc:
                    text = (
                        "Timeout provider test OK\n"
                        f"max_reconnects={max_reconnects}\n"
                        f"calls={len(calls)}\n"
                        f"statuses={len(statuses)}\n"
                        f"error={type(exc).__name__}: {exc}\n"
                        f"log={', '.join(log)}"
                    )
                else:
                    text = "Timeout provider test FAILED: expected timeout"
            else:
                text = f"Unknown OpenAgent test: {name}"
        finally:
            self._ask_provider_once = old_once  # type: ignore[method-assign]
            self._show_agent_action = old_show  # type: ignore[method-assign]
            asyncio.sleep = old_sleep
        await self.edit(event, html.escape(text), as_html=True)

    def _config_export_blocked_keys(self) -> set[str]:
        return {"api_key", "provider", "model", "custom_base_url"}

    def _exportable_config(self) -> dict[str, Any]:
        blocked = self._config_export_blocked_keys()
        data = self.config.to_dict()
        return {
            key: value
            for key, value in data.items()
            if key not in blocked and value is not None
        }

    async def _read_import_payload(self, event: Event) -> str:
        raw = self._args_raw(event)
        if raw.strip():
            payload = raw.strip()
            if not payload.startswith("{"):
                raise ValueError(
                    "Pass a JSON object after .oaimport or reply to openagent-settings.json"
                )
            return payload
        reply = await event.get_reply_message()
        if not reply:
            return ""
        file_name = getattr(getattr(reply, "file", None), "name", None) or ""
        if file_name.lower().endswith(".json"):
            data = await reply.download_media(file=bytes)
            if data:
                payload = data.decode("utf-8", errors="replace").strip()
                if payload.startswith("{"):
                    return payload
                raise ValueError("Replied .json file does not contain a JSON object")
        text = getattr(reply, "raw_text", None) or getattr(reply, "text", None) or ""
        if text.strip():
            payload = text.strip()
            if payload.startswith("{"):
                return payload
            raise ValueError(
                "Replied message is not OpenAgent settings JSON. Reply to openagent-settings.json or JSON text."
            )
        data = await reply.download_media(file=bytes)
        if data:
            payload = data.decode("utf-8", errors="replace").strip()
            if payload.startswith("{"):
                return payload
            raise ValueError("Replied file does not contain a JSON object")
        return ""

    def _parse_import_config(self, payload: str) -> dict[str, Any]:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid OpenAgent settings JSON: {exc.msg}") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON object expected")
        settings = data.get("settings", data)
        if not isinstance(settings, dict):
            raise ValueError("settings object expected")
        return settings

    async def _apply_import_config(
        self, settings: dict[str, Any]
    ) -> tuple[list[str], list[str], list[str]]:
        blocked = self._config_export_blocked_keys()
        known = set(self.config.keys())
        applied: list[str] = []
        skipped: list[str] = []
        failed: list[str] = []
        for key, value in settings.items():
            key = str(key)
            if key in blocked or key not in known:
                skipped.append(key)
                continue
            try:
                self.config[key] = value
                applied.append(key)
            except Exception as exc:
                failed.append(f"{key}: {exc}")
        if applied:
            for key in applied:
                self._invalidate_config_caches(key)
            await self.save_config()
        return applied, skipped, failed

    @staticmethod
    def _rich_text_html(text: str, *, limit: int = 30000) -> str:
        text = str(text or "")
        if len(text) > limit:
            text = text[:limit] + "\n… [truncated]"
        escaped = html.escape(text)
        paragraphs = []
        for part in re.split(r"\n{2,}", escaped):
            part = part.strip()
            if part:
                paragraphs.append(f"<p>{part.replace(chr(10), '<br>')}</p>")
        return "".join(paragraphs) or "<p></p>"

    def _rich_bot_system_prompt(self, prompt: str) -> str:
        return (
            self._system_prompt(prompt) + "\n\n## Bot command final answer format\n"
            "For this bot command, the final answer is sent as Telegram Rich Message HTML. "
            "Use BlockRich/Rich HTML block formatting directly in the final answer: "
            '<p>, <blockquote>, <pre><code class="language-python">, <details><summary>, '
            "<ul>/<ol>/<li>, <table>/<caption>/<tr>/<th>/<td>, <footer>, <tg-math>, "
            "<tg-math-block>, <tg-emoji>, <tg-reference>, <tg-time>, and media block tags when useful. "
            "Return only the answer body. Do not wrap it in Markdown fences. "
            "The earlier no-XML rule applies only to tool-call syntax; final Rich HTML tags are allowed here."
        )

    @bot_command(
        "oa",
        doc_ru="<запрос> спросить OpenAgent через rich draft streaming",
        doc_en="<prompt> ask OpenAgent using rich draft streaming",
    )
    async def bot_oa(self, event: Event) -> None:
        if event.sender_id != self.kernel.ADMIN_ID:
            return None

        prompt = self.args_raw(event).strip()
        if not prompt:
            await event.reply("Usage: oa <prompt>")
            return

        bot = self.subinline.bot
        if bot is None or not hasattr(bot, "send_draft_message"):
            await event.reply("Rich draft bot client is unavailable")
            return

        target = getattr(event, "chat_id", None) or getattr(event, "sender_id", None)
        if target is None:
            await event.reply("Can't resolve target chat for rich draft")
            return

        draft_id = int.from_bytes(uuid.uuid4().bytes[:8], "big", signed=True)
        started = time.monotonic()

        async def push_draft(label: str) -> None:
            safe_label = html.escape(label)
            with contextlib.suppress(Exception):
                await bot.send_draft_message(
                    target,
                    html=f"<tg-thinking>{safe_label}</tg-thinking>",
                    draft_id=draft_id,
                    noautolink=True,
                )

        await push_draft("OpenAgent думает…")
        task = asyncio.create_task(
            self._ask_agent(
                prompt,
                status_event=None,
                source_event=event,
                attachments=[],
                started_at=started,
                system_override=self._rich_bot_system_prompt(prompt),
            )
        )
        task_id = f"bot_oa:{draft_id}"
        self._background_tool_tasks[task_id] = task

        tick = 0
        try:
            while not task.done():
                await asyncio.sleep(1.5)
                tick += 1
                elapsed = time.monotonic() - started
                await push_draft(f"OpenAgent генерирует ответ… {elapsed:.1f}s")

            answer, agent_log, thinking_notes, tool_trace = await task
            elapsed = time.monotonic() - started
            self._remember_context(
                getattr(event, "chat_id", None),
                prompt,
                answer,
                tool_trace,
                thinking_notes,
            )
            final_html = answer.strip() if answer.strip() else "<p></p>"
            if "<" not in final_html or ">" not in final_html:
                final_html = self._rich_text_html(final_html)
            await bot.send_rich_message(
                target,
                html=final_html,
                message=answer[:4096] if answer else "",
            )
        except Exception as exc:
            await push_draft("OpenAgent словил ошибку")
            error_html = (
                "<p><b>OpenAgent error</b></p>"
                f"<blockquote><code>{html.escape(str(exc))}</code></blockquote>"
            )
            with contextlib.suppress(Exception):
                if bot is not None and hasattr(bot, "send_rich_message"):
                    await bot.send_rich_message(target, html=error_html, fallback=True)
                    return
            await event.reply(f"OpenAgent error: {exc}")
        finally:
            self._background_tool_tasks.pop(task_id, None)
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    @command(
        "oa",
        alias=["agent"],
        doc_ru="<запрос> спросить ИИ агента; --flash/-f быстрый режим; --new[=имя] новый чат; --chats меню; --clear очистить",
        doc_en="<prompt> ask AI agent; --flash/-f fast mode; --new[=name] new chat; --chats menu; --clear clear",
    )
    async def cmd_oa(self, event: Event) -> None:
        parser = self._oa_arg_parser(event)
        prompt = (
            self._oa_prompt_from_parser(parser)
            if parser is not None
            else self._args_raw(event)
        )
        new_chat, new_chat_name = self._oa_new_chat_arg(parser)
        test_name = self._oa_test_name(parser)
        flash_mode = self._oa_flash_arg(parser)
        if test_name:
            await self._run_oa_test(event, test_name)
            return
        if prompt.strip() == "--clear" or (
            parser is not None and parser.get_flag("clear")
        ):
            chat_id = getattr(event, "chat_id", None)
            if chat_id is not None:
                session = self._get_active_session(int(chat_id))
                session.messages.clear()
                self._tool_memory.pop(int(chat_id), None)
                self._touch_session(session)
                await self.edit(
                    event, html.escape(self.strings("context_cleared")), as_html=True
                )
            else:
                await self.edit(event, self.strings("need_text"))
            return
        if prompt.strip() == "--chats" or (
            parser is not None and parser.get_flag("chats")
        ):
            chat_id = getattr(event, "chat_id", None)
            if chat_id is not None:
                await self._show_sessions_panel(event, int(chat_id), force_inline=True)
            else:
                await self.edit(event, self.strings("need_text"))
            return
        reply_context, attachments = await self._reply_context(event)
        if not prompt and reply_context:
            prompt = self.strings("reply_analyze_prompt")
        if not prompt:
            chat_id = getattr(event, "chat_id", None)
            if chat_id is not None:
                if new_chat:
                    session = self._new_session(
                        int(chat_id), name=new_chat_name or None
                    )
                    self.session_manager.set_preference(int(chat_id), "continue")
                    await self._show_sessions_panel(
                        event,
                        int(chat_id),
                        force_inline=True,
                        alert=self.strings("chat_created", name=session.name),
                    )
                    return
                await self._show_sessions_panel(event, int(chat_id), force_inline=True)
            else:
                await self.edit(event, self.strings("need_text"))
            return

        full_prompt = prompt
        if reply_context:
            full_prompt += f"\n\nReply context:\n{reply_context}"

        chat_id = getattr(event, "chat_id", None)
        if chat_id is not None:
            if new_chat:
                self._new_session(int(chat_id), name=new_chat_name or None)
                self.session_manager.set_preference(int(chat_id), "continue")
            else:
                pref = self._session_prefs.get(int(chat_id), "ask")
                sessions = self._get_chat_sessions(int(chat_id))
                if pref == "new":
                    self._fresh_session(int(chat_id))
                elif pref == "ask" and len(sessions) > 1:
                    prompt_token = self._store_pending_prompt(
                        int(chat_id),
                        prompt,
                        full_prompt,
                        attachments,
                        source_event=event,
                    )
                    await self._show_oa_choice_panel(event, int(chat_id), prompt_token)
                    return

        cancel_token = str(uuid.uuid4())
        self._set_placeholder_context(event, cancel_token)
        self.log.debug(
            "OA cmd_oa: chat_id=%s prompt_len=%d reply=%s attachments=%d",
            chat_id,
            len(prompt),
            bool(reply_context),
            len(attachments or []),
        )
        loading = await self._start_inline_status(
            event,
            self._thinking_text(),
            self._runtime_control_buttons(cancel_token, event),
        )
        started = time.monotonic()
        self.log.debug(
            "OA cmd_oa: status_event type=%s has_edit=%s has_status_buttons=%s",
            type(loading).__name__,
            hasattr(loading, "edit"),
            hasattr(loading, "_openagent_status_buttons"),
        )
        try:
            answer, agent_log, thinking_notes, tool_trace = await self._ask_agent(
                full_prompt,
                status_event=loading or event,
                source_event=event,
                attachments=attachments,
                cancel_token=cancel_token,
                started_at=started,
                flash_mode=flash_mode,
            )
            self._last_request_at = time.time()
            elapsed = time.monotonic() - started
            self._remember_context(
                getattr(event, "chat_id", None),
                full_prompt,
                answer,
                tool_trace,
                thinking_notes,
            )
            await self._reply_text(
                loading or event,
                answer,
                title=self._response_title(
                    elapsed,
                    tool_count=len(agent_log),
                    thinking_notes=thinking_notes,
                ),
                prompt=prompt,
                agent_log=agent_log,
                thinking_notes=thinking_notes,
                buttons=self._final_buttons(
                    getattr(event, "chat_id", None),
                    prompt,
                    full_prompt,
                    attachments,
                    source_event=event,
                ),
                edit_current=True,
            )
            self._store_last_loading(getattr(event, "chat_id", None), loading)
            self._cleanup_runtime_run(cancel_token)
        except Exception as exc:
            self._cleanup_runtime_run(cancel_token)
            await self._reply_error_answer(
                loading or event,
                exc,
                prompt=prompt,
                full_prompt=full_prompt,
                attachments=attachments,
                source_event=event,
                chat_id=getattr(event, "chat_id", None),
                started_at=started,
                source="OpenAgent",
            )

    @command(
        "oaexport",
        doc_ru="экспорт настроек OpenAgent без секретов",
        doc_en="export OpenAgent settings without secrets",
    )
    async def cmd_oaexport(self, event: Event) -> None:
        payload = {
            "name": "OpenAgent settings",
            "version": 1,
            "blocked_keys": sorted(self._config_export_blocked_keys()),
            "settings": self._exportable_config(),
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        data = io.BytesIO(text.encode("utf-8"))
        data.name = "openagent-settings.json"
        try:
            await self.client.send_file(
                event.chat_id,
                data,
                caption="OpenAgent settings export (without provider/API secrets)",
            )
            with contextlib.suppress(Exception):
                await event.delete()
        except Exception:
            await self.edit(event, f"<pre>{html.escape(text)}</pre>", as_html=True)

    @command(
        "oaimport",
        doc_ru="импорт настроек OpenAgent без секретов из reply/JSON",
        doc_en="import OpenAgent settings without secrets from reply/JSON",
    )
    async def cmd_oaimport(self, event: Event) -> None:
        try:
            payload = await self._read_import_payload(event)
            if not payload:
                await self.edit(
                    event,
                    "Reply to openagent-settings.json or pass JSON after .oaimport",
                )
                return
            settings = self._parse_import_config(payload)
            applied, skipped, failed = await self._apply_import_config(settings)
        except Exception as exc:
            await self.edit(
                event, self.strings("error", error=html.escape(str(exc))), as_html=True
            )
            return
        lines = [
            "OpenAgent settings import complete",
            f"applied: {len(applied)}",
            f"skipped: {len(skipped)}",
            f"failed: {len(failed)}",
        ]
        if skipped:
            lines.append("skipped keys: " + ", ".join(sorted(skipped)[:30]))
        if failed:
            lines.append("failed keys: " + "; ".join(failed[:10]))
        await self.edit(
            event,
            "<blockquote>" + html.escape("\n".join(lines)) + "</blockquote>",
            as_html=True,
        )

    @command(
        "skills", doc_ru="список скиллов OpenAgent", doc_en="list OpenAgent skills"
    )
    async def cmd_skills(self, event: Event) -> None:
        arg = self._args_raw(event)
        if arg in {"-repo", "--repo", "repo"}:
            try:
                text = await self._format_skill_repo_list()
            except Exception as exc:
                await self.edit(
                    event,
                    html.escape(self.strings("error", error=str(exc))),
                    as_html=True,
                )
                return
            await self.edit(event, "<pre>" + html.escape(text) + "</pre>", as_html=True)
            return

        skills = self._list_skills()
        if not skills:
            await self.edit(event, self.strings("skills_empty"))
            return
        lines = []
        for path in skills:
            try:
                text = path.read_text(encoding="utf-8")
                first_line = text.splitlines()[0] if text.splitlines() else ""
                frontmatter_name = re.search(
                    r"^name:\s*(.+)$", text, flags=re.MULTILINE
                )
                frontmatter_description = re.search(
                    r"^description:\s*(.+)$", text, flags=re.MULTILINE
                )
            except Exception:
                first_line = ""
                frontmatter_name = None
                frontmatter_description = None
            name = (
                frontmatter_name.group(1).strip()
                if frontmatter_name
                else self._skill_name_from_path(path)
            )
            title = (
                frontmatter_description.group(1).strip()
                if frontmatter_description
                else (
                    first_line.lstrip("# ").strip()
                    if first_line.startswith("#")
                    else name
                )
            )
            lines.append(f"- {name}: {title}")
        await self.edit(
            event, "<pre>" + html.escape("\n".join(lines)) + "</pre>", as_html=True
        )

    @command(
        "skillinstall",
        alias=["ssinstall"],
        doc_ru="<name> установить OpenAgent skill из repo",
        doc_en="<name> install OpenAgent skill from repo",
    )
    async def cmd_skillinstall(self, event: Event) -> None:
        name = self._args_raw(event)
        if not name:
            await self.edit(event, self.strings("skillinstall_usage"))
            return
        try:
            saved_name = await self._install_repo_skill(name)
        except Exception as exc:
            await self.edit(
                event, html.escape(self.strings("error", error=str(exc))), as_html=True
            )
            return
        await self.edit(
            event,
            self.strings("skill_installed", name=html.escape(saved_name)),
            as_html=True,
        )

    @command(
        "sendss", doc_ru="<name> отправить .md скилл", doc_en="<name> send skill .md"
    )
    async def cmd_sendss(self, event: Event) -> None:
        name = self._args_raw(event)
        if not name:
            await self.edit(event, self.strings("sendss_usage"))
            return
        path = self._find_skill_path(name)
        if not path.exists():
            await self.edit(event, self.strings("skill_not_found"))
            return
        await self.client.send_file(
            event.chat_id,
            str(path),
            caption=f"<b>Skill:</b> <code>{html.escape(self._skill_name_from_path(path))}</code>",
            parse_mode="html",
        )
        try:
            await event.delete()
        except Exception:
            pass

    @command(
        "imss",
        doc_ru="[name] импортировать .md скилл из reply",
        doc_en="[name] import .md skill from reply",
    )
    async def cmd_imss(self, event: Event) -> None:
        reply = await event.get_reply_message()
        if not reply:
            await self.edit(event, self.strings("imss_need_reply"))
            return

        name = self._args_raw(event)
        file_name = getattr(getattr(reply, "file", None), "name", None) or ""
        content = ""
        try:
            data = await reply.download_media(file=bytes)
            if data:
                content = data.decode("utf-8", errors="replace")
        except Exception:
            content = ""

        if not content:
            content = (
                getattr(reply, "raw_text", None) or getattr(reply, "text", "") or ""
            )
        if not content.strip():
            await self.edit(event, self.strings("skill_empty"))
            return

        if not name:
            if file_name.lower().endswith(".md"):
                name = Path(file_name).stem
            else:
                match = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
                name = match.group(1).strip() if match else "skill"

        saved_name = self._save_skill(name, content)
        await self.edit(
            event,
            self.strings("skill_imported", name=html.escape(saved_name)),
            as_html=True,
        )

    @command("delss", doc_ru="<name> удалить скилл", doc_en="<name> delete skill")
    async def cmd_delss(self, event: Event) -> None:
        name = self._args_raw(event)
        if not name:
            await self.edit(event, self.strings("delss_usage"))
            return
        path = self._find_skill_path(name)
        if not path.exists():
            await self.edit(event, self.strings("skill_not_found"))
            return
        path.unlink()
        try:
            if path.name == "SKILL.md" and not any(path.parent.iterdir()):
                path.parent.rmdir()
        except Exception:
            pass
        await self.edit(
            event,
            self.strings(
                "skill_deleted", name=html.escape(self._skill_name_from_path(path))
            ),
            as_html=True,
        )

    def _format_oaplugin_overview(self) -> str:
        installed = self._plugins
        text = self.strings("plugins_enabled_title")
        if not installed:
            text += self.strings("plugins_none_installed")
        else:
            for pname, plugin in sorted(installed.items()):
                display_name = self._plugin_meta_text(plugin, "name", default=pname)
                version = self._plugin_meta_text(plugin, "version", default="?")
                desc = self._plugin_meta_text(
                    plugin,
                    "description",
                    default=self.strings("plugin_no_description"),
                )
                author = self._plugin_meta_text(plugin, "author")
                tools = self._plugin_tool_names(plugin)[:5]
                item_lines = [
                    f"<b>{html.escape(display_name)}</b> <code>v{html.escape(version)}</code>"
                ]
                if display_name.lower() != str(pname).lower():
                    item_lines.append(
                        f"{html.escape(self.strings('plugin_id_label'))}: "
                        f"<code>{html.escape(str(pname))}</code>"
                    )
                if desc:
                    item_lines.append(html.escape(desc))
                if author:
                    item_lines.append(
                        f"{html.escape(self.strings('plugin_author_label'))}: {html.escape(author)}"
                    )
                if tools:
                    tools_text = ", ".join(
                        f"<code>{html.escape(tool)}</code>" for tool in tools
                    )
                    item_lines.append(
                        f"{html.escape(self.strings('plugin_tools_label'))}: {tools_text}"
                    )
                text += "<blockquote>" + "\n".join(item_lines) + "</blockquote>\n"
        text += self.strings("plugins_total", count=len(installed))
        return text

    @command(
        "oaplugin",
        doc_ru="управление плагинами OpenAgent",
        doc_en="manage OpenAgent plugins",
    )
    async def cmd_oaplugin(self, event: Event) -> None:
        """Show plugin manager or install a plugin from replied .py file."""
        if await event.get_reply_message():
            try:
                saved_name = await self._install_plugin_from_reply(event)
            except Exception as exc:
                await self.edit(
                    event,
                    self.strings("plugin_install_failed", error=html.escape(str(exc))),
                    as_html=True,
                )
                return
            await self.edit(
                event,
                self.strings("plugin_installed", name=html.escape(saved_name)),
                as_html=True,
            )
            return

        text = self._format_oaplugin_overview()

        buttons = [
            [
                self.Button.inline(
                    self.strings("plugin_catalog_btn"),
                    self._oaplugin_catalog,
                    args=(0,),
                    style="primary",
                ),
                self.Button.inline(
                    self.strings("plugin_manager_btn"),
                    self._oaplugin_manager,
                    args=(0,),
                    style="primary",
                ),
            ],
            [
                self.Button.inline(
                    self.strings("close_btn"), self._oaplugin_close, style="danger"
                ),
            ],
        ]

        chat_id = getattr(event, "chat_id", None)
        if chat_id:
            try:
                await self.inline(
                    chat_id,
                    text,
                    buttons=buttons,
                    ttl=900,
                    parse_mode="html",
                    reply_to=getattr(event, "reply_to", None),
                )
                await event.delete()
            except Exception:
                await self.edit(event, text, as_html=True)
        else:
            await self.edit(event, text, as_html=True)

    @callback(ttl=900)
    async def _oaplugin_close(self, call: InlineMessage) -> None:
        try:
            await call.delete()
        except Exception:
            await call.answer()

    @callback(ttl=900)
    async def _oaplugin_catalog(self, call: InlineMessage, page: int = 0) -> None:
        """Show available plugins from repo (xheta-style)."""
        plugins = self._plugins_cache
        if not plugins:
            plugins = await self._fetch_repo_plugins()
        if not plugins:
            await call.answer(self.strings("plugin_repo_empty"), alert=True)
            return
        if page < 0 or page >= len(plugins):
            await call.answer()
            return
        m = plugins[page]
        name = self._doc_text(m.get("name", "?"), default="?")
        author = self._doc_text(m.get("author", "?"), default="?")
        version = self._doc_text(m.get("version", "?"), default="?")
        desc = self._doc_text(
            m.get("description", self.strings("plugin_no_description")),
            default=self.strings("plugin_no_description"),
        )
        tools = self._string_list(m.get("tools", []))
        permissions = self._string_list(m.get("permissions", []))
        requirements = self._string_list(m.get("requirements", []))
        fname = m.get("file_name", "")
        plugin_key = self._safe_plugin_name(
            m.get("plugin_name") or fname.replace(".py", "") or name
        )
        installed = plugin_key in self._plugins

        text = (
            f"📦 <b>{html.escape(name)}</b> "
            f"<code>v{html.escape(version)}</code> "
            f"by <code>{html.escape(author)}</code>\n\n"
        )
        text += f"📝 {html.escape(desc)}\n"
        if tools:
            tools_str = ", ".join(f"<code>{html.escape(t)}</code>" for t in tools[:8])
            if len(tools) > 8:
                tools_str += self.strings("plugin_more_tools", count=len(tools) - 8)
            text += f"\n🔧 <b>{html.escape(self.strings('plugin_tools_label'))}:</b> {tools_str}"
        if permissions:
            perms_str = ", ".join(
                f"<code>{html.escape(item)}</code>" for item in permissions
            )
            text += f"\n🔐 <b>{html.escape(self.strings('plugin_permissions_label'))}:</b> {perms_str}"
        if requirements:
            reqs_str = ", ".join(
                f"<code>{html.escape(item)}</code>" for item in requirements
            )
            text += f"\n📦 <b>{html.escape(self.strings('plugin_requirements_label'))}:</b> {reqs_str}"
        text += f"\n\n🔢 {page + 1}/{len(plugins)}"

        buttons = []
        raw_url = m.get("download_url", "")
        if installed:
            buttons.append(
                [
                    self.Button.inline(
                        self.strings("plugin_installed_btn"),
                        self._oaplugin_noop,
                        style="primary",
                    )
                ]
            )
        else:
            buttons.append(
                [
                    self.Button.inline(
                        self.strings("plugin_install_btn"),
                        self._oaplugin_install,
                        args=(fname.replace(".py", ""), page),
                        style="primary",
                    )
                ]
            )
        if raw_url:
            buttons[0].append(self.Button.url(self.strings("plugin_code_btn"), raw_url))

        nav = []
        if page > 0:
            nav.append(
                self.Button.inline(
                    "⬅️", self._oaplugin_catalog, args=(page - 1,), style="primary"
                )
            )
        nav.append(
            self.Button.inline(
                f"📋 {page + 1}/{len(plugins)}", self._oaplugin_noop, style="primary"
            )
        )
        if page < len(plugins) - 1:
            nav.append(
                self.Button.inline(
                    "➡️", self._oaplugin_catalog, args=(page + 1,), style="primary"
                )
            )
        if nav:
            buttons.append(nav)
        buttons.append(
            [
                self.Button.inline(
                    self.strings("back_btn"), self._oaplugin_main, style="primary"
                )
            ]
        )

        try:
            await call.edit(text, buttons=buttons, parse_mode="html")
        except Exception:
            pass

    @callback(ttl=900)
    async def _oaplugin_noop(self, call: InlineMessage) -> None:
        await call.answer()

    @callback(ttl=900)
    async def _oaplugin_main(self, call: InlineMessage) -> None:
        """Return to main plugin page."""
        text = self._format_oaplugin_overview()
        buttons = [
            [
                self.Button.inline(
                    self.strings("plugin_catalog_btn"),
                    self._oaplugin_catalog,
                    args=(0,),
                    style="primary",
                ),
                self.Button.inline(
                    self.strings("plugin_manager_btn"),
                    self._oaplugin_manager,
                    args=(0,),
                    style="primary",
                ),
            ],
            [
                self.Button.inline(
                    self.strings("close_btn"), self._oaplugin_close, style="danger"
                ),
            ],
        ]
        try:
            await call.edit(text, buttons=buttons, parse_mode="html")
        except Exception:
            pass

    @callback(ttl=900)
    async def _oaplugin_install(
        self, call: InlineMessage, name: str, page: int = 0
    ) -> None:
        """Download and install a plugin from repo."""
        await call.answer(self.strings("plugin_installing"), alert=False)
        try:
            saved_name = await self._install_plugin_from_repo(name)
            await call.answer(
                self.strings("plugin_installed_alert", name=saved_name), alert=True
            )
        except Exception as exc:
            await call.answer(self.strings("generic_error", error=str(exc)), alert=True)
            return
        plugins = self._plugins_cache
        if plugins and page < len(plugins):
            await self._oaplugin_catalog(call, page)
        else:
            await self._oaplugin_catalog(call, 0)

    @callback(ttl=900)
    async def _oaplugin_manager(self, call: InlineMessage, page: int = 0) -> None:
        """Show installed plugins with delete option."""
        installed = list(self._plugins.items())
        if not installed:
            await call.answer(self.strings("plugin_manager_no_installed"), alert=True)
            return
        if page < 0 or page >= len(installed):
            await call.answer()
            return
        plugin_id, plugin = installed[page]
        plugin_id = str(plugin_id or getattr(plugin, "name", "") or "?")
        display_name = self._plugin_meta_text(plugin, "name", default=plugin_id)
        version = self._plugin_meta_text(plugin, "version", default="?")
        desc = self._plugin_meta_text(
            plugin, "description", default=self.strings("plugin_no_description")
        )
        author = self._plugin_meta_text(plugin, "author")
        tools = self._plugin_tool_names(plugin)
        permissions = self._plugin_permissions(plugin)
        requirements = self._plugin_requirements(plugin)

        text = f"<b>⚙️ {html.escape(display_name)}</b>\n"
        if display_name.lower() != plugin_id.lower():
            text += f"{html.escape(self.strings('plugin_id_label'))}: <code>{html.escape(plugin_id)}</code>\n"
        text += f"{html.escape(self.strings('plugin_version_label'))}: <code>{html.escape(version)}</code>\n"
        if author:
            text += f"{html.escape(self.strings('plugin_author_label'))}: {html.escape(author)}\n"
        if desc:
            text += f"\n{html.escape(desc)}\n"
        if tools:
            tools_str = ", ".join(
                f"<code>{html.escape(tool)}</code>" for tool in tools[:8]
            )
            if len(tools) > 8:
                tools_str += self.strings("plugin_more_tools", count=len(tools) - 8)
            text += (
                f"\n{html.escape(self.strings('plugin_tools_label'))}: {tools_str}\n"
            )
        if permissions:
            perms_str = ", ".join(
                f"<code>{html.escape(item)}</code>" for item in permissions
            )
            text += f"{html.escape(self.strings('plugin_permissions_label'))}: {perms_str}\n"
        if requirements:
            reqs_str = ", ".join(
                f"<code>{html.escape(item)}</code>" for item in requirements
            )
            text += f"{html.escape(self.strings('plugin_requirements_label'))}: {reqs_str}\n"
        text += "\n"
        text += self.strings("plugin_actions_title")
        row1 = [
            self.Button.inline(
                self.strings("plugin_delete_btn"),
                self._oaplugin_uninstall,
                args=(plugin_id, page),
                style="danger",
            )
        ]
        buttons = [row1]
        if len(installed) > 1:
            nav = []
            if page > 0:
                nav.append(
                    self.Button.inline(
                        "⬅️", self._oaplugin_manager, args=(page - 1,), style="primary"
                    )
                )
            nav.append(
                self.Button.inline(
                    f"{page + 1}/{len(installed)}", self._oaplugin_noop, style="primary"
                )
            )
            if page < len(installed) - 1:
                nav.append(
                    self.Button.inline(
                        "➡️", self._oaplugin_manager, args=(page + 1,), style="primary"
                    )
                )
            buttons.append(nav)
        buttons.append(
            [
                self.Button.inline(
                    self.strings("back_btn"), self._oaplugin_main, style="primary"
                )
            ]
        )
        try:
            await call.edit(text, buttons=buttons, parse_mode="html")
        except Exception:
            pass

    @callback(ttl=900)
    async def _oaplugin_uninstall(
        self, call: InlineMessage, name: str, page: int = 0
    ) -> None:
        """Delete a plugin."""
        try:
            name = self._safe_plugin_name(name)
            fpath = self._plugin_files.get(name)
            is_builtin = bool(fpath and self._is_builtin_plugin_file(fpath))
            if is_builtin:
                self._disabled_plugins.add(name)
                self._save_disabled_plugins()
            self._unregister_plugin(name)
            plugins_dir = self._resolve_plugins_dir()
            if fpath and fpath.exists() and not is_builtin:
                try:
                    fpath.resolve().relative_to(plugins_dir.resolve())
                    fpath.unlink()
                except ValueError:
                    pass
            if not is_builtin:
                for extra in (
                    plugins_dir / f"{name}.py",
                    plugins_dir / f"{name}_plugin.py",
                ):
                    if extra.exists():
                        extra.unlink()
            await call.answer(
                self.strings("plugin_deleted_alert", name=name), alert=True
            )
        except Exception as exc:
            await call.answer(self.strings("generic_error", error=str(exc)), alert=True)
            return
        await self._oaplugin_manager(
            call, min(page, len(self._plugins) - 1) if self._plugins else 0
        )
