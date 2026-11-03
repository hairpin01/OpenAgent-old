from __future__ import annotations

# Standard library
import asyncio
import html
import time
import uuid
from typing import Any
from urllib.parse import urlparse

# Third-party dependencies
import aiohttp

# SDK
import core.lib.loader.module_base as loader
import core.lib.loader.module_config as cfg
import core.lib.types as typ
from cubkit import load_strings
from utils.tg_emoji import tg_emoji_to_link

# Build config
from .Settings import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_STREAM,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_TOOL_RULES_ENABLED,
    DEFAULT_TOOL_TIMEOUT,
    MAX_DETAIL_CHARS,
    MAX_MODELS_LISTED,
    MAX_REPLY_CHARS,
    MODELS_ARG,
    MODELS_PAGE_SIZE,
    PROGRESS_UPDATE_INTERVAL,
    STREAM_UPDATE_INTERVAL,
    CUSTOM_EMOJI,
    DEFAULT_CONTEXT_ENABLED,
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_CONTEXT_WINDOW,
    MAX_CONTEXT_MESSAGES,
    CONTEXT_RESET_ARGS,
    CHAT_ID,
)

# Tools
from .OpenAgentLib.Tools import Tool, ToolContext, load_tools

# Harness: the agent loop and the provider (UI-agnostic)
from .OpenAgentLib.harness import AgentError, RunInput, run as harness_run
from .OpenAgentLib.providers import OpenAIProvider

# Chat context: DB history store + provider message building
from .OpenAgentLib.context import ContextStore, build_messages, estimate_tokens

# Text rendering
from .OpenAgentLib.Format.TextRender import RenderContext, render as render_text


class OpenAgent(loader.ModuleBase):
    name = "OpenAgent"
    version = "1.0.0"
    author = "@hairpin00"
    description: dict[str, str] = {
        "ru": "mini клиент LLM (OpenAI-совместимый API)",
        "en": "mini LLM client (OpenAI compatible API)",
    }

    strings = load_strings()

    # User config
    config = cfg.ModuleConfig(
        cfg.ConfigValue(
            "api_key",
            "",
            description=lambda self: self.strings("cfg_api_key"),
            validator=cfg.Secret(),
        ),
        cfg.ConfigValue(
            "model",
            DEFAULT_MODEL,
            description=lambda self: self.strings("cfg_model"),
            validator=cfg.String(),
        ),
        cfg.ConfigValue(
            "base_url",
            DEFAULT_BASE_URL,
            description=lambda self: self.strings("cfg_base_url"),
            validator=cfg.Link(),
        ),
        cfg.ConfigValue(
            "temperature",
            DEFAULT_TEMPERATURE,
            description=lambda self: self.strings("cfg_temperature"),
            validator=cfg.Float(min=0.0, max=2.0),
        ),
        cfg.ConfigValue(
            "timeout",
            DEFAULT_TIMEOUT,
            description=lambda self: self.strings("cfg_timeout"),
            validator=cfg.Integer(min=1, max=600),
        ),
        cfg.ConfigValue(
            "system_prompt",
            DEFAULT_SYSTEM_PROMPT,
            description=lambda self: self.strings("cfg_system_prompt"),
            validator=cfg.String(),
        ),
        cfg.ConfigValue(
            "context_enabled",
            DEFAULT_CONTEXT_ENABLED,
            description=lambda self: self.strings("cfg_context_enabled"),
            validator=cfg.Boolean(),
        ),
        cfg.ConfigValue(
            "context_size",
            DEFAULT_CONTEXT_SIZE,
            description=lambda self: self.strings("cfg_context_size"),
            validator=cfg.Integer(min=0, max=MAX_CONTEXT_MESSAGES),
        ),
        cfg.ConfigValue(
            "context_window",
            DEFAULT_CONTEXT_WINDOW,
            description=lambda self: self.strings("cfg_context_window"),
            validator=cfg.Integer(min=0, max=100_000_000),
        ),
        cfg.ConfigValue(
            "tool_rules",
            DEFAULT_TOOL_RULES_ENABLED,
            description=lambda self: self.strings("cfg_tool_rules"),
            validator=cfg.Boolean(),
        ),
        cfg.ConfigValue(
            "stream",
            DEFAULT_STREAM,
            description=lambda self: self.strings("cfg_stream"),
            validator=cfg.Boolean(),
        ),
    )

    _http: aiohttp.ClientSession | None = None
    _active_requests: dict[str, dict[str, Any]] = {}
    _models_cache: tuple[str, ...] = ()
    _tools: list[Tool] = []
    _provider: OpenAIProvider | None = None
    _context_store: ContextStore | None = None


    async def on_load(self) -> None:
        await super().on_load()

        self._active_requests = {}
        self._models_cache = ()
        self._tools = load_tools(self.log)
        self._context_store = ContextStore(self.db, self.name.lower(), self.log)
        self.log.debug("OpenAgent tools loaded: %s", len(self._tools))

        me = await self.client.get_me()
        if me.premium is False:
            for key, html_emoji in CUSTOM_EMOJI.items():
                CUSTOM_EMOJI[key] = tg_emoji_to_link(html_emoji)

        # One shared session for all requests, closed in on_unload
        self._http = aiohttp.ClientSession()

    async def on_unload(self) -> None:
        for state in self._active_requests.values():
            state["task"].cancel()
        self._active_requests.clear()
        self._tools = []
        self._provider = None
        self._context_store = None

        if self._http is not None:
            await self._http.close()
            self._http = None
        await super().on_unload()

    @property
    def provider(self) -> OpenAIProvider:
        """The active provider; built once the http session exists."""
        if self._provider is None:
            self._provider = OpenAIProvider(
                self._http, self.config, self.strings, self.log
            )
        return self._provider

    async def _chat_with_tools(
        self,
        prompt: str,
        history: list[dict[str, str]],
        state: dict[str, Any],
        message: Any,
        event: typ.Event,
        chat_id: int | str,
    ) -> str:
        """Thin UI adapter over the provider-agnostic harness loop.

        The harness drives the chat <-> tool steps; this method only
        translates its events into rich-message updates.
        """
        async def on_event(name: str, payload: dict[str, Any]) -> None:
            # Any harness event (tool_start / tool_done) re-renders the
            # progress message with the current state (RUN timer, pills).
            await self._edit_rich(message, self._build_progress_text(state))

        async def on_usage(usage: dict[str, Any]) -> None:
            # Latest provider step's token counts for the token_stats widget
            state["tokens"] = usage
            # What the model has seen so far = size of the session context
            total = int(usage.get("total_tokens") or 0) or (
                int(usage.get("prompt_tokens") or 0)
                + int(usage.get("completion_tokens") or 0)
            )
            if total:
                state["context_tokens"] = total
                state["context_estimated"] = False

        async def on_stream(text: str) -> None:
            # Picked up by the refresh loop (_wait_for_reply); no edit here,
            # so a fast stream cannot flood Telegram with message edits
            state["stream_text"] = text

        tool_context = ToolContext(
            logger=self.log,
            http=self._http,
            event=event,
            chat_id=chat_id,
            client=self.client,
            tools=self._tools,
            state=state,
        )
        messages = build_messages(
            self.config["system_prompt"],
            self.config["tool_rules"],
            self._tools,
            history,
            prompt,
        )
        # Estimate until the provider reports real usage for this request
        state["context_tokens"] = estimate_tokens(messages)
        state["context_estimated"] = True
        return await harness_run(
            RunInput(
                messages=messages,
                tools=self._tools,
                tool_context=tool_context,
                state=state,
                provider=self.provider,
                logger=self.log,
                strings=self.strings,
                on_event=on_event,
                on_usage=on_usage,
                tool_timeout=DEFAULT_TOOL_TIMEOUT,
                on_stream=on_stream if self.config["stream"] else None,
            )
        )

    @staticmethod
    def _clip(text: str) -> str:
        text = str(text)
        if len(text) <= MAX_REPLY_CHARS:
            return text
        return text[:MAX_REPLY_CHARS].rstrip() + "…"

    def _provider_label(self) -> str:
        raw = str(self.config["base_url"]).strip()
        parsed = urlparse(raw)
        label = parsed.netloc or parsed.path or raw or "provider"
        if len(label) > 40:
            label = label[:39] + "…"
        return label

    def _header(self, state: dict[str, Any], kind: str) -> str:
        return render_text(
            self,
            "status_header",
            RenderContext(state=state, kind=kind)
        )

    def _guest_mention(self) -> str:
        username = (
            str(self.kernel.config.get("inline_bot_username") or "").strip().lstrip("@")
        )
        return f"@{username}" if username else "@bot"


    def _build_progress_text(self, state: dict[str, Any]) -> str:
        return render_text(
            self,
            "progress",
            RenderContext(kind="running", state=state),
        )

    def _build_result_text(
        self, state: dict[str, Any], kind: str, text: str = "", message: typ.Message = None
    ) -> str:
        fmt = "cancelled" if kind == "cancelled" else "result"
        # Kept so a tool button can re-render the final message on click
        state["last_result"] = {"kind": kind, "text": text, "message": message}
        return render_text(
            self,
            fmt,
            RenderContext(kind=kind, text=text, copy_text=text, state=state, extra={"message": message}),
        )

    def _build_error_text(self, text: str, detail: str | None = None) -> str:
        state = {"error_detail": detail} if detail else None
        return render_text(
            self,
            "error_standalone",
            RenderContext(kind="error", text=text, state=state),
        )

    def _build_usage_text(self) -> str:
        mention = self._guest_mention()
        command = html.escape(f"{mention} agent ", quote=True)
        return render_text(self, "usage", RenderContext(copy_text=command, label=self.strings("btn_copy_prompt")))

    def _render_models(self, state: dict[str, Any]) -> str:
        """Render one page of the model list; ``state`` keeps the full list."""
        models = list(state.get("models") or [])
        pages = max((len(models) + MODELS_PAGE_SIZE - 1) // MODELS_PAGE_SIZE, 1)
        page = max(0, min(int(state.get("page") or 0), pages - 1))
        state["page"] = page
        start = page * MODELS_PAGE_SIZE
        return render_text(
            self,
            "models",
            RenderContext(
                state=state,
                extra={
                    "models": models[start : start + MODELS_PAGE_SIZE],
                    "page": page,
                    "pages": pages,
                },
            ),
        )

    async def _build_models_text(self) -> str:
        try:
            list_models = await self.provider.models()
        except AgentError as exc:
            return self._build_error_text(str(exc), detail=exc.detail)
        except Exception as exc:
            return self._build_error_text(
                self.strings("err_unexpected", error=str(exc))
            )

        self._models_cache = tuple(list_models)
        return self._render_models({"models": list(list_models), "page": 0})


    async def _edit_rich(self, message: Any, text: str) -> None:
        try:
            await message.edit_rich(text)
        except Exception as exc:
            self.log.debug("OpenAgent message update failed: %s", exc)

    async def _wait_for_reply(
        self, task: asyncio.Task[str], message: Any, state: dict[str, Any]
    ) -> str:
        """Wait for the provider while refreshing the header and the live answer."""
        last_len = -1
        last_edit = 0.0
        while True:
            streaming = bool(state.get("stream_text"))
            interval = STREAM_UPDATE_INTERVAL if streaming else PROGRESS_UPDATE_INTERVAL
            timeout = interval if state["ui_alive"] else None
            done, _ = await asyncio.wait({task}, timeout=timeout)
            if done:
                return task.result()

            if streaming:
                # Nothing new in the answer: the header timer alone can wait
                # for the normal (slower) refresh
                length = len(state["stream_text"])
                now = time.perf_counter()
                if length == last_len and now - last_edit < PROGRESS_UPDATE_INTERVAL:
                    continue
                last_len, last_edit = length, now

            try:
                await message.edit_rich(self._build_progress_text(state))
                state["ui_errors"] = 0
            except Exception as exc:
                # A stream edits often, so one failed edit (flood wait) must
                # not kill the live view; a plain wait gives up at once
                state["ui_errors"] = int(state.get("ui_errors") or 0) + 1
                if not streaming or state["ui_errors"] >= 3:
                    state["ui_alive"] = False
                self.log.debug("OpenAgent progress update failed: %s", exc)


    def _build_notice_text(
        self, text: str, style: str = "primary", label: str | None = None
    ) -> str:
        return render_text(
            self,
            "notice",
            RenderContext(
                kind="notice",
                text=text,
                extra={"label": label or self.strings("status_done"), "style": style},
            ),
        )

    async def _handle_context_reset(self, event: typ.Message) -> str:
        chat_id = "global"
        if chat_id is None:
            return self._build_notice_text(
                self.strings("err_context_no_chat"),
                style="danger",
                label=self.strings("status_error"),
            )
        removed = await self._context_store.clear(chat_id)
        text = (
            self.strings("context_cleared", count=removed)
            if removed
            else self.strings("context_empty")
        )
        return self._build_notice_text(text)


    @loader.callback()
    async def on_cancel_request(self, call: typ.InlineMessage, data=None) -> None:
        if not call.sender_id == self.kernel.ADMIN_ID:
            return

        state = self._active_requests.get(data) if data else None
        if state is None or state["task"].done():
            await call.answer(self.strings("already_finished"))
            return
        state["cancelled"] = True
        state["task"].cancel()
        await call.answer(self.strings("cancelling"))

    async def on_supplement(self, event: typ.InlineMessage, text, data=None) -> None:
        """Queue extra user text for a running request (Supplement input button)."""
        if getattr(event, "user_id", None) != self.kernel.ADMIN_ID:
            return

        text = str(text or "").strip()
        state = self._active_requests.get(data) if data else None
        if not text or state is None or state["task"].done():
            return
        # The harness drains this queue before the next provider step
        state.setdefault("supplements", []).append(text)

    @loader.callback()
    async def on_tool_args(self, call: typ.InlineMessage, data=None) -> None:
        """Toggle the details block (tool, description, args, output, time)."""
        if not call.sender_id == self.kernel.ADMIN_ID:
            return
        if not data:
            return

        state, entry = data
        entry["open"] = not entry.get("open")
        # Drop the cached buttons so they are rebuilt with the new style
        entry.pop("button", None)
        entry.pop("details_button", None)
        await call.answer()

        last = state.get("last_result")
        if last is None:
            # Still running: the next progress refresh renders the block
            return
        await self._edit_rich(
            call,
            self._build_result_text(
                state, last["kind"], last["text"], last["message"]
            ),
        )

    @staticmethod
    def _repeat_prompt(state: dict[str, Any], from_history: bool) -> str:
        """Prompt shown by the current view, used by the Again button."""
        if from_history:
            history = state.get("history") or []
            total = len(history) // 2
            pos = int(state.get("history_pos", total - 1))
            if 0 <= pos < total:
                return str(history[2 * pos]["content"])
        return str(state.get("prompt") or "")

    @loader.callback()
    async def on_repeat_request(self, call: typ.InlineMessage, data=None) -> None:
        """Repeat the prompt of the exchange currently shown in the message."""
        if not call.sender_id == self.kernel.ADMIN_ID:
            return
        if not data:
            return

        state, from_history = data
        prompt = self._repeat_prompt(state, bool(from_history))
        if not prompt:
            await call.answer(self.strings("repeat_empty"))
            return

        await call.answer()
        # Re-run through the normal guest flow, editing the same message
        call.guest_args = prompt
        call.reply_rich = call.edit_rich
        call.reply = call.edit
        try:
            await self.guest_cmd_agent(call)
        except Exception as exc:
            await self.kernel.handle_error(
                exc, message="OpenAgent repeat request failed"
            )

    def _build_history_text(self, state: dict[str, Any]) -> str:
        """Render the exchange at ``state['history_pos']`` (the latest one is the normal result)."""
        history = state["history"]
        pos = state["history_pos"]
        total = len(history) // 2
        last = state.get("last_result")
        if pos == total - 1 and last is not None:
            return self._build_result_text(
                state, last["kind"], last["text"], last["message"]
            )
        return render_text(
            self,
            "history",
            RenderContext(
                kind="done",
                text=history[2 * pos + 1]["content"],
                state=state,
                extra={
                    "prompt": history[2 * pos]["content"],
                    "pos": pos,
                    "total": total,
                },
            ),
        )

    @loader.callback()
    async def on_history_scroll(self, call: typ.InlineMessage, data=None) -> None:
        """Back / Next through the saved exchanges, editing the message in place."""
        if not call.sender_id == self.kernel.ADMIN_ID:
            return
        if not data:
            return

        state, step = data
        total = len(state.get("history") or []) // 2
        pos = int(state.get("history_pos", total - 1)) + step
        if pos < 0:
            await call.answer(self.strings("history_first"))
            return
        if pos >= total:
            await call.answer(self.strings("history_last"))
            return

        state["history_pos"] = pos
        await call.answer()
        await self._edit_rich(call, self._build_history_text(state))

    @loader.callback()
    async def on_change_model(self, call: typ.InlineMessage, data=None) -> None:
        if not call.sender_id == self.kernel.ADMIN_ID:
            return

        if not data:
            return

        if isinstance(data, tuple) and len(data) == 2:
            state, model = data
            model = str(model)
        else:
            state, model = None, str(data)
        old_model = str(self.config["model"])
        self.config["model"] = model
        try:
            await self.kernel.save_module_config(self.name, self.config.to_dict())
        except Exception as exc:
            self.config["model"] = old_model
            self.log.error("OpenAgent config save failed: %s", exc)
            await call.answer(
                self.strings("err_config_save", error=str(exc)[:MAX_DETAIL_CHARS])
            )
            return

        self.kernel.store_module_config_schema(self.name, self.config)
        await call.answer(self.strings("model_changed", model=model))
        if state is not None:
            await self._edit_rich(call, self._render_models(state))
        elif self._models_cache:
            await self._edit_rich(
                call,
                self._render_models({"models": list(self._models_cache), "page": 0}),
            )

    @loader.callback()
    async def on_models_scroll(self, call: typ.InlineMessage, data=None) -> None:
        """Back / Next through the paginated model list."""
        if not call.sender_id == self.kernel.ADMIN_ID:
            return
        if not data:
            return

        state, step = data
        models = state.get("models") or []
        pages = max((len(models) + MODELS_PAGE_SIZE - 1) // MODELS_PAGE_SIZE, 1)
        page = int(state.get("page") or 0) + step
        if page < 0:
            await call.answer(self.strings("models_first"))
            return
        if page >= pages:
            await call.answer(self.strings("models_last"))
            return

        state["page"] = page
        await call.answer()
        await self._edit_rich(call, self._render_models(state))

    @loader.guest_command(
        "agent",
        doc_ru="<запрос> | models | reset - спросить LLM, показать модели, очистить контекст",
        doc_en="<prompt> | models | reset - ask the LLM, list models, clear context",
    )
    async def guest_cmd_agent(self, event: typ.Event) -> None:
        raw = event.guest_args or ""
        r = getattr(event, "message", None) or event
        try:
            reply = await r.get_reply_message()
        except AttributeError:
            reply = None

        msg = None
        if reply is not None:
            msg = reply.message

        file_text_parts: list[str] = []
        if msg is not None:
            # Handle document (file)
            if getattr(msg, "document", None):
                doc = msg.document
                # Only process text-based files (by mime type or extension)
                mime = getattr(doc, "mime_type", "") or ""
                name = getattr(doc, "file_name", "") or "file"
                is_text = mime.startswith("text/") or mime in (
                    "application/json",
                    "application/xml",
                    "application/javascript",
                    "application/x-python-code",
                ) or name.lower().endswith(
                    (".txt", ".md", ".py", ".js", ".json", ".xml", ".html", ".css", ".csv", ".log", ".ini", ".cfg", ".yaml", ".yml", ".toml", ".sh", ".sql")
                )
                if is_text:
                    try:
                        file_bytes = await self.client.download_file(doc, bytes)
                        content = file_bytes.decode("utf-8", errors="replace")
                        file_text_parts.append(
                            f"\n\n--- Attached file: {name} ({len(content)} chars) ---\n{content}\n--- End of {name} ---"
                        )
                    except Exception as exc:
                        self.log.debug("OpenAgent failed to read attached document: %s", exc)
                        file_text_parts.append(f"\n\n[Attached file: {name} - could not read: {exc}]")
                else:
                    file_text_parts.append(f"\n\n[Attached file: {name} (type: {mime or 'unknown'}) - binary file, not processed]")

            # Handle photo (image)
            elif getattr(msg, "photo", None):
                file_text_parts.append("\n\n[Attached image - vision support depends on the model/provider]")

        # Combine prompt with file content
        if file_text_parts:
            raw = (raw + "".join(file_text_parts)).strip()

        if not raw:
            await event.reply_rich(self._build_usage_text())
            return

        if raw == MODELS_ARG:
            await event.reply_rich(await self._build_models_text())
            return

        if raw.strip().lower() in CONTEXT_RESET_ARGS:
            await event.reply_rich(await self._handle_context_reset(event))
            return

        if reply is not None:
            raw = raw + f"\n[reply context]:\n{reply.html_text}"

        start = time.perf_counter()
        request_id = uuid.uuid4().hex
        ttl = int(self.config["timeout"]) + 60
        state: dict[str, Any] = {
            "start": start,
            "model": str(self.config["model"]),
            "provider": self._provider_label(),
            "cancelled": False,
            "ui_alive": True,
            "prompt": raw,
            "tool": "",
            "tool_calls": 0,
            "user_id": getattr(event, "sender_id", None)
            or getattr(getattr(event, "message", None), "sender_id", None),
        }
        try:
            cancel_button = self.Button.rich.inline(
                self.strings("btn_cancel"),
                self.on_cancel_request,
                data=request_id,
                style="danger",
                ttl=ttl,
                html_tag=True,
            )
            # TextRender's cancel_button module joins both into one row
            state["cancel_btn"] = cancel_button
            state["supplement_btn"] = self.Button.rich.input(
                self.strings("btn_supplement"),
                self.on_supplement,
                data=request_id,
                style="primary",
                ttl=ttl,
                html_tag=True,
            )

            message = await event.reply_rich(self._build_progress_text(state))

            chat_id = CHAT_ID
            context_enabled = (
                chat_id is not None and bool(self.config["context_enabled"])
            )

            history = await self._context_store.load(chat_id) if context_enabled else []
            state['context_count'] = len(history) // 2
            if context_enabled:
                # Snapshot for the pager: errors can still scroll back
                state["history"] = history
                state["history_pos"] = max(len(history) // 2 - 1, 0)
            task = asyncio.create_task(
                self._chat_with_tools(raw, history, state, message, event, chat_id)
            )
            state["task"] = task
            self._active_requests[request_id] = state


        except BaseException:
            task.cancel()
            self._active_requests.pop(request_id, None)
            raise

        if message is None:
            task.cancel()
            self._active_requests.pop(request_id, None)
            return

        try:
            text = await self._wait_for_reply(task, message, state)
        except asyncio.CancelledError:
            if not state["cancelled"]:
                raise
            await self._edit_rich(message, self._build_result_text(state, "cancelled"))
        except AgentError as exc:
            state["error_detail"] = exc.detail
            await self._edit_rich(
                message, self._build_result_text(state, "error", str(exc), message=message)
            )
        except Exception as exc:
            await self._edit_rich(
                message,
                self._build_result_text(
                    state,
                    "error",
                    self.strings("err_unexpected", error=str(exc)),
                ),
            )
        else:
            if context_enabled:
                state["context_count"] = await self._context_store.append(
                    chat_id, raw, text, state.get("exchange_tools") or None
                )
                # Snapshot for the history pager: stable even if later requests
                # append to the stored context
                state["history"] = await self._context_store.load(chat_id)
                state["history_pos"] = len(state["history"]) // 2 - 1
            await self._edit_rich(message, self._build_result_text(state, "done", text, message=message))
        finally:
            self._active_requests.pop(request_id, None)
            if not task.done():
                task.cancel()
