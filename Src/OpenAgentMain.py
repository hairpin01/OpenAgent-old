from __future__ import annotations

# Standard library
import ast
import asyncio
import html
import time
import uuid
import json
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
    CHAT_PATH,
    DEBUG,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_TOOL_TIMEOUT,
    MAX_DETAIL_CHARS,
    MAX_MODELS_LISTED,
    MAX_REPLY_CHARS,
    MAX_TOOL_ITERATIONS,
    MODELS_ARG,
    MODELS_PATH,
    PROGRESS_UPDATE_INTERVAL,
    CUSTOM_EMOJI,
    RICH_HTML_RULES,
    DEFAULT_CONTEXT_ENABLED,
    DEFAULT_CONTEXT_SIZE,
    MAX_CONTEXT_MESSAGES,
    CONTEXT_RESET_ARGS,
)

# Tools
from .OpenAgentLib.Tools import (
    Tool,
    ToolContext,
    call_tool,
    load_tools,
    tool_schemas,
)
# Text rendering
from .OpenAgentLib.Format.TextRender import RenderContext, render as render_text


class AgentError(Exception):
    """Provider or configuration error with a ready user-facing text."""


def _error_detail(body: Any) -> str:
    """Extract a short error message from a provider response body."""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return str(err.get("message", ""))[:MAX_DETAIL_CHARS]
        if err:
            return str(err)[:MAX_DETAIL_CHARS]
    return ""


class OpenAgent(loader.ModuleBase):
    name = "OpenAgent"
    version = "1.0.0"
    author = "@hairpin00"
    description: dict[str, str] = {
        "ru": "mini клиент LLM (OpenAI-совместимый API)",
        "en": "mini LLM client (OpenAI compatible API)",
    }

    strings = load_strings()

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
            validator=cfg.Integer(min=1, max=MAX_CONTEXT_MESSAGES),
        ),
    )

    _http: aiohttp.ClientSession | None = None
    _active_requests: dict[str, dict[str, Any]] = {}
    _models_cache: tuple[str, ...] = ()
    _tools: list[Tool] = []


    async def on_load(self) -> None:
        await super().on_load()

        self._active_requests = {}
        self._models_cache = ()
        self._tools = load_tools(self.log)
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

        if self._http is not None:
            await self._http.close()
            self._http = None
        await super().on_unload()


    async def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Any:
        """Send a GET or POST request to the provider and return JSON."""
        api_key = str(self.config["api_key"]).strip()
        if not api_key:
            raise AgentError(self.strings("err_no_key"))
        if self._http is None:
            raise AgentError(self.strings("err_network", error="no session"))

        url = str(self.config["base_url"]).rstrip("/") + path
        headers = {"Authorization": f"Bearer {api_key}"}
        timeout = aiohttp.ClientTimeout(total=int(self.config["timeout"]))
        if DEBUG:
            self.log.debug(f"{method} {url}")

        try:
            async with self._http.request(
                method, url, json=payload, headers=headers, timeout=timeout
            ) as resp:
                body = await resp.json(content_type=None)
                status = resp.status
        except asyncio.TimeoutError:
            raise AgentError(self.strings("err_timeout")) from None
        except (aiohttp.ClientError, ValueError) as exc:
            raise AgentError(self.strings("err_network", error=str(exc))) from None

        if status >= 400:
            raise AgentError(
                self.strings("err_http", status=status, detail=_error_detail(body))
            )
        return body

    def _build_messages(
        self, prompt: str, history: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        system = str(self.config["system_prompt"]).strip()
        if system:
            messages.append({
                "role": "system",
                "content": (
                    f'<rules>{RICH_HTML_RULES}</rules>'
                    + "\nUser system prompt:\n<user-prompt>" + system + "</user-prompt>"
                ),
            })

        messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

    async def _chat_request(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": str(self.config["model"]),
            "messages": messages,
            "temperature": float(self.config["temperature"]),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        data = await self._request("POST", CHAT_PATH, payload)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            raise AgentError(self.strings("err_bad_reply")) from None
        if not isinstance(message, dict):
            raise AgentError(self.strings("err_bad_reply"))
        return message

    async def _chat_with_tools(
        self,
        prompt: str,
        history: list[dict[str, str]],
        state: dict[str, Any],
        message: Any,
        event: typ.Event,
        chat_id: int | str,
    ) -> str:
        messages = self._build_messages(prompt, history)
        tools = tool_schemas(self._tools)
        context = ToolContext(
            logger=self.log,
            http=self._http,
            event=event,
            chat_id=chat_id,
        )

        for step in range(MAX_TOOL_ITERATIONS):
            response = await self._chat_request(messages, tools)
            calls = response.get("tool_calls") or []
            if not calls:
                content = response.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise AgentError(self.strings("err_bad_reply"))
                return content.strip()

            messages.append({
                "role": "assistant",
                "content": response.get("content") or "",
                "tool_calls": calls,
            })

            for index, call in enumerate(calls):
                function = call.get("function") or {}
                name = str(function.get("name") or "")
                raw_arguments = function.get("arguments") or "{}"
                state["tool_calls"] = int(state.get("tool_calls") or 0) + 1

                if isinstance(raw_arguments, str):
                    try:
                        arguments = json.loads(raw_arguments)
                    except (TypeError, ValueError):
                        arguments = None
                else:
                    arguments = raw_arguments

                if not isinstance(arguments, dict):
                    result = json.dumps(
                        {"ok": False, "error": "tool arguments must be a JSON object"},
                        ensure_ascii=False,
                    )
                else:
                    state["tool"] = name
                    await self._edit_rich(message, self._build_progress_text(state))
                    if DEBUG:
                        self.log.debug("tool call %s: %s", name, arguments)
                    result = await call_tool(
                        self._tools,
                        name,
                        arguments,
                        context,
                        timeout=DEFAULT_TOOL_TIMEOUT,
                    )
                    state["tool"] = ""

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id") or f"call_{step}_{index}",
                    "content": result,
                })

        raise AgentError(self.strings("err_tool_limit"))

    async def _models(self) -> list[str]:
        """GET the model list and return it as plain text."""
        data = await self._request("GET", MODELS_PATH)
        items = data.get("data", []) if isinstance(data, dict) else []
        ids = sorted(
            {
                str(item.get("id", ""))
                for item in items
                if isinstance(item, dict) and item.get("id")
            }
        )
        return ids[:MAX_MODELS_LISTED]

    @staticmethod
    def _clip(text: str) -> str:
        text = str(text)
        if len(text) <= MAX_REPLY_CHARS:
            return text
        return text[:MAX_REPLY_CHARS].rstrip() + "…"

    async def _load_context(self, chat_id):
        try:
            raw = await self.db.db_get(self.name.lower(), f"ctx_{chat_id}")
        except Exception as exc:
            self.log.error("OpenAgent context load failed: %s", exc)
            return []
    
        if raw is None:
            return []
            
        if isinstance(raw, str):
            s = raw.strip()
            if not s:
                return []
            try:
                raw = json.loads(s)
            except ValueError:
                try:
                    raw = ast.literal_eval(s)
                except (ValueError, SyntaxError):
                    self.log.error("OpenAgent ctx unparsable: %r", raw[:200])
                    return []
    
        if not isinstance(raw, list):
            return []

        limit = int(self.config["context_size"]) * 2
        out: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content:
                out.append({"role": role, "content": content})
        return out[-limit:]

    async def _save_context(self, chat_id: int | str, history: list[dict[str, str]]) -> None:
        try:
            await self.db.db_set(self.name.lower(), f"ctx_{chat_id}", history)
        except Exception as exc:
            self.log.error("OpenAgent context save failed: %s", exc)

    async def _clear_context(self, chat_id: int | str) -> int:
        history = await self._load_context(chat_id)
        removed = len(history)
        if removed:
            await self._save_context(chat_id, [])
        return removed

    async def _append_context(self, chat_id: int | str, prompt: str, answer: str) -> int:
        history = await self._load_context(chat_id)
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": answer})
        limit = int(self.config["context_size"]) * 2
        if len(history) > limit:
            history = history[-limit:]
        await self._save_context(chat_id, history)
        return len(history) // 2

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
        return render_text(
            self,
            fmt,
            RenderContext(kind=kind, text=text, copy_text=text, state=state, extra={"message": message}),
        )

    def _build_error_text(self, text: str) -> str:
        return render_text(
            self,
            "error_standalone",
            RenderContext(kind="error", text=text),
        )

    def _build_usage_text(self) -> str:
        mention = agent._guest_mention()
        command = html.escape(f"{mention} agent ", quote=True)
        return render_text(self, "usage", RenderContext(copy_text=command, label=self.strings("btn_copy_prompt")))

    def _render_models(self, models: tuple[str, ...] | list[str]) -> str:
        return render_text(
            self,
            "models",
            RenderContext(extra={"models": list(models)}),
        )

    async def _build_models_text(self) -> str:
        try:
            list_models = await self._models()
        except AgentError as exc:
            return self._build_error_text(str(exc))
        except Exception as exc:
            return self._build_error_text(
                self.strings("err_unexpected", error=str(exc))
            )

        self._models_cache = tuple(list_models)
        return self._render_models(list_models)


    async def _edit_rich(self, message: Any, text: str) -> None:
        try:
            await message.edit_rich(text)
        except Exception as exc:
            self.log.debug("OpenAgent message update failed: %s", exc)

    async def _wait_for_reply(
        self, task: asyncio.Task[str], message: Any, state: dict[str, Any]
    ) -> str:
        """Wait for the provider while refreshing the elapsed-time header."""
        while True:
            timeout = PROGRESS_UPDATE_INTERVAL if state["ui_alive"] else None
            done, _ = await asyncio.wait({task}, timeout=timeout)
            if done:
                return task.result()

            try:
                await message.edit_rich(self._build_progress_text(state))
            except Exception as exc:
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
        removed = await self._clear_context(chat_id)
        text = (
            self.strings("context_cleared", count=removed)
            if removed
            else self.strings("context_empty")
        )
        return self._build_notice_text(text)


    @loader.callback()
    async def on_cancel_request(self, call: typ.InlineMessage, data=None) -> None:
        state = self._active_requests.get(data) if data else None
        if state is None or state["task"].done():
            await call.answer(self.strings("already_finished"))
            return
        state["cancelled"] = True
        state["task"].cancel()
        await call.answer(self.strings("cancelling"))

    @loader.callback()
    async def on_change_model(self, call: typ.InlineMessage, data=None) -> None:
        if not data:
            return

        model = str(data)
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
        if self._models_cache:
            await self._edit_rich(call, self._render_models(self._models_cache))

    @loader.guest_command(
        "agent",
        doc_ru="<запрос> | models | reset - спросить LLM, показать модели, очистить контекст",
        doc_en="<prompt> | models | reset - ask the LLM, list models, clear context",
    )
    async def guest_cmd_agent(self, event: typ.Event) -> None:
        raw = event.guest_args
        if not raw:
            await event.reply_rich(self._build_usage_text())
            return

        if raw == MODELS_ARG:
            await event.reply_rich(await self._build_models_text())
            return

        if raw.strip().lower() in CONTEXT_RESET_ARGS:
            await event.reply_rich(await self._handle_context_reset(event))
            return

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
            state["cancel_tag"] = (
                f"<tg-button-row>{cancel_button}</tg-button-row>"
            )

            message = await event.reply_rich(self._build_progress_text(state))

            chat_id = "global"
            context_enabled = (
                chat_id is not None and bool(self.config["context_enabled"])
            )

            history = await self._load_context(chat_id) if context_enabled else []
            task = asyncio.create_task(
                self._chat_with_tools(raw, history, state, message, event, chat_id)
            )
            state['context_count'] = len(history) // 2
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
                    prompt=raw,
                ),
            )
        else:
            if context_enabled:
                state["context_count"] = await self._append_context(
                    chat_id, raw, text
                )
            await self._edit_rich(message, self._build_result_text(state, "done", text, message=message))
        finally:
            self._active_requests.pop(request_id, None)
            if not task.done():
                task.cancel()
