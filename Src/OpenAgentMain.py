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
    CHAT_PATH,
    DEBUG,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    MAX_DETAIL_CHARS,
    MAX_MODELS_LISTED,
    MAX_REPLY_CHARS,
    MODELS_ARG,
    MODELS_PATH,
    PROGRESS_UPDATE_INTERVAL,
    CUSTOM_EMOJI,
)


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
    )

    _http: aiohttp.ClientSession | None = None
    _active_requests: dict[str, dict[str, Any]] = {}
    _models_cache: tuple[str, ...] = ()

    async def on_load(self) -> None:
        await super().on_load()

        self._active_requests = {}
        self._models_cache = ()

        me = await self.client.get_me()
        if me.premium is False:

            for key, html in CUSTOM_EMOJI.items():
                CUSTOM_EMOJI[key] = tg_emoji_to_link(html)

        # One shared session for all requests, closed in on_unload
        self._http = aiohttp.ClientSession()

    async def on_unload(self) -> None:
        for state in self._active_requests.values():
            state["task"].cancel()
        self._active_requests.clear()

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

    async def _chat(self, prompt: str) -> str:
        """POST a chat completion request and return the reply text."""
        messages: list[dict[str, str]] = []
        system = str(self.config["system_prompt"]).strip()
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        data = await self._request(
            "POST",
            CHAT_PATH,
            {
                "model": str(self.config["model"]),
                "messages": messages,
                "temperature": float(self.config["temperature"]),
            },
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise AgentError(self.strings("err_bad_reply")) from None
        if not isinstance(content, str) or not content.strip():
            raise AgentError(self.strings("err_bad_reply"))
        return content.strip()

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

    def _provider_label(self) -> str:
        raw = str(self.config["base_url"]).strip()
        parsed = urlparse(raw)
        label = parsed.netloc or parsed.path or raw or "provider"
        if len(label) > 40:
            label = label[:39] + "…"
        return label

    def _header(self, state: dict[str, Any], kind: str) -> str:
        elapsed = max(time.perf_counter() - state["start"], 0.0)
        model = html.escape(str(state["model"]))
        provider = html.escape(str(state["provider"]))

        if kind == "running":
            badges = [
                f"<tg-pill style='success'>{model}</tg-pill>",
                f"<tg-pill style='primary'>{provider}</tg-pill>",
            ]
        else:
            if kind == "done":
                style, label = "success", self.strings("status_done")
            elif kind == "error":
                style, label = "danger", self.strings("status_error")
            else:
                style, label = "danger", self.strings("status_cancelled")
            badges = [
                f"<tg-pill style='{style}'>{html.escape(label)}</tg-pill>",
                f"<tg-pill style='primary'>{model}</tg-pill>",
            ]

        badges.append(f"<tg-pill style='primary'>⏱ {elapsed:.2f}s</tg-pill>")
        return f"<p>{' '.join(badges)}</p>"

    def _build_progress_text(self, state: dict[str, Any]) -> str:
        return "\n".join(
            [
                self._header(state, "running"),
                f"<blockquote>{CUSTOM_EMOJI['think']} "
                f"<i>{html.escape(self.strings('thinking'))}</i></blockquote>",
                state["cancel_tag"],
            ]
        )

    def _build_result_text(
        self, state: dict[str, Any], kind: str, text: str = ""
    ) -> str:
        if kind == "cancelled":
            body = (
                f"<blockquote>{html.escape(self.strings('cancelled_body'))}"
                "</blockquote>"
            )
        elif text:
            body = (
                f"<blockquote expandable>{html.escape(self._clip(text))}"
                "</blockquote>"
            )
        else:
            body = (
                f"<blockquote>{html.escape(self.strings('err_bad_reply'))}"
                "</blockquote>"
            )
        return f"{self._header(state, kind)}\n{body}"

    def _build_error_text(self, text: str) -> str:
        model = html.escape(str(self.config["model"]))
        return "\n".join(
            [
                f"<p><tg-pill style='danger'>"
                f"{html.escape(self.strings('status_error'))}</tg-pill> "
                f"<tg-pill style='primary'>{model}</tg-pill></p>",
                f"<blockquote expandable>{html.escape(self._clip(text))}"
                "</blockquote>",
            ]
        )

    def _guest_mention(self) -> str:
        username = (
            str(self.kernel.config.get("inline_bot_username") or "").strip().lstrip("@")
        )
        return f"@{username}" if username else "@bot"

    def _build_usage_text(self) -> str:
        mention = self._guest_mention()
        model = html.escape(str(self.config["model"]))
        command = html.escape(f"{mention} agent ", quote=True)
        label = html.escape(self.strings("btn_copy_prompt"), quote=True)
        return "\n".join(
            [
                f"<p><tg-pill style='success'>OpenAgent</tg-pill> <tg-pill style='primary'>{model}</tg-pill></p>",
                f"<blockquote>{html.escape(self.strings('usage', mention=mention))}"
                "</blockquote>",
                "<tg-button-row align='left'>"
                f"<tg-button type='copy_text' text='{command}' style='primary'>"
                f"{label}</tg-button></tg-button-row>",
            ]
        )

    def _render_models(self, models: tuple[str, ...] | list[str]) -> str:
        current = str(self.config["model"])
        header = (
            f"<p>{CUSTOM_EMOJI['bals']} "
            f"<b>{html.escape(self.strings('models_title'))}</b> "
            f"<tg-pill style='success'>{html.escape(current)}</tg-pill></p>"
        )
        if not models:
            return (
                f"{header}\n<blockquote>"
                f"{html.escape(self.strings('models_empty'))}</blockquote>"
            )

        rows = []
        for model in models:
            selected = model == current
            style = "success" if selected else "primary"
            marker = CUSTOM_EMOJI["ok"] if selected else "•"
            button = self.Button.rich.inline(
                model,
                self.on_change_model,
                data=model,
                style=style,
                html_tag=True,
            )
            rows.append(f"{marker} {button}")
        return "\n".join(
            [header, "<blockquote expandable>", "<br>".join(rows), "</blockquote>"]
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
        doc_ru="<запрос> | models - спросить LLM или показать модели",
        doc_en="<prompt> | models - ask the LLM or list models",
    )
    async def guest_cmd_agent(self, event: typ.Event) -> None:
        raw = event.guest_args
        if not raw:
            await event.reply_rich(self._build_usage_text())
            return

        if raw == MODELS_ARG:
            await event.reply_rich(await self._build_models_text())
            return

        start = time.perf_counter()
        request_id = uuid.uuid4().hex
        task = asyncio.create_task(self._chat(raw))
        ttl = int(self.config["timeout"]) + 60
        state: dict[str, Any] = {
            "task": task,
            "start": start,
            "model": str(self.config["model"]),
            "provider": self._provider_label(),
            "cancelled": False,
            "ui_alive": True,
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
                "<tg-button-row>" f"{cancel_button}</tg-button-row>"
            )
            self._active_requests[request_id] = state
            message = await event.reply_rich(self._build_progress_text(state))
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
                message, self._build_result_text(state, "error", str(exc))
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
            await self._edit_rich(message, self._build_result_text(state, "done", text))
        finally:
            self._active_requests.pop(request_id, None)
            if not task.done():
                task.cancel()
