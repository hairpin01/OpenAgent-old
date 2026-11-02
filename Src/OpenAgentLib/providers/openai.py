"""OpenAI-compatible provider: chat/completions + models listing."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import aiohttp

from Settings import (
    CHAT_PATH,
    DEBUG,
    MAX_DETAIL_CHARS,
    MAX_MODELS_LISTED,
    MAX_PROVIDER_ERROR_CHARS,
    MODELS_PATH,
)
from ..harness import AgentError


def _one_line(text: object) -> str:
    """Flatten ``text`` to one line (for pill-sized error summaries)."""
    return " ".join(str(text).split())


def error_detail(body: Any, raw: str = "") -> str:
    """Extract the provider error text from a response body.

    JSON bodies take the message from ``error.message`` / ``error`` /
    ``message``; anything else falls back to the raw response text, so
    non-JSON error pages are not lost either.
    """
    text = ""
    if isinstance(body, dict):
        err = body.get("error", body.get("message", ""))
        if isinstance(err, dict):
            text = str(err.get("message") or err.get("detail") or err.get("details") or "")
        elif err:
            text = str(err)
    elif isinstance(body, str) and body:
        text = body
    if not text:
        text = _one_line(raw)
    text = text.strip()
    return text[:MAX_PROVIDER_ERROR_CHARS]


class OpenAIProvider:
    """Speaks the OpenAI wire format (chat/completions, models)."""

    def __init__(
        self,
        http: aiohttp.ClientSession | None,
        config: Any,  # ModuleConfig: read live, may change at runtime
        strings: Callable[..., str],
        log: Any,
    ) -> None:
        self._http = http
        self._config = config
        self._strings = strings
        self._log = log

    async def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Any:
        """Send a GET or POST request to the provider and return JSON."""
        api_key = str(self._config["api_key"]).strip()
        if not api_key:
            raise AgentError(self._strings("err_no_key"))
        if self._http is None:
            raise AgentError(self._strings("err_network", error="no session"))

        url = str(self._config["base_url"]).rstrip("/") + path
        headers = {"Authorization": f"Bearer {api_key}"}
        timeout = aiohttp.ClientTimeout(total=int(self._config["timeout"]))
        if DEBUG:
            self._log.debug(f"{method} {url}")

        try:
            async with self._http.request(
                method, url, json=payload, headers=headers, timeout=timeout
            ) as resp:
                status = resp.status
                # Read as text first: providers may answer with a non-JSON
                # error page, and the text must not be lost on json failure
                raw = (await resp.read()).decode("utf-8", "replace")
                try:
                    body: Any = json.loads(raw)
                except ValueError:
                    body = None
        except asyncio.TimeoutError:
            raise AgentError(self._strings("err_timeout")) from None
        except (aiohttp.ClientError, ValueError) as exc:
            raise AgentError(self._strings("err_network", error=str(exc))) from None

        if status >= 400:
            detail = error_detail(body, raw)
            short = _one_line(detail)
            if len(short) > MAX_DETAIL_CHARS:
                short = short[:MAX_DETAIL_CHARS].rstrip() + "…"
            if short:
                text = self._strings("err_http", status=status, detail=short)
            else:
                text = self._strings("err_http", status=status, detail="").rstrip(": ")
            raise AgentError(text, detail=detail or None)
        return body

    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """One chat completion; returns (assistant message, usage)."""
        payload: dict[str, Any] = {
            "model": str(self._config["model"]),
            "messages": messages,
            "temperature": float(self._config["temperature"]),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        data = await self._request("POST", CHAT_PATH, payload)
        # Some providers wrap errors in a 200 body
        if (
            isinstance(data, dict)
            and "choices" not in data
            and (data.get("error") or data.get("message"))
        ):
            detail = error_detail(data, "")
            raise AgentError(self._strings("err_bad_reply"), detail=detail or None)
        try:
            choice = data["choices"][0]
            message = choice.get("message")
        except (KeyError, IndexError, TypeError):
            raise AgentError(self._strings("err_bad_reply")) from None
        if not isinstance(message, dict):
            raise AgentError(self._strings("err_bad_reply"))
        usage = data.get("usage") if isinstance(data, dict) else None
        return message, usage if isinstance(usage, dict) else None

    async def models(self) -> list[str]:
        """GET the model list and return it as plain ids."""
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
