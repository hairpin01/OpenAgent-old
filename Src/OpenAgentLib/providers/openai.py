"""OpenAI-compatible provider: chat/completions + models listing."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable

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

    def _prepare(self, path: str) -> tuple[str, dict[str, str]]:
        """Validate the config and return (url, headers) for ``path``."""
        api_key = str(self._config["api_key"]).strip()
        if not api_key:
            raise AgentError(self._strings("err_no_key"))
        if self._http is None:
            raise AgentError(self._strings("err_network", error="no session"))

        url = str(self._config["base_url"]).rstrip("/") + path
        return url, {"Authorization": f"Bearer {api_key}"}

    def _http_error(self, status: int, body: Any, raw: str) -> AgentError:
        """Build the user-facing error for an HTTP status >= 400."""
        detail = error_detail(body, raw)
        short = _one_line(detail)
        if len(short) > MAX_DETAIL_CHARS:
            short = short[:MAX_DETAIL_CHARS].rstrip() + "…"
        if short:
            text = self._strings("err_http", status=status, detail=short)
        else:
            text = self._strings("err_http", status=status, detail="").rstrip(": ")
        return AgentError(text, detail=detail or None)

    async def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Any:
        """Send a GET or POST request to the provider and return JSON."""
        url, headers = self._prepare(path)
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
            raise self._http_error(status, body, raw)
        return body

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """One chat completion; returns (assistant message, usage).

        With ``on_delta`` the answer is requested as a stream and every text
        piece is passed to it as it arrives; the returned message is still
        the complete one, so callers handle both modes the same way.
        """
        payload: dict[str, Any] = {
            "model": str(self._config["model"]),
            "messages": messages,
            "temperature": float(self._config["temperature"]),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        if on_delta is not None:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
            return await self._chat_stream(payload, on_delta)

        data = await self._request("POST", CHAT_PATH, payload)
        return self._parse_completion(data)

    def _parse_completion(
        self, data: Any
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Pick (message, usage) out of a non-streamed completion body."""
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

    async def _chat_stream(
        self, payload: dict[str, Any], on_delta: Callable[[str], Awaitable[None]]
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """POST a streamed completion (SSE) and fold the chunks into a message."""
        url, headers = self._prepare(CHAT_PATH)
        seconds = int(self._config["timeout"])
        # For a stream the timeout is the longest silence between chunks:
        # a long answer may take more than ``seconds`` in total
        timeout = aiohttp.ClientTimeout(
            total=None, sock_connect=seconds, sock_read=seconds
        )
        if DEBUG:
            self._log.debug(f"POST {url} (stream)")

        parts: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        usage: dict[str, Any] | None = None

        try:
            async with self._http.post(
                url, json=payload, headers=headers, timeout=timeout
            ) as resp:
                if resp.status >= 400:
                    raw = (await resp.read()).decode("utf-8", "replace")
                    try:
                        body: Any = json.loads(raw)
                    except ValueError:
                        body = None
                    raise self._http_error(resp.status, body, raw)

                if "text/event-stream" not in resp.headers.get("Content-Type", ""):
                    # The provider ignored stream=true: plain completion body
                    raw = (await resp.read()).decode("utf-8", "replace")
                    try:
                        data: Any = json.loads(raw)
                    except ValueError:
                        raise AgentError(self._strings("err_bad_reply")) from None
                    message, usage = self._parse_completion(data)
                    content = message.get("content")
                    if isinstance(content, str) and content:
                        await on_delta(content)
                    return message, usage

                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", "replace").strip()
                    # Skip blanks, ":" keep-alive comments and other SSE fields
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                    except ValueError:
                        continue
                    if not isinstance(data, dict):
                        continue

                    if data.get("error") and not data.get("choices"):
                        detail = error_detail(data, "")
                        raise AgentError(
                            self._strings("err_bad_reply"), detail=detail or None
                        )
                    if isinstance(data.get("usage"), dict):
                        usage = data["usage"]

                    choices = data.get("choices") or []
                    if not choices or not isinstance(choices[0], dict):
                        continue
                    delta = choices[0].get("delta") or {}
                    if not isinstance(delta, dict):
                        continue

                    text = delta.get("content")
                    if isinstance(text, str) and text:
                        parts.append(text)
                        await on_delta(text)

                    for item in delta.get("tool_calls") or []:
                        if not isinstance(item, dict):
                            continue
                        index = item.get("index")
                        if not isinstance(index, int):
                            # No index: a new id starts a call, otherwise the
                            # chunk continues the last one
                            index = len(calls) if item.get("id") else max(calls, default=0)
                        entry = calls.setdefault(
                            index, {"id": "", "name": "", "arguments": ""}
                        )
                        if item.get("id"):
                            entry["id"] = str(item["id"])
                        function = item.get("function") or {}
                        if function.get("name") and not entry["name"]:
                            entry["name"] = str(function["name"])
                        if isinstance(function.get("arguments"), str):
                            entry["arguments"] += function["arguments"]
        except asyncio.TimeoutError:
            raise AgentError(self._strings("err_timeout")) from None
        except (aiohttp.ClientError, ValueError) as exc:
            raise AgentError(self._strings("err_network", error=str(exc))) from None

        message: dict[str, Any] = {"role": "assistant", "content": "".join(parts)}
        if calls:
            message["tool_calls"] = [
                {
                    "id": entry["id"] or f"call_{index}",
                    "type": "function",
                    "function": {
                        "name": entry["name"],
                        "arguments": entry["arguments"] or "{}",
                    },
                }
                for index, entry in sorted(calls.items())
            ]
        return message, usage

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
