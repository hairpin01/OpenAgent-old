# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Any

import aiohttp


class OpenAgentHttpClient:
    """Reusable aiohttp transport owned by the OpenAgent lifecycle."""

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(  # cubkit: ignore[missing-cleanup]
                cookie_jar=aiohttp.DummyCookieJar()
            )

    async def session(self) -> aiohttp.ClientSession:
        await self.start()
        assert self._session is not None
        return self._session

    async def close(self) -> None:
        session = self._session
        self._session = None
        if session is not None and not session.closed:
            await session.close()

    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        timeout_seconds: int,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        session = await self.session()
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout,
            ) as response:
                text = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"HTTP {response.status}: {text[:800]}")
                try:
                    data = await response.json()
                except Exception as exc:
                    raise RuntimeError(f"Invalid JSON response: {text[:800]}") from exc
                if not isinstance(data, dict):
                    raise RuntimeError("Invalid JSON response: expected an object")
                return data
        except TimeoutError as exc:
            raise RuntimeError(
                f"Provider request timed out after {timeout_seconds}s. "
                "Increase OpenAgent timeout or use a faster model for this task."
            ) from exc


__all__ = ["OpenAgentHttpClient"]
