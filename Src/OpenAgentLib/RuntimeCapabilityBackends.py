# SPDX-License-Identifier: MIT
"""Concrete, parent-owned backends for isolated v2 plugin capabilities."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from .PluginCapabilities import CapabilityGrant, CapabilityProtocolError


class RuntimeFilesystemBackend:
    def invoke(
        self, operation: str, payload: Mapping[str, Any], grant: CapabilityGrant
    ) -> Mapping[str, Any]:
        path = Path(str(payload["path"]))
        if operation == "read":
            text = path.read_text(encoding="utf-8", errors="replace")
            return {
                "content": text,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        if operation == "list":
            return {"entries": sorted(item.name for item in path.iterdir())}
        if operation == "write":
            content = str(payload["content"])
            expected = payload.get("expected_hash")
            if path.exists() and expected:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != expected:
                    raise CapabilityProtocolError("workspace file version changed")
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if payload["mode"] == "append" else "w"
            with path.open(mode, encoding="utf-8") as handle:
                handle.write(content)
            return {
                "changed": True,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        raise CapabilityProtocolError("unsupported filesystem operation")


class RuntimeProcessBackend:
    def invoke(
        self, operation: str, payload: Mapping[str, Any], grant: CapabilityGrant
    ) -> Mapping[str, Any]:
        if operation != "run":
            raise CapabilityProtocolError("unsupported process operation")
        completed = subprocess.run(
            list(payload["argv"]),
            cwd=str(payload["cwd"]),
            env=dict(payload["env"]),
            capture_output=True,
            text=True,
            timeout=float(payload["timeout_seconds"]),
            check=False,
        )
        limit = int(payload["max_output_bytes"])
        return {
            "exit_code": completed.returncode,
            "stdout": completed.stdout[:limit],
            "stderr": completed.stderr[:limit],
        }


class RuntimeHttpsBackend:
    def invoke(
        self, operation: str, payload: Mapping[str, Any], grant: CapabilityGrant
    ) -> Mapping[str, Any]:
        if operation != "fetch":
            raise CapabilityProtocolError("unsupported HTTPS operation")
        request = Request(str(payload["url"]), headers={"User-Agent": "OpenAgent"})
        with urlopen(request, timeout=float(payload["timeout_seconds"])) as response:
            body = response.read(int(payload["max_bytes"]))
            return {
                "status": response.status,
                "body": body.decode("utf-8", errors="replace"),
                "redirect_urls": [response.url],
            }


class RuntimeConfigurationBackend:
    def __init__(self, config: Any) -> None:
        self._config = config

    def invoke(
        self, operation: str, payload: Mapping[str, Any], grant: CapabilityGrant
    ) -> Mapping[str, Any]:
        key = str(payload["key"])
        if operation == "get":
            return {"value": self._config.get(key)}
        if operation == "set":
            self._config[key] = payload["value"]
            return {"changed": True}
        raise CapabilityProtocolError("unsupported configuration operation")


__all__ = [
    "RuntimeConfigurationBackend",
    "RuntimeFilesystemBackend",
    "RuntimeHttpsBackend",
    "RuntimeProcessBackend",
]
