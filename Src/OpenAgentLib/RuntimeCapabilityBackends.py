# SPDX-License-Identifier: MIT
"""Concrete, parent-owned backends for isolated v2 plugin capabilities."""

from __future__ import annotations

import hashlib
import http.client
import os
import secrets
import socket
import subprocess
from contextlib import contextmanager
from collections.abc import Mapping
from typing import Any, Callable, Iterator
from urllib.parse import urlsplit

from .PluginCapabilities import CapabilityGrant, CapabilityProtocolError


class RuntimeFilesystemBackend:
    _DIRECTORY_FLAGS = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    )
    _FILE_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    _FILE_WRITE_FLAGS = os.O_WRONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)

    def invoke(
        self, operation: str, payload: Mapping[str, Any], grant: CapabilityGrant
    ) -> Mapping[str, Any]:
        root = payload.get("root")
        if not isinstance(root, str) or not root:
            raise CapabilityProtocolError("filesystem operation lacks a grant root")
        components = self._components(
            payload.get("components"), allow_empty=operation == "list"
        )
        try:
            root_fd = os.open(root, self._DIRECTORY_FLAGS)
            try:
                if operation == "read":
                    with self._open_parent(root_fd, components, create=False) as (
                        parent_fd,
                        name,
                    ):
                        content = self._read_bytes(parent_fd, name)
                    return {
                        "content": content.decode("utf-8", errors="replace"),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                if operation == "list":
                    with self._open_directory(
                        root_fd, components, create=False
                    ) as directory_fd:
                        return {"entries": sorted(os.listdir(directory_fd))}
                if operation == "write":
                    return self._write(root_fd, components, payload)
            finally:
                os.close(root_fd)
        except OSError as exc:
            raise CapabilityProtocolError(
                "workspace path is unavailable or unsafe"
            ) from exc
        raise CapabilityProtocolError("unsupported filesystem operation")

    @staticmethod
    def _components(value: Any, *, allow_empty: bool) -> tuple[str, ...]:
        if (
            not isinstance(value, tuple)
            or (not allow_empty and not value)
            or any(
                not isinstance(component, str)
                or not component
                or component in {".", ".."}
                or "/" in component
                for component in value
            )
        ):
            raise CapabilityProtocolError("filesystem operation has invalid components")
        return value

    @contextmanager
    def _open_directory(
        self, root_fd: int, components: tuple[str, ...], *, create: bool
    ) -> Iterator[int]:
        opened: list[int] = []
        current_fd = root_fd
        try:
            for component in components:
                try:
                    next_fd = os.open(
                        component, self._DIRECTORY_FLAGS, dir_fd=current_fd
                    )
                except FileNotFoundError:
                    if not create:
                        raise
                    try:
                        os.mkdir(component, 0o700, dir_fd=current_fd)
                    except FileExistsError:
                        pass
                    next_fd = os.open(
                        component, self._DIRECTORY_FLAGS, dir_fd=current_fd
                    )
                opened.append(next_fd)
                current_fd = next_fd
            yield current_fd
        finally:
            for descriptor in reversed(opened):
                os.close(descriptor)

    @contextmanager
    def _open_parent(
        self, root_fd: int, components: tuple[str, ...], *, create: bool
    ) -> Iterator[tuple[int, str]]:
        with self._open_directory(root_fd, components[:-1], create=create) as parent_fd:
            yield parent_fd, components[-1]

    def _read_bytes(
        self, parent_fd: int, name: str, *, missing_ok: bool = False
    ) -> bytes | None:
        try:
            descriptor = os.open(name, self._FILE_READ_FLAGS, dir_fd=parent_fd)
        except FileNotFoundError:
            if missing_ok:
                return None
            raise
        try:
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 65_536):
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    def _write(
        self, root_fd: int, components: tuple[str, ...], payload: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        content = payload.get("content")
        mode = payload.get("mode")
        if not isinstance(content, str) or mode not in {"overwrite", "append"}:
            raise CapabilityProtocolError(
                "filesystem write has invalid content or mode"
            )
        data = content.encode("utf-8")
        with self._open_parent(root_fd, components, create=True) as (parent_fd, name):
            existing = self._read_bytes(parent_fd, name, missing_ok=True)
            expected = payload.get("expected_hash")
            if expected is not None and existing is not None:
                actual = hashlib.sha256(existing).hexdigest()
                if actual != expected:
                    raise CapabilityProtocolError("workspace file version changed")
            if mode == "overwrite":
                self._atomic_replace(parent_fd, name, data)
            else:
                self._append(parent_fd, name, data)
            current = self._read_bytes(parent_fd, name)
        assert current is not None
        return {"changed": True, "sha256": hashlib.sha256(current).hexdigest()}

    def _atomic_replace(self, parent_fd: int, name: str, data: bytes) -> None:
        temporary: str | None = None
        descriptor: int | None = None
        try:
            for _ in range(3):
                candidate = f".openagent-{secrets.token_hex(16)}.tmp"
                try:
                    descriptor = os.open(
                        candidate,
                        self._FILE_WRITE_FLAGS | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=parent_fd,
                    )
                except FileExistsError:
                    continue
                temporary = candidate
                break
            if descriptor is None or temporary is None:
                raise CapabilityProtocolError(
                    "could not create a safe workspace temp file"
                )
            self._write_all(descriptor, data)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(temporary, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            temporary = None
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary is not None:
                os.unlink(temporary, dir_fd=parent_fd)

    def _append(self, parent_fd: int, name: str, data: bytes) -> None:
        descriptor = os.open(
            name,
            self._FILE_WRITE_FLAGS | os.O_APPEND | os.O_CREAT,
            0o600,
            dir_fd=parent_fd,
        )
        try:
            self._write_all(descriptor, data)
        finally:
            os.close(descriptor)

    @staticmethod
    def _write_all(descriptor: int, data: bytes) -> None:
        remaining = memoryview(data)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("could not write workspace file")
            remaining = remaining[written:]


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


class _PinnedHttpsConnection(http.client.HTTPSConnection):
    """Connect directly to a broker-selected address while retaining HTTPS identity."""

    def __init__(
        self,
        hostname: str,
        port: int,
        address: str,
        address_family: int,
        timeout: float,
    ) -> None:
        super().__init__(hostname, port=port, timeout=timeout)
        self._address = address
        self._address_family = address_family

    def connect(self) -> None:
        if self._tunnel_host:
            raise CapabilityProtocolError("HTTPS proxy tunneling is not allowed")
        sock = socket.socket(self._address_family, socket.SOCK_STREAM)
        try:
            sock.settimeout(self.timeout)
            endpoint: tuple[Any, ...] = (
                (self._address, self.port)
                if self._address_family == socket.AF_INET
                else (self._address, self.port, 0, 0)
            )
            sock.connect(endpoint)
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
        except BaseException:
            sock.close()
            raise


class RuntimeHttpsBackend:
    _REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

    def __init__(
        self,
        connection_factory: Callable[
            [str, int, str, int, float], Any
        ] = _PinnedHttpsConnection,
    ) -> None:
        self._connection_factory = connection_factory

    def invoke(
        self, operation: str, payload: Mapping[str, Any], grant: CapabilityGrant
    ) -> Mapping[str, Any]:
        if operation != "fetch":
            raise CapabilityProtocolError("unsupported HTTPS operation")
        url = str(payload["url"])
        hostname = str(payload["hostname"])
        port = int(payload["port"])
        connection = self._connection_factory(
            hostname,
            port,
            str(payload["address"]),
            int(payload["address_family"]),
            float(payload["timeout_seconds"]),
        )
        parsed = urlsplit(url)
        target = parsed.path or "/"
        if parsed.query:
            target += f"?{parsed.query}"
        host_header = f"[{hostname}]" if ":" in hostname else hostname
        try:
            connection.request(
                "GET",
                target,
                headers={"Host": host_header, "User-Agent": "OpenAgent"},
            )
            response = connection.getresponse()
            if response.status in self._REDIRECT_STATUSES:
                location = response.getheader("Location")
                if not isinstance(location, str) or not location:
                    raise CapabilityProtocolError(
                        "HTTPS redirect has no Location header"
                    )
                return {"status": response.status, "redirect_url": location}
            max_bytes = int(payload["max_bytes"])
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise CapabilityProtocolError(
                    "HTTPS response exceeds granted byte limit"
                )
            return {
                "status": response.status,
                "body": body.decode("utf-8", errors="replace"),
            }
        finally:
            connection.close()


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
