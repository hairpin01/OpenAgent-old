from __future__ import annotations

import ast
import importlib
from pathlib import Path
import sys
from typing import Any, Mapping

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "Src"))
sys.path.insert(0, str(ROOT.parent / "repo-MCUB-fork" / "OpenAgent"))

from OpenAgentLib.PluginCapabilities import (  # noqa: E402
    CapabilityBroker,
    CapabilityErrorCode,
    CapabilityGrant,
    CapabilityRequest,
)
from OpenAgentLib.PluginSDK import (
    CapabilityCallContext,
    CapabilityClient,
    CapabilityFamily,
)  # noqa: E402
from OpenAgentLib.ToolCompatibility import TOOL_COMPATIBILITY_MATRIX  # noqa: E402
from OpenAgentLib.ToolKernel import ToolCall  # noqa: E402
from OpenAgentLib.ToolPolicy import (  # noqa: E402
    ConfirmationState,
    ToolConfirmationGrant,
    ToolPolicyCatalog,
    ToolPolicyEngine,
    ToolPolicyRequest,
    ToolPolicyRule,
)

TARGET_MODULES = ("terminal", "ast_grep", "file", "web")


class RecordingTransport:
    def __init__(self, *responses: Mapping[str, Any]) -> None:
        self.frames: list[Mapping[str, Any]] = []
        self._responses = list(responses)

    def request(self, frame: Mapping[str, Any]) -> Mapping[str, Any]:
        self.frames.append(frame)
        if not self._responses:
            raise AssertionError("unexpected capability request")
        return self._responses.pop(0)


class BrokerTransport:
    def __init__(
        self,
        broker: CapabilityBroker,
        call: ToolCall,
        policy_request: ToolPolicyRequest,
        grant: CapabilityGrant,
    ) -> None:
        self.broker = broker
        self.call = call
        self.policy_request = policy_request
        self.grant = grant

    def request(self, frame: Mapping[str, Any]) -> Mapping[str, Any]:
        request = CapabilityRequest(
            frame["host_request_id"],
            frame["call_id"],
            frame["canonical_tool_id"],
            frame["actor_scope"],
            frame["grant_id"],
            CapabilityFamily(frame["capability"]),
            frame["operation"],
            frame["capability_request_id"],
            frame["payload"],
        )
        return self.broker.dispatch(
            self.call, self.policy_request, self.grant, request
        ).to_envelope()


class AstGrepBackend:
    def __init__(self) -> None:
        self.calls: list[Mapping[str, Any]] = []

    def invoke(
        self, operation: str, payload: Mapping[str, Any], grant: CapabilityGrant
    ) -> Mapping[str, Any]:
        assert operation == "run"
        self.calls.append(payload)
        argv = payload["argv"]
        assert argv[0] == "ast-grep"
        assert "--update-all" in argv
        target = Path(payload["cwd"]) / argv[-1]
        target.write_text(
            target.read_text(encoding="utf-8").replace("print", "logger.info"),
            encoding="utf-8",
        )
        return {
            "exit_code": 0,
            "stdout": "--- fixture.py\n+++ fixture.py\n",
            "stderr": "",
        }


def _modules() -> tuple[object, ...]:
    return tuple(
        importlib.import_module(f"plugins.{module}") for module in TARGET_MODULES
    )


def _declaration(module: object, tool_id: str) -> object:
    return next(tool for tool in module.MANIFEST.tools if tool.canonical_id == tool_id)


def _call(module: object, tool_id: str, arguments: Mapping[str, Any]) -> ToolCall:
    declaration = _declaration(module, tool_id)
    return ToolCall(f"call-{tool_id}", declaration.to_tool_spec(), tool_id, arguments)


def _capability(
    call: ToolCall,
    transport: RecordingTransport | BrokerTransport,
    *,
    actor_scope: str = "actor:test",
) -> CapabilityClient:
    return CapabilityClient(
        CapabilityCallContext(
            "host-1", call.call_id, call.spec.canonical_id, actor_scope, "grant-1"
        ),
        transport,
    )


def _policy(call: ToolCall) -> tuple[ToolPolicyEngine, ToolPolicyRequest]:
    rule = ToolPolicyRule(
        call.spec.canonical_id,
        call.spec.capabilities,
        call.spec.confirmation,
        call.spec.concurrency,
        call.spec.idempotency,
        call.spec.migration_disposition,
    )
    confirmation = (
        ConfirmationState.APPROVED
        if call.spec.confirmation.value == "required"
        else ConfirmationState.MISSING
    )
    request = ToolPolicyRequest(
        frozenset({call.spec.canonical_id}),
        call.spec.capabilities,
        confirmation=confirmation,
        confirmation_grant=(
            ToolConfirmationGrant.for_call("confirmation-1", call)
            if confirmation is ConfirmationState.APPROVED
            else None
        ),
        requested_timeout=30,
        maximum_timeout=60,
        remaining_calls=1,
        remaining_token_budget=100,
        estimated_tokens=0,
    )
    return ToolPolicyEngine(ToolPolicyCatalog((rule,))), request


def _request(
    grant: CapabilityGrant,
    capability: CapabilityFamily,
    operation: str,
    payload: Mapping[str, Any],
) -> CapabilityRequest:
    return CapabilityRequest(
        grant.host_request_id,
        grant.call_id,
        grant.canonical_tool_id,
        grant.actor_scope,
        grant.grant_id,
        capability,
        operation,
        "request-1",
        payload,
    )


def _global_dns(host: str, *_args: Any, **_kwargs: Any) -> list[tuple[Any, ...]]:
    if host == "redirect.invalid":
        return [(None, None, None, None, ("127.0.0.1", 443))]
    return [(None, None, None, None, ("93.184.216.34", 443))]


def test_matrix_canonical_ids_and_aliases_are_declared_exactly() -> None:
    expected = {
        entry.canonical_id: entry
        for entry in TOOL_COMPATIBILITY_MATRIX
        if entry.source_module in TARGET_MODULES
    }
    declared = {
        tool.canonical_id: tool
        for module in _modules()
        for tool in module.MANIFEST.tools
    }

    assert set(declared) == set(expected)
    for tool_id, entry in expected.items():
        assert declared[tool_id].aliases == entry.aliases
        assert declared[tool_id].capabilities == frozenset({entry.capability_class})


def test_resource_modules_have_no_ambient_or_direct_resource_apis() -> None:
    plugins_root = ROOT.parent / "repo-MCUB-fork" / "OpenAgent" / "plugins"
    forbidden = (
        "create_subprocess_shell",
        "create_subprocess_exec",
        "shell=True",
        "os.system",
        "subprocess",
        "aiohttp",
        "socket",
        "Path.cwd",
        "pathlib",
        "self.agent",
        "self.config",
        "tool_registry",
        "tool_map",
    )
    for name in (*TARGET_MODULES, "_resource_v2"):
        source = (plugins_root / f"{name}.py").read_text(encoding="utf-8")
        ast.parse(source)
        assert not [token for token in forbidden if token in source], name


def test_terminal_argv_is_allowlisted_and_shell_metacharacters_remain_data() -> None:
    terminal = importlib.import_module("plugins.terminal")
    call = _call(
        terminal, "terminal.run", {"argv": ["echo", "; touch /tmp/pwned"], "cwd": "."}
    )
    transport = RecordingTransport(
        {
            "ok": True,
            "data": {"exit_code": 0, "stdout": "; touch /tmp/pwned", "stderr": ""},
        }
    )

    result = terminal.HANDLERS["terminal.run"](call, _capability(call, transport))

    assert result["stdout"] == "; touch /tmp/pwned"
    assert transport.frames[0]["payload"]["argv"] == ["echo", "; touch /tmp/pwned"]
    with pytest.raises(Exception):
        _call(terminal, "terminal.run", {"command": "echo injected"})
    blocked = _call(terminal, "terminal.run", {"argv": ["sh", "-c", "id"], "cwd": "."})
    with pytest.raises(ValueError, match="allowlisted"):
        terminal.HANDLERS["terminal.run"](
            blocked, _capability(blocked, RecordingTransport())
        )
    find_escape = _call(
        terminal,
        "terminal.run",
        {"argv": ["find", ".", "-exec", "/bin/sh", "-c", "id", ";"], "cwd": "."},
    )
    with pytest.raises(ValueError, match="allowlisted"):
        terminal.HANDLERS["terminal.run"](
            find_escape, _capability(find_escape, RecordingTransport())
        )
    path_escape = _call(
        terminal, "terminal.run", {"argv": ["cat", "/etc/passwd"], "cwd": "."}
    )
    with pytest.raises(ValueError, match="grant-relative"):
        terminal.HANDLERS["terminal.run"](
            path_escape, _capability(path_escape, RecordingTransport())
        )


def test_terminal_process_output_is_bounded() -> None:
    terminal = importlib.import_module("plugins.terminal")
    call = _call(terminal, "terminal.run", {"argv": ["echo", "large"], "cwd": "."})
    transport = RecordingTransport(
        {"ok": True, "data": {"exit_code": 0, "stdout": "x" * 20_000, "stderr": ""}}
    )

    result = terminal.HANDLERS["terminal.run"](call, _capability(call, transport))

    assert len(result["stdout"].encode("utf-8")) == 12_000
    assert result["truncated"] is True


def test_resource_handlers_reject_malformed_capability_success_frames() -> None:
    terminal = importlib.import_module("plugins.terminal")
    call = _call(terminal, "terminal.run", {"argv": ["echo", "ok"], "cwd": "."})

    with pytest.raises(ValueError, match="boolean ok"):
        terminal.HANDLERS["terminal.run"](
            call, _capability(call, RecordingTransport({"data": {}}))
        )
    with pytest.raises(ValueError, match="bounded text"):
        terminal.HANDLERS["terminal.run"](
            call,
            _capability(
                call,
                RecordingTransport(
                    {"ok": True, "data": {"exit_code": 0, "stdout": "ok"}}
                ),
            ),
        )


def test_terminal_inspect_uses_only_fixed_metadata_argv() -> None:
    terminal = importlib.import_module("plugins.terminal")
    call = _call(
        terminal, "terminal.inspect", {"operation": "git-status", "cwd": "project"}
    )
    transport = RecordingTransport(
        {"ok": True, "data": {"exit_code": 0, "stdout": " M file.py", "stderr": ""}}
    )

    terminal.HANDLERS["terminal.inspect"](call, _capability(call, transport))

    assert transport.frames[0]["payload"]["argv"] == ["git", "status", "--short"]
    with pytest.raises(Exception):
        _call(terminal, "terminal.inspect", {"command": "id", "cwd": "project"})


def test_scoped_ast_grep_replace_changes_only_granted_fixture(tmp_path: Path) -> None:
    ast_grep = importlib.import_module("plugins.ast_grep")
    fixture = tmp_path / "fixture.py"
    untouched = tmp_path / "untouched.py"
    fixture.write_text("print('fixture')\n", encoding="utf-8")
    untouched.write_text("print('untouched')\n", encoding="utf-8")
    call = _call(
        ast_grep,
        "ast_grep.replace",
        {
            "pattern": "print($$$)",
            "rewrite": "logger.info($$$)",
            "lang": "python",
            "path": "fixture.py",
            "apply": True,
        },
    )
    policy, policy_request = _policy(call)
    backend = AstGrepBackend()
    broker = CapabilityBroker(policy, {CapabilityFamily.PROCESS: backend})
    grant = CapabilityGrant.for_call(
        "grant-1",
        "host-1",
        call,
        CapabilityFamily.PROCESS,
        frozenset({"run"}),
        {
            "executables": ["ast-grep"],
            "cwd_root": str(tmp_path),
            "max_timeout_seconds": 30,
            "max_output_bytes": 24_000,
            "max_args": 128,
            "max_arg_length": 8_192,
            "env_allowlist": [],
        },
    )

    result = ast_grep.HANDLERS["ast_grep.replace"](
        call,
        _capability(
            call,
            BrokerTransport(broker, call, policy_request, grant),
            actor_scope=grant.actor_scope,
        ),
    )

    assert call.spec.capabilities == frozenset({"filesystem-write"})
    assert fixture.read_text(encoding="utf-8") == "logger.info('fixture')\n"
    assert untouched.read_text(encoding="utf-8") == "print('untouched')\n"
    assert result == {
        "ok": True,
        "exit_code": 0,
        "result": "--- fixture.py\n+++ fixture.py\n",
        "truncated": False,
        "applied": True,
    }


def test_ast_grep_rejects_traversal_before_capability_request() -> None:
    ast_grep = importlib.import_module("plugins.ast_grep")
    call = _call(
        ast_grep,
        "ast_grep.search",
        {"pattern": "print($$$)", "lang": "python", "path": "../secret.py"},
    )
    transport = RecordingTransport()

    with pytest.raises(ValueError, match="grant-relative"):
        ast_grep.HANDLERS["ast_grep.search"](call, _capability(call, transport))
    assert not transport.frames


def test_ast_grep_rejects_glob_traversal_before_capability_request() -> None:
    ast_grep = importlib.import_module("plugins.ast_grep")
    call = _call(
        ast_grep,
        "ast_grep.search",
        {
            "pattern": "print($$$)",
            "lang": "python",
            "path": "src",
            "globs": ["../**/*.py"],
        },
    )
    transport = RecordingTransport()

    with pytest.raises(ValueError, match="glob"):
        ast_grep.HANDLERS["ast_grep.search"](call, _capability(call, transport))
    assert not transport.frames


def test_granted_workspace_list_read_and_https_fetch_use_narrow_operations() -> None:
    terminal = importlib.import_module("plugins.terminal")
    web = importlib.import_module("plugins.web")
    list_call = _call(terminal, "terminal.list_files", {"path": "."})
    read_call = _call(terminal, "terminal.read_file", {"path": "README.md"})
    fetch_call = _call(web, "web.fetch_url", {"url": "https://example.com"})
    list_transport = RecordingTransport(
        {"ok": True, "data": {"entries": ["README.md", "src/"]}}
    )
    read_transport = RecordingTransport({"ok": True, "data": {"content": "hello"}})
    fetch_transport = RecordingTransport(
        {
            "ok": True,
            "data": {
                "url": "https://example.com",
                "content_type": "text/plain",
                "content": "hello",
            },
        }
    )

    assert terminal.HANDLERS["terminal.list_files"](
        list_call, _capability(list_call, list_transport)
    )["entries"] == ["README.md", "src/"]
    assert (
        terminal.HANDLERS["terminal.read_file"](
            read_call, _capability(read_call, read_transport)
        )["content"]
        == "hello"
    )
    assert (
        web.HANDLERS["web.fetch_url"](
            fetch_call, _capability(fetch_call, fetch_transport)
        )["content"]
        == "hello"
    )
    assert list_transport.frames[0]["capability"] == "workspace-fs"
    assert list_transport.frames[0]["operation"] == "list"
    assert read_transport.frames[0]["operation"] == "read"
    assert fetch_transport.frames[0]["capability"] == "https-fetch"


def test_file_edit_uses_read_version_as_atomic_write_guard() -> None:
    file_plugin = importlib.import_module("plugins.file")
    call = _call(
        file_plugin,
        "file.edit",
        {"path": "notes/todo.txt", "search": "old", "replace": "new"},
    )
    transport = RecordingTransport(
        {"ok": True, "data": {"content": "old\n", "version": "hash-1"}},
        {"ok": True, "data": {"version": "hash-2"}},
    )

    result = file_plugin.HANDLERS["file.edit"](call, _capability(call, transport))

    assert result == {
        "ok": True,
        "path": "notes/todo.txt",
        "changed": True,
        "version": "hash-2",
    }
    assert transport.frames[1]["payload"] == {
        "path": "notes/todo.txt",
        "content": "new\n",
        "mode": "overwrite",
        "expected_hash": "hash-1",
    }


def test_file_empty_write_uses_backend_changed_result() -> None:
    file_plugin = importlib.import_module("plugins.file")
    call = _call(file_plugin, "file.write", {"path": "notes/todo.txt", "content": ""})
    transport = RecordingTransport(
        {"ok": True, "data": {"version": "hash-2", "changed": True}}
    )

    assert (
        file_plugin.HANDLERS["file.write"](call, _capability(call, transport))[
            "changed"
        ]
        is True
    )


def test_file_edit_conflict_does_not_report_an_overwrite() -> None:
    file_plugin = importlib.import_module("plugins.file")
    call = _call(
        file_plugin,
        "file.edit",
        {"path": "notes/todo.txt", "search": "old", "replace": "new"},
    )
    transport = RecordingTransport(
        {"ok": True, "data": {"content": "old\n", "version": "hash-1"}},
        {"ok": False, "data": {}, "error": "conflict"},
    )

    with pytest.raises(ValueError, match="denied"):
        file_plugin.HANDLERS["file.edit"](call, _capability(call, transport))
    assert transport.frames[1]["payload"]["expected_hash"] == "hash-1"


def test_file_patch_rejects_escaped_header_before_reading() -> None:
    file_plugin = importlib.import_module("plugins.file")
    call = _call(
        file_plugin,
        "file.patch",
        {
            "path": "notes/todo.txt",
            "patch": "--- a/../secret.txt\n+++ b/../secret.txt\n@@ -1 +1 @@\n-old\n+new\n",
        },
    )
    transport = RecordingTransport()

    with pytest.raises(ValueError, match="grant-relative"):
        file_plugin.HANDLERS["file.patch"](call, _capability(call, transport))
    assert not transport.frames


def test_file_patch_applies_a_scoped_unified_diff_with_guard() -> None:
    file_plugin = importlib.import_module("plugins.file")
    call = _call(
        file_plugin,
        "file.patch",
        {
            "path": "notes/todo.txt",
            "patch": "--- a/notes/todo.txt\n+++ b/notes/todo.txt\n@@ -1 +1 @@\n-old\n+new\n",
        },
    )
    transport = RecordingTransport(
        {"ok": True, "data": {"content": "old\n", "version": "hash-1"}},
        {"ok": True, "data": {"version": "hash-2"}},
    )

    result = file_plugin.HANDLERS["file.patch"](call, _capability(call, transport))

    assert result["changed"] is True
    assert transport.frames[1]["payload"]["content"] == "new\n"
    assert transport.frames[1]["payload"]["expected_hash"] == "hash-1"


def test_broker_rejects_symlink_escape_and_normalizes_guarded_write(
    tmp_path: Path,
) -> None:
    file_plugin = importlib.import_module("plugins.file")
    call = _call(
        file_plugin, "file.write", {"path": "notes/todo.txt", "content": "new"}
    )
    policy, policy_request = _policy(call)
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    grant = CapabilityGrant.for_call(
        "grant-1",
        "host-1",
        call,
        CapabilityFamily.WORKSPACE_FS,
        frozenset({"write"}),
        {"root": str(tmp_path)},
    )
    backend = RecordingFilesystemBackend()
    broker = CapabilityBroker(policy, {CapabilityFamily.WORKSPACE_FS: backend})

    escaped = broker.dispatch(
        call,
        policy_request,
        grant,
        _request(
            grant,
            CapabilityFamily.WORKSPACE_FS,
            "write",
            {"path": "link/escape.txt", "content": "x"},
        ),
    )
    accepted = broker.dispatch(
        call,
        policy_request,
        grant,
        _request(
            grant,
            CapabilityFamily.WORKSPACE_FS,
            "write",
            {
                "path": "notes/todo.txt",
                "content": "new",
                "mode": "overwrite",
                "expected_hash": "hash-1",
            },
        ),
    )

    assert escaped.error is CapabilityErrorCode.INVALID_REQUEST
    assert accepted.ok
    assert backend.calls == [
        {
            "operation": "write",
            "payload": {
                "root": str(tmp_path),
                "components": ("notes", "todo.txt"),
                "content": "new",
                "mode": "overwrite",
                "expected_hash": "hash-1",
            },
        }
    ]


class RecordingFilesystemBackend:
    def __init__(self) -> None:
        self.calls: list[Mapping[str, Any]] = []

    def invoke(
        self, operation: str, payload: Mapping[str, Any], grant: CapabilityGrant
    ) -> Mapping[str, Any]:
        self.calls.append({"operation": operation, "payload": dict(payload)})
        return {"version": "hash-2"}


def test_https_capability_blocks_private_dns_and_private_redirects() -> None:
    web = importlib.import_module("plugins.web")
    call = _call(web, "web.fetch_url", {"url": "https://initial.invalid"})
    policy, policy_request = _policy(call)
    grant = CapabilityGrant.for_call(
        "grant-1",
        "host-1",
        call,
        CapabilityFamily.HTTPS_FETCH,
        frozenset({"fetch"}),
        {"max_timeout_seconds": 20, "max_bytes": 262_144},
    )
    request = _request(
        grant,
        CapabilityFamily.HTTPS_FETCH,
        "fetch",
        {
            "url": "https://initial.invalid",
            "timeout_seconds": 10,
            "max_bytes": 1_024,
        },
    )
    private_broker = CapabilityBroker(
        policy,
        {CapabilityFamily.HTTPS_FETCH: RecordingHttpsBackend()},
        resolver=lambda *_args, **_kwargs: [
            (None, None, None, None, ("127.0.0.1", 443))
        ],
    )
    redirect_broker = CapabilityBroker(
        policy,
        {
            CapabilityFamily.HTTPS_FETCH: RecordingHttpsBackend(
                ["https://redirect.invalid"]
            )
        },
        resolver=_global_dns,
    )

    assert (
        private_broker.dispatch(call, policy_request, grant, request).error
        is CapabilityErrorCode.INVALID_REQUEST
    )
    assert (
        redirect_broker.dispatch(call, policy_request, grant, request).error
        is CapabilityErrorCode.BACKEND_ERROR
    )


class RecordingHttpsBackend:
    def __init__(self, redirects: list[str] | None = None) -> None:
        self.redirects = redirects or []

    def invoke(
        self, operation: str, payload: Mapping[str, Any], grant: CapabilityGrant
    ) -> Mapping[str, Any]:
        return {"content": "ok", "redirect_urls": self.redirects}


def test_broker_rejects_undeclared_workspace_operation() -> None:
    file_plugin = importlib.import_module("plugins.file")
    call = _call(file_plugin, "file.read_text", {"path": "notes/todo.txt"})
    policy, policy_request = _policy(call)
    grant = CapabilityGrant.for_call(
        "grant-1",
        "host-1",
        call,
        CapabilityFamily.WORKSPACE_FS,
        frozenset({"read"}),
        {"root": str(ROOT)},
    )
    broker = CapabilityBroker(
        policy, {CapabilityFamily.WORKSPACE_FS: RecordingFilesystemBackend()}
    )

    response = broker.dispatch(
        call,
        policy_request,
        grant,
        _request(
            grant, CapabilityFamily.WORKSPACE_FS, "delete", {"path": "notes/todo.txt"}
        ),
    )
    wrong_capability = broker.dispatch(
        call,
        policy_request,
        grant,
        _request(
            grant,
            CapabilityFamily.PROCESS,
            "run",
            {
                "argv": ["echo", "no"],
                "cwd": ".",
                "timeout_seconds": 1,
                "max_output_bytes": 1,
            },
        ),
    )

    assert response.error is CapabilityErrorCode.UNKNOWN_OPERATION
    assert wrong_capability.error is CapabilityErrorCode.INVALID_GRANT
