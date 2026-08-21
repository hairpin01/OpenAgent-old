# OpenAgent Plugins

OpenAgent plugins use the shipped isolated v2 system. A plugin is a Python source
file that publishes a static `PluginManifest`. The parent process inspects the
source with the AST and records its hash, but never imports or executes the
plugin. A statically admitted handler runs only in a Linux Bubblewrap host.

Linux with `/usr/bin/bwrap` is required for external plugins. There is no
unsandboxed fallback.

## Manifest

Import declarations from `OpenAgentLib.PluginSDK`. `PluginManifest` is immutable
and versioned. Its fields are:

| Field | Meaning |
| --- | --- |
| `plugin_id` | Lowercase dotted identifier, such as `example.echo`. |
| `version` | Plugin version. |
| `api_version` | Must be `"2"`. |
| `entrypoint` | Dotted symbol resolved inside the isolated worker, normally `plugins.example.HANDLERS`. |
| `tools` | Non-empty tuple of `PluginToolDeclaration` values. |
| `capabilities` | Declared capability classes used by the plugin's tools. |
| `manifest_version` | Must be `"2"`. |
| `metadata` | Frozen JSON metadata. |

Each `PluginToolDeclaration` defines a canonical tool ID, optional normalized
aliases, JSON input and output schemas, a description, confirmation policy,
concurrency and idempotency classes, migration disposition, and one or more
declared capability classes. Canonical IDs and aliases must be unique. An alias
cannot collide with a canonical ID.

The minimal safe shape is:

```python
from typing import Any, Callable, Mapping

from OpenAgentLib.PluginSDK import CapabilityClient, PluginManifest, PluginToolDeclaration
from OpenAgentLib.ToolKernel import ToolCall


def echo(call: ToolCall, capability: CapabilityClient) -> Mapping[str, Any]:
    return {"text": call.arguments["text"]}


MANIFEST = PluginManifest(
    plugin_id="example.echo",
    version="2.0.0",
    api_version="2",
    entrypoint="plugins.example.HANDLERS",
    tools=(PluginToolDeclaration(
        canonical_id="example.echo",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        capabilities=frozenset({"configuration"}),
    ),),
    capabilities=frozenset({"configuration"}),
)

HANDLERS: Mapping[str, Callable[[ToolCall, CapabilityClient], Mapping[str, Any]]] = {
    "example.echo": echo,
}
```

A handler receives a validated `ToolCall` and a `CapabilityClient`. It returns a
JSON object matching the declared output schema. It must not use parent objects,
ambient credentials, direct Telegram clients, direct filesystem APIs, or direct
network and process APIs.

## Admission And Execution

Top-level plugin files are discovered deterministically. For each file,
`PluginDiscovery.inspect_v2_plugin_source` parses the source without importing
it. Admission recognizes either a `MANIFEST` or `PLUGIN_MANIFEST` assignment
calling `PluginManifest` imported from `OpenAgentLib.PluginSDK`, or the shipped
static factory marker: `build_plugin(...)` imported from `_telegram_v2`, as used
by the sibling Telegram modules. The factory marker is recognized only for
static admission and data extraction; it does not execute the factory in the
parent. Arbitrary factories, aliases, or dynamic manifest construction remain
rejected. The admitted source is stored with its SHA-256 digest.

At tool-call time the isolated invoker verifies the admitted source hash,
plugin ID, plugin version, canonical tool ID, and declared `HANDLERS` entrypoint.
The worker then loads the handler inside Bubblewrap. The parent owns tool policy,
confirmation, retry and concurrency decisions. The worker cannot change those
decisions through its manifest or return value.

## Capabilities And Grants

`CapabilityClient.request` sends correlated JSON frames. It carries the host
request ID, call ID, canonical tool ID, actor scope, grant ID, capability family,
operation, request ID, and JSON payload. The client has no ambient parent object.

The supported capability families and their grant-controlled operations are:

| Family | Operations |
| --- | --- |
| `telegram` | Telegram operations defined by the frozen schemas, using opaque IDs and data. |
| `workspace-fs` | `read`, `write`, `list`, within the granted workspace root. |
| `process` | `run`, with an allowlisted executable, arguments, environment, root, and timeout. |
| `https-fetch` | `fetch`, restricted to validated public HTTPS URLs. |
| `scheduling` | `schedule`. |
| `configuration` | `get`, `set`, in the grant-namespaced JSON settings store. |
| `mcub-control` | The explicitly defined MCUB control operations only. |

The parent issues an immutable `CapabilityGrant` for exactly one active tool
call. A request is accepted only when its grant matches the call, actor scope,
canonical tool, capability family, operation, constraints, and correlation
IDs. Grants are not capabilities themselves: policy and grant validation still
apply, requests are replay-protected, and unknown families or operations fail
closed. A manifest may declare required capabilities, but cannot grant itself
access or widen a parent grant.

## Aliases And Migration

The v2 registry is built from the canonical tool IDs and aliases in the frozen
`TOOL_COMPATIBILITY_MATRIX`. Aliases are explicit migration mappings, not
runtime ownership inferred from declaration order. The matrix also freezes
schemas, confirmation, capability, concurrency, idempotency, and migration
disposition.

Migration lasts one release. A legacy declaration may be converted to a v2
manifest only when its canonical ID, handler ownership, aliases, schemas, and
policy metadata match the matrix. The conversion is a boundary aid and does not
restore legacy execution.

These legacy APIs and execution paths are removed and rejected before module
execution:

- `OpenAgentPlugin` class loading and its `on_load` or lifecycle hooks.
- `tool_registry` and `tool_map` declarations.
- `_dispatch_tool` and the old in-process plugin loader.
- Legacy `agent`, `client`, `kernel`, or watcher injection and callbacks.
- Parent-side importing of plugin modules or direct handler calls.
- Unsandboxed execution, ambient agent or configuration access, and direct resource APIs.
- Explicitly rejected aliases, including `chat.search` and `eval.python.telegram`.

`tool_registry` and `tool_map` may appear in migration diagnostics or tests as
rejected legacy input, but they are not valid v2 plugin fields. A source that
does not contain the direct manifest marker or the shipped `build_plugin(...)`
marker raises a migration error without executing its module code.

## Testing And Examples

The sibling v2 examples live in `../repo-MCUB-fork/OpenAgent/plugins/`. They
include Telegram handlers, workspace and process resource handlers, web fetch,
task scheduling, MCUB control, and the `eval` example. Telegram siblings use
the statically recognized `build_plugin("...")` marker from `_telegram_v2` and
publish its resulting `MANIFEST` and `HANDLERS`; other examples construct a
`PluginManifest` directly. In both forms, the parent extracts declarations
without running plugin code, and resource handlers use `CapabilityClient`
rather than direct resource APIs.

Relevant checks include:

```bash
python -m pytest -q tests/test_plugin_sdk.py tests/test_plugin_host.py tests/test_legacy_execution_removed.py
python -m pytest -q tests/test_plugin_telegram_v2.py tests/test_plugin_resource_v2.py tests/test_plugin_runtime_v2.py
```

The release verification also requires `cubkit check . --release`,
`cubkit lint . --release --no-cache`, a reproducible release build, and
`python -m py_compile` on the built artifact. A host without `/usr/bin/bwrap`
must fail closed rather than execute a plugin outside isolation.

## Migration Checklist

- Define one immutable v2 `PluginManifest` with API and manifest version `"2"`.
- Define every tool with `PluginToolDeclaration`, JSON schemas, aliases, policy metadata, and capabilities.
- Export a mapping named by the manifest entrypoint, normally `HANDLERS`.
- Make every handler accept `(ToolCall, CapabilityClient)` and return schema-valid JSON.
- Replace direct Telegram, filesystem, process, network, scheduling, configuration, and MCUB access with the matching capability request.
- Compare canonical IDs and aliases with `TOOL_COMPATIBILITY_MATRIX`; reject unmapped legacy aliases.
- Run the static admission, isolated-host, capability, migration, and release checks.
- Verify Bubblewrap is present on the target Linux host. Do not add an unsandboxed fallback.
