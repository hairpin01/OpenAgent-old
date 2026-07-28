# OpenAgent Plugins

## Overview

OpenAgent uses a **dynamic plugin system** similar to how MCUB loads modules:
the agent does **not** know in advance which plugins are installed.
Plugins are auto-discovered from two directories, loaded at runtime,
and register their tools into the shared registry.

Plugins replace the old hardcoded tool map.
This makes the tool set extensible without modifying the module itself.

---

## Architecture

```
OpenAgent-MCUB-repo.py          # module (loader, core tools, dispatch)
OpenAgent/
├── doc/                         # this documentation
└── plugins/                     # bundled plugins (shipped with repo)
    ├── terminal.py
    ├── ast_grep.py
    ├── web.py
    ├── mcub.py
    ├── message.py
    ├── dialog.py
    ├── chat.py
    ├── moderation.py
    ├── profile.py
    ├── file.py
    ├── contacts.py
    └── creation.py
```

**Scan order** (at startup):

1. `OpenAgent/plugins/` — bundled plugins (repo/main branch)
2. `openagent_plugins/` — external installed plugins (user's workspace)

External plugins override bundled ones with the same name without warning.

---

## Plugin Lifecycle

1. **Discovery** — `_load_installed_plugins()` scans both directories.
2. **Import** — `_register_plugin_from_file()` imports the `.py` file dynamically.
3. **Registration** — `_register_plugin()` stores the plugin and its config defaults.
4. **Tool map merge** — `_get_tool_map()` merges core entries with each plugin's `tool_map`.
5. **Dispatch** — `_dispatch_tool()` looks up the handler either from the plugin or from the module itself.

---

## Plugin API

### Base class (`OpenAgentPlugin`)

```python
class OpenAgentPlugin:
    name: str = ""                    # unique plugin identifier
    version: str = "0.1.0"
    tool_registry: tuple[str, ...] = ()    # advertised tool names
    tool_map: dict[str, str] = {}          # tool name → handler method
    config_defaults: dict[str, object] = {}  # optional config keys

    def __init__(self, agent):
        self._agent = agent

    @property
    def agent(self):
        return self._agent

    async def on_load(self):
        """Called after registration (optional)."""
```

### Required fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Unique plugin name. Used as tool group prefix. |
| `tool_registry` | `tuple[str]` | Tool names advertised to the model. |
| `tool_map` | `dict[str, str]` | Maps each tool name → a handler method name (string). |

### Handler methods

Each handler is an `async` method on the plugin class.
The dispatch system inspects the method signature and passes arguments by name.

**Supported parameter names (any subset, order irrelevant):**

| Parameter | Value |
|-----------|-------|
| `tool_name` | The matched tool name (e.g. `"message.send"`) |
| `attrs_raw` | Raw XML attributes string from the tool call |
| `body` | The body text (after attributes) |
| `source_event` | The Telegram `NewMessage.Event` that triggered the conversation |
| `command` | Same as `body` (for terminal-like tools) |
| `query` | Same as `body` (for search-like tools) |
| `mode` | One of `private`/`groups`/`all` (for dialog listing) |
| `target` | Target user/chat name |
| `kind` | `"group"` or `"channel"` (for creation tools) |

**Example:**

```python
class MyPlugin:
    name = "myplugin"
    tool_map = {
        "myplugin.hello": "cmd_hello",
    }

    async def cmd_hello(self, body: str) -> str:
        return f"Hello, {body or 'world'}!"
```

### Config defaults

If your plugin needs runtime configuration:

```python
class TerminalPlugin:
    name = "terminal"
    config_defaults = {
        "terminal_timeout": 30,       # int
        "terminal_enabled": True,     # bool
        "terminal_steps": 3,          # int
    }
```

The loader creates proper `ConfigValue` entries with automatic type detection
(`Boolean` for `bool`, `Integer` for `int`, `Float` for `float`,
`List` for `list`, `String` for everything else).

### Lifecycle hooks

Plugins may optionally implement hook methods. Hooks are fully trusted: they run
inside the OpenAgent process and can see or mutate tool calls, prompts, messages,
results, and errors.

Hook order is controlled by `hook_priority` on the plugin class. Higher priority
runs first; equal priority keeps plugin registration order.

| Hook | When it runs | Typical use |
|------|--------------|-------------|
| `on_load()` | After plugin registration | Initialize state, patch methods, warm caches |
| `on_unload()` | When plugin is unregistered/disabled when possible | Cleanup resources; patches are restored automatically |
| `before_tool(context)` | Before tool resolution and dangerous-tool confirmation | Rewrite/cancel tool calls, enforce policy, aliases |
| `after_tool(context)` | After a tool succeeds, before result is shown | Post-process result, audit, redact output |
| `on_tool_error(context)` | After a tool handler raises | Replace error text, audit failures |
| `before_agent(context)` | Before provider-specific user content/messages are built | Rewrite prompt, attachments, provider/system override |
| `before_agent_messages(context)` | After provider messages are built, before model requests | Inject system/context messages, inspect final prompt |
| `after_agent(context)` | Before final answer is returned | Format/redact/append final answer |

#### Hook result

Hooks can either mutate the context object directly or return
`PluginHookResult`:

```python
from OpenAgentLib.PluginBase import PluginHookResult

return PluginHookResult(cancel=True, result="Tool was blocked")
```

Fields:

| Field | Meaning |
|-------|---------|
| `cancel` | Stop the current hook chain. In `before_*` hooks this can skip the operation. |
| `result` | Replacement tool result / agent answer. |
| `reason` | Human-readable fallback when cancelling without `result`. |

Returning a plain non-`None` value is treated like
`PluginHookResult(result=value)`. For `before_tool`, `before_agent`, and
`before_agent_messages`, plain return values do **not** cancel execution; use
`cancel=True` when you intentionally want to stop the operation.

#### Tool hook context

`before_tool`, `after_tool`, and `on_tool_error` receive `ToolHookContext`:

```python
context.agent          # OpenAgent instance
context.tool_name      # mutable tool name
context.attrs_raw      # mutable raw XML attrs
context.body           # mutable body
context.source_event   # Telegram source event
context.status_event   # status/edit event
context.agent_log      # current agent log list
context.started_at     # monotonic start timestamp or None
context.thinking_notes # thinking note list or None
context.plugin_owner   # resolved plugin owner after tool lookup
context.result         # tool result or HOOK_NO_RESULT
context.error          # exception in on_tool_error, else None
context.metadata       # free dict for cooperating hooks
```

Example alias/safety hook:

```python
from OpenAgentLib.PluginBase import PluginHookResult

class GuardPlugin:
    name = "guard"
    hook_priority = 100

    async def before_tool(self, context):
        if context.tool_name == "shell":
            context.tool_name = "terminal.run"

        if context.tool_name == "terminal.run" and "rm -rf" in context.body:
            return PluginHookResult(cancel=True, result="Blocked dangerous command")
```

`before_tool` runs before dangerous-tool confirmation. If a hook rewrites a safe
tool into a dangerous one, confirmation is checked against the rewritten
`tool_name` and `body`.

#### Agent hook context

`before_agent`, `before_agent_messages`, and `after_agent` receive
`AgentHookContext`:

```python
context.agent              # OpenAgent instance
context.prompt             # mutable user prompt
context.provider           # mutable provider key; normalized after before_agent
context.source_event       # Telegram source event
context.status_event       # status/edit event
context.attachments        # mutable list of attachment dicts
context.cancel_token       # generation cancel token
context.system_override    # mutable system prompt override
context.flash_mode         # bool
context.messages           # provider messages after they are built
context.thinking_messages  # thinking-pass messages after they are built
context.agent_log          # current agent log list
context.thinking_notes     # current thinking notes list
context.tool_trace         # collected tool trace messages
context.answer             # final answer in after_agent
context.metadata           # free dict for cooperating hooks
```

Example final-answer formatter:

```python
class FormatPlugin:
    name = "format"

    async def after_agent(self, context):
        context.answer = context.answer.strip()
        if context.answer and not context.answer.endswith("."):
            context.answer += "."
```

> Security note: `before_agent_messages` can see system prompts, history,
> thinking messages, tool traces, and user content. Only install trusted plugins.

### Method patch helpers

`OpenAgentPlugin` also provides helpers for temporary method/attribute patching.
All patches are tracked and restored in LIFO order by `restore_patches()`.
The plugin engine also restores them automatically during unload, even if a
custom `on_unload()` does not call `super().on_unload()`.

| Helper | Purpose |
|--------|---------|
| `patch_attr(target, name, value, create=False)` | Replace an attribute and remember the original value. |
| `patch_method(target, method_name, replacement, bind=None)` | Replace a method. Instance plain functions are auto-bound. |
| `patch_agent_method(method_name, replacement, bind=None)` | Shortcut for patching `self.agent`. |
| `wrap_method(target, method_name, wrapper, bind=None)` | Wrap an existing method with `wrapper(original, *args, **kwargs)`. |
| `wrap_agent_method(method_name, wrapper, bind=None)` | Shortcut for wrapping `self.agent`. |
| `restore_patch(patch)` | Restore one `MethodPatch`. |
| `restore_patches()` / `unpatch_all()` | Restore all patches registered by this plugin. |

Example: wrapping OpenAgent tool dispatch:

```python
class DispatchAuditPlugin:
    name = "dispatch_audit"

    async def on_load(self):
        self.wrap_agent_method("_dispatch_tool", self.audit_dispatch)

    async def audit_dispatch(self, original, *args, **kwargs):
        tool_name = args[0] if args else kwargs.get("name", "unknown")
        self.agent.log.info(f"Tool started: {tool_name}")
        try:
            return await original(*args, **kwargs)
        finally:
            self.agent.log.info(f"Tool finished: {tool_name}")
```

Example: temporarily replacing a method:

```python
class PatchExamplePlugin:
    name = "patch_example"

    async def on_load(self):
        async def replacement(agent_self, *args, **kwargs):
            return "patched result"

        self.patch_agent_method("some_agent_method", replacement)

    async def on_unload(self):
        # Optional: PluginEngine restores patches automatically too.
        self.restore_patches()
```

Patch behavior:

- Patching an instance method creates a temporary shadow attribute on that
  instance; restore removes the shadow and exposes the original class method.
- Patching a class method restores the original class attribute.
- `patch_attr(..., create=False)` requires the target attribute to exist and is
  safer against typos. Use `create=True` for temporary new attributes.
- `MethodPatch.restore()` is idempotent and returns `True` only on first restore.

---

## Installing plugins

### From reply (`.oaplugin` on a `.py` file)

1. Send or forward a `.py` plugin file to the chat.
2. Reply to it with `.oaplugin`.
3. The plugin is validated (compiled), saved to `openagent_plugins/`, and registered.

```text
You:  .oaplugin
      ↑ reply to file.py
Bot:  Plugin installed: myplugin
```

### From catalog (`📦 Каталог`)

1. Run `.oaplugin` → tap `📦 Каталог`.
2. Browse plugins from the repository.
3. Tap `📥 Установить`.

Installed plugins appear in `⚙️ Менеджер` where you can also delete them.

---

## Creating a plugin

Simplest plugin skeleton:

```python
# scop: inline
# SPDX-License-Identifier: MIT

from typing import Any


class PingPlugin:
    name = "ping"
    version = "0.1.0"
    description = "Simple ping tool"

    tool_registry = ("ping.check",)
    tool_map = {
        "ping": "cmd_ping",
        "ping.check": "cmd_ping",
    }

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    async def cmd_ping(self, body: str) -> str:
        return "pong"
```

Save it, reply with `.oaplugin`, done.

### Calling module internals

Plugins can access the agent's methods through `self.agent`:

```python
await self.agent._web_search(query)          # web search
await self.agent._run_mcub_command(cmd, ev)  # MCUB command
await self.agent._send_userbot_message(msg, ev, chat=...)  # send message
await self.agent._misc_tool(name, attrs, body, ev)         # misc Telegram ops
data = self.agent._parse_xml_attrs(attrs_raw)              # parse XML attributes
```

> ⚠️ Methods starting with `_` are not public API — they may change between versions.

---

## Bundled plugins

| File | Name | Tools | Description |
|------|------|-------|-------------|
| `terminal.py` | `terminal` | `terminal.run`, `.inspect`, `.list_files`, `.read_file`, `.git_status` | Shell commands |
| `ast_grep.py` | `ast_grep` | `ast_grep.search`, `.replace` | AST-based structural search and rewrites |
| `web.py` | `web` | `web.search`, `.fetch_url`, `.read_html`, `.extract_links`, `.summarize_page` | Web search/fetch |
| `mcub.py` | `mcub` | `mcub.command`, `.config`, `.modules`, `.install`, `.reload` | MCUB kernel commands |
| `message.py` | `message` | `message.send*`, `.reply`, `.edit`, `.forward`, `.delete`, `.pin`, `.react`, `.get`, `.search`, `.history`, `.mark_read`, `.typing`, `.schedule`, `.draft` | Telegram messaging |
| `dialog.py` | `dialog` | `dialog.list_*`, `.search`, `.archive`, `.unarchive`, `.leave`, `.export_invite`, `.get_photo`, `.set_photo` | Dialog management |
| `chat.py` | `chat` | `chat.info`, `.participants`, `.admins`, `.permissions`, `.common_with_user`, `.set_*`, `.slowmode`, `.invite_link` | Chat settings |
| `moderation.py` | `moderation` | `moderation.mute`, `.unmute`, `.ban`, `.unban`, `.kick`, `.promote`, `.demote`, `.pin`, `.delete_messages`, `.get_admins` | Moderation |
| `profile.py` | `profile` | `profile.get*`, `.update_*`, `.set_photo`, `.download_photo`, `.common_chats` | User profile |
| `file.py` | `file` | `file.send`, `.download_media`, `.read_text` | File operations |
| `contacts.py` | `contacts` | `contacts.add`, `.delete`, `.block`, `.unblock`, `.entity` | Contact management |
| `creation.py` | `creation` | `creation.channel`, `.group`, `.bot`, `.private_invite` | Channel/group/bot creation |

---

## Tool dispatch rules

When the model calls a tool:

1. **Plugin `tool_map` is checked first** — exact match by tool name.
2. **Core map is checked** — module-tied tools (skills, code, context, todo, utility, thinking).
3. **Misc aliases** — legacy shortcuts like `get_admins`, `edit_message`, `block_user` are routed to `_misc_tool`.
4. If nothing matches, the closest tool names are suggested.

The tool group (first segment before `.`) is used to find the owning plugin,
but the tool map lookup can match any key regardless of group.
