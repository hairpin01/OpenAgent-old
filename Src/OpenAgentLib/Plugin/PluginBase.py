# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass, field
from types import MethodType
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.lib.types import Kernel

HOOK_NO_RESULT = object()

__all__ = [
    "AgentHookContext",
    "HOOK_NO_RESULT",
    "MethodPatch",
    "OpenAgentPlugin",
    "PluginHookResult",
    "ToolHookContext",
]


@dataclass
class PluginHookResult:
    """Optional return value for plugin hooks.

    Hooks may also mutate the context object directly. Use ``cancel=True`` to
    stop the current operation, and ``result=...`` to replace the user-facing
    result/answer.
    """

    cancel: bool = False
    result: Any = HOOK_NO_RESULT
    reason: str = ""

    @property
    def has_result(self) -> bool:
        return self.result is not HOOK_NO_RESULT


@dataclass
class ToolHookContext:
    """Mutable context passed to plugin tool hooks."""

    agent: Any
    tool_name: str
    attrs_raw: str = ""
    body: str = ""
    source_event: Any | None = None
    status_event: Any | None = None
    agent_log: list[str] = field(default_factory=list)
    started_at: float | None = None
    thinking_notes: list[str] | None = None
    plugin_owner: Any | None = None
    result: Any = HOOK_NO_RESULT
    error: BaseException | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentHookContext:
    """Mutable context passed to plugin agent hooks."""

    agent: Any
    prompt: str
    provider: str
    source_event: Any | None = None
    status_event: Any | None = None
    attachments: list[dict[str, str]] = field(default_factory=list)
    cancel_token: str | None = None
    system_override: str | None = None
    flash_mode: bool = False
    messages: list[dict[str, Any]] = field(default_factory=list)
    thinking_messages: list[dict[str, Any]] = field(default_factory=list)
    agent_log: list[str] = field(default_factory=list)
    thinking_notes: list[str] = field(default_factory=list)
    tool_trace: list[dict[str, str]] = field(default_factory=list)
    answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodPatch:
    """Handle returned by plugin patch helpers.

    Restore patches in reverse order with ``plugin.restore_patches()`` or one by
    one with ``plugin.restore_patch(patch)``.
    """

    target: Any
    name: str
    original: Any = HOOK_NO_RESULT
    had_original: bool = True
    active: bool = True

    def restore(self) -> bool:
        """Restore the original attribute/method. Returns True on first restore."""
        if not self.active:
            return False
        if self.had_original:
            setattr(self.target, self.name, self.original)
        else:
            try:
                delattr(self.target, self.name)
            except AttributeError:
                pass
        self.active = False
        return True


class OpenAgentPlugin:
    """Base class for OpenAgent plugins.

    Keep this module small and dependency-light: user plugins can import the
    public base class without pulling in the whole plugin/skill engine.
    """

    name: str = ""
    version: str = "0.1.0"
    author: str = ""
    description: str = ""
    manifest: dict[str, Any] = {}
    permissions: tuple[str, ...] = ()
    requirements: tuple[str, ...] = ()
    tool_registry: tuple[str, ...] = ()
    tool_map: dict[str, str] = {}
    tool_docs: dict[str, dict[str, str]] = {}
    tool_schemas: dict[str, dict[str, Any]] = {}
    dangerous_tools: set[str] = set()
    config_defaults: dict[str, object] = {}
    hook_priority: int = 0

    def __init__(self, agent: Any) -> None:
        self._agent = agent
        self.kernel: "Kernel" = self._agent.kernel
        self.client = self._agent.client
        self._method_patches: list[MethodPatch] = []

    @property
    def agent(self) -> Any:
        return self._agent

    def add_runtime_comment(self, runtime_token: str | None, comment: str) -> bool:
        """Queue a live comment for the current OpenAgent run."""
        return self._agent.add_runtime_comment(runtime_token, comment)

    def create_background_tool_task(
        self,
        *,
        tool_name: str,
        attrs_raw: str = "",
        body: str = "",
        source_event: Any | None = None,
        status_event: Any | None = None,
        runtime_token: str | None = None,
        label: str = "",
    ) -> str:
        """Run an OpenAgent tool in background and comment when it finishes."""
        return self._agent.create_background_tool_task(
            tool_name=tool_name,
            attrs_raw=attrs_raw,
            body=body,
            source_event=source_event,
            status_event=status_event,
            runtime_token=runtime_token,
            label=label,
        )

    def _patch_stack(self) -> list[MethodPatch]:
        patches = getattr(self, "_method_patches", None)
        if not isinstance(patches, list):
            patches = []
            self._method_patches = patches
        return patches

    def _should_bind_patch(
        self, target: Any, replacement: Any, bind: bool | None
    ) -> bool:
        if bind is not None:
            return bind
        if isinstance(target, type):
            return False
        return getattr(replacement, "__self__", None) is None

    def _log_patch_restore_error(self, patch: MethodPatch, exc: BaseException) -> None:
        logger = getattr(getattr(self, "_agent", None), "log", None)
        warning = getattr(logger, "warning", None)
        if callable(warning):
            warning(
                f"OpenAgent plugin patch restore failed: "
                f"{patch.target!r}.{patch.name}: {exc}"
            )

    def _own_attr_value(self, target: Any, name: str) -> tuple[bool, Any]:
        try:
            attrs = vars(target)
        except TypeError:
            return True, getattr(target, name)
        if name in attrs:
            return True, attrs[name]
        return False, HOOK_NO_RESULT

    def patch_attr(
        self,
        target: Any,
        name: str,
        value: Any,
        *,
        create: bool = False,
    ) -> MethodPatch:
        """Patch any attribute and remember how to restore it.

        ``create=False`` protects against typos by requiring the attribute to
        exist. Use ``create=True`` when intentionally adding a temporary attr.
        """
        attr_name = str(name or "").strip()
        if not attr_name:
            raise ValueError("patch attribute name is required")
        try:
            getattr(target, attr_name)
        except AttributeError:
            if not create:
                raise
            original = HOOK_NO_RESULT
            had_original = False
        else:
            had_original, original = self._own_attr_value(target, attr_name)
        patch = MethodPatch(
            target=target,
            name=attr_name,
            original=original,
            had_original=had_original,
        )
        setattr(target, attr_name, value)
        self._patch_stack().append(patch)
        return patch

    def patch_method(
        self,
        target: Any,
        method_name: str,
        replacement: Any,
        *,
        bind: bool | None = None,
    ) -> MethodPatch:
        """Replace a method and automatically restore it on unload.

        For class targets, pass a plain function. For instance targets, plain
        functions are bound to the target automatically; already-bound methods
        are installed as-is.
        """
        value = replacement
        if self._should_bind_patch(target, replacement, bind):
            value = MethodType(replacement, target)
        return self.patch_attr(target, method_name, value)

    def patch_agent_method(
        self,
        method_name: str,
        replacement: Any,
        *,
        bind: bool | None = None,
    ) -> MethodPatch:
        """Convenience wrapper for patching a method on ``self.agent``."""
        return self.patch_method(self.agent, method_name, replacement, bind=bind)

    def wrap_method(
        self,
        target: Any,
        method_name: str,
        wrapper: Any,
        *,
        bind: bool | None = None,
    ) -> MethodPatch:
        """Wrap an existing method with ``wrapper(original, *args, **kwargs)``.

        ``original`` is the method as it existed before patching. For instance
        targets the generated wrapper is bound automatically, and ``original`` is
        already bound, so wrappers can call ``original(*args, **kwargs)``.
        """
        original = getattr(target, method_name)
        should_bind = self._should_bind_patch(target, wrapper, bind)

        if should_bind:

            def patched(_patched_self: Any, *args: Any, **kwargs: Any) -> Any:
                return wrapper(original, *args, **kwargs)

        else:

            def patched(*args: Any, **kwargs: Any) -> Any:
                return wrapper(original, *args, **kwargs)

        patched.__name__ = f"patched_{method_name}"
        patched.__doc__ = getattr(wrapper, "__doc__", None)
        return self.patch_method(target, method_name, patched, bind=should_bind)

    def wrap_agent_method(
        self,
        method_name: str,
        wrapper: Any,
        *,
        bind: bool | None = None,
    ) -> MethodPatch:
        """Convenience wrapper for ``wrap_method(self.agent, ...)``."""
        return self.wrap_method(self.agent, method_name, wrapper, bind=bind)

    def restore_patch(self, patch: MethodPatch) -> bool:
        """Restore one patch handle and remove it from the plugin stack."""
        try:
            restored = patch.restore()
        except Exception as exc:
            self._log_patch_restore_error(patch, exc)
            return False
        patches = self._patch_stack()
        if patch in patches:
            patches.remove(patch)
        return restored

    def restore_patches(self) -> bool:
        """Restore all patches registered by this plugin in LIFO order."""
        ok = True
        patches = self._patch_stack()
        while patches:
            patch = patches.pop()
            try:
                patch.restore()
            except Exception as exc:
                ok = False
                self._log_patch_restore_error(patch, exc)
        return ok

    unpatch_all = restore_patches

    async def on_load(self) -> None:
        """Called after plugin is registered."""
        pass

    async def on_unload(self) -> None:
        """Called when plugin is unregistered/disabled when possible."""
        self.restore_patches()

    async def before_tool(self, context: ToolHookContext) -> PluginHookResult | None:
        """Called before any OpenAgent tool is executed.

        Mutate ``context.tool_name``, ``context.attrs_raw`` or ``context.body``
        to rewrite the tool call. Return ``PluginHookResult(cancel=True,
        result="...")`` to skip execution and provide a result.
        """
        return None

    async def after_tool(self, context: ToolHookContext) -> PluginHookResult | None:
        """Called after a tool succeeds; mutate/replace ``context.result``."""
        return None

    async def on_tool_error(self, context: ToolHookContext) -> PluginHookResult | None:
        """Called after a tool raises; may replace the formatted error result."""
        return None

    async def before_agent(self, context: AgentHookContext) -> PluginHookResult | None:
        """Called before prompt attachments are converted into provider messages."""
        return None

    async def before_agent_messages(
        self, context: AgentHookContext
    ) -> PluginHookResult | None:
        """Called after provider messages are built, before provider requests."""
        return None

    async def after_agent(self, context: AgentHookContext) -> PluginHookResult | None:
        """Called before the final answer is returned to the user."""
        return None
