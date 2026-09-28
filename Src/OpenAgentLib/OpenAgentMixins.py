# SPDX-License-Identifier: MIT
# Auto-loaded OpenAgent runtime mixins for MCUB.
# Source: https://github.com/hairpin01/repo-MCUB-fork/tree/main/build/OpenAgent/src

from __future__ import annotations

from .ContextService import (
    OpenAgentContextService,
    _OpenAgentContextMixin,
)
from .Lifecycle import _OpenAgentLifecycleMixin
from .Manager import Session as Session
from .Manager.OASession import OASession
from .Manager.Session import (
    _SESSION_PREFERENCES,
    _OpenAgentSessionsMixin,
)
from .Placeholders import (
    _PLACEHOLDER_RE,
    OpenAgentProviderService,
    OpenAgentTemplateService,
    _OpenAgentProviderMixin,
)
from .Plugin.PluginBase import OpenAgentPlugin
from .Plugin.PluginsEngine import (
    _OpenAgentAgentLoopMixin,
    _OpenAgentPluginSkillMixin,
    _OpenAgentStatusMixin,
    _OpenAgentTelegramMediaMixin,
)
from .ResponseAgent import _OpenAgentResponseMixin
from .TodoService import (
    _DEFAULT_TODO_STATUS_MAP,
    _TODO_STATUS_ALIASES,
    _WHITESPACE_RE,
    OpenAgentTodoService,
    _OpenAgentTodoMixin,
)
from .ToolDispatch import (
    _DEFAULT_TOOL_STATUS_EMOJIS,
    _TOOL_GROUP_ALIASES,
    OpenAgentToolDisplayService,
    _OpenAgentRuntimeToolsMixin,
    _OpenAgentToolDisplayMixin,
    _OpenAgentToolRegistryMixin,
)

OPENAGENT_LIB_VERSION = "0.8.1-main.build:1054"  # fallback

__all__ = [
    "OPENAGENT_LIB_VERSION",
    "_DEFAULT_TODO_STATUS_MAP",
    "_DEFAULT_TOOL_STATUS_EMOJIS",
    "_PLACEHOLDER_RE",
    "_SESSION_PREFERENCES",
    "_TODO_STATUS_ALIASES",
    "_TOOL_GROUP_ALIASES",
    "_WHITESPACE_RE",
    "OASession",
    "OpenAgentContextService",
    "OpenAgentPlugin",
    "OpenAgentProviderService",
    "OpenAgentTemplateService",
    "OpenAgentTodoService",
    "OpenAgentToolDisplayService",
    "Session",
    "_OpenAgentAgentLoopMixin",
    "_OpenAgentContextMixin",
    "_OpenAgentLifecycleMixin",
    "_OpenAgentPluginSkillMixin",
    "_OpenAgentProviderMixin",
    "_OpenAgentResponseMixin",
    "_OpenAgentRuntimeToolsMixin",
    "_OpenAgentSessionsMixin",
    "_OpenAgentStatusMixin",
    "_OpenAgentTelegramMediaMixin",
    "_OpenAgentTodoMixin",
    "_OpenAgentToolDisplayMixin",
    "_OpenAgentToolRegistryMixin",
]
