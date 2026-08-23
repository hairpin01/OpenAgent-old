"""Frozen migration contract for installed-plugin state consumers.

Entries are deliberately static: implementation tasks must replace each
consumer with the stated registry query or transition instead of extending a
second mutable view of installed plugins.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstalledPluginConsumer:
    path: str
    owner: str
    field: str
    access: str
    behavior: str
    replacement: str
    migration_task: str


LIFECYCLE_STATES = frozenset(
    {"admitted", "installed", "enabled", "active", "unloading", "failed", "disabled", "removed"}
)

PRESENTATION_FIELDS = frozenset(
    {"id", "manifest", "source", "path", "tools", "version", "author", "description", "permissions", "requirements"}
)


CONSUMERS = (
    InstalledPluginConsumer("Src/OpenAgentLib/Lifecycle.py", "_OpenAgentLifecycleMixin.on_load", "_plugins", "write", "Initializes the legacy instance map.", "registry.create()", "2"),
    InstalledPluginConsumer("Src/OpenAgentLib/Lifecycle.py", "_OpenAgentLifecycleMixin.on_load", "_plugin_files", "write", "Initializes the legacy filename map.", "registry.create()", "2"),
    InstalledPluginConsumer("Src/OpenAgentLib/Lifecycle.py", "_OpenAgentLifecycleMixin.on_load", "_disabled_plugins", "write", "Loads persisted disabled IDs before admission.", "registry.restore_enabled()", "3"),
    InstalledPluginConsumer("Src/OpenAgentLib/Lifecycle.py", "_OpenAgentLifecycleMixin.on_load", "_v2_source_event", "write", "Initializes request event callback context.", "registry.bind_action()", "7"),
    InstalledPluginConsumer("Src/OpenAgentLib/Lifecycle.py", "_OpenAgentLifecycleMixin.on_load", "_background_tool_tasks", "write", "Initializes unowned background tool tasks.", "registry.own_task()", "7"),
    InstalledPluginConsumer("Src/OpenAgentLib/Lifecycle.py", "_OpenAgentLifecycleMixin.on_load", "_plugin_unload_tasks", "write", "Initializes deferred unload task ownership.", "registry.own_task()", "7"),
    InstalledPluginConsumer("Src/OpenAgentLib/Lifecycle.py", "_OpenAgentLifecycleMixin.on_load", "_plugins_cache", "write", "Initializes the repository catalog cache.", "registry.catalog_snapshot()", "6"),
    InstalledPluginConsumer("Src/OpenAgentLib/Lifecycle.py", "_OpenAgentLifecycleMixin.on_load", "_v2_plugin_sources", "item_update", "Adds sibling static sources after installed-source admission.", "registry.install()", "2"),
    InstalledPluginConsumer("Src/OpenAgentLib/Lifecycle.py", "_OpenAgentLifecycleMixin.on_load", "_v2_plugin_sources", "iteration", "Passes admitted static sources to the invoker.", "registry.snapshot(state='active')", "7"),
    InstalledPluginConsumer("Src/OpenAgentLib/Lifecycle.py", "_OpenAgentLifecycleMixin.on_load", "_v2_plugin_invoker", "write", "Binds the isolated invoker to admitted sources.", "registry.bind_action()", "7"),
    InstalledPluginConsumer("Src/OpenAgentLib/Lifecycle.py", "_OpenAgentLifecycleMixin.on_unload", "_plugin_unload_tasks", "read", "Begins global plugin teardown.", "registry.begin_unload_all()", "7"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._load_disabled_plugins", "_disabled_plugins", "persistence_read", "Reads disabled IDs from disk for startup reconstruction.", "registry.restore_enabled()", "3"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._save_disabled_plugins", "_disabled_plugins", "indirect_read", "Persists disabled IDs.", "registry.snapshot(state='disabled')", "3"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._load_installed_plugins", "_v2_plugin_sources", "write", "Replaces the admitted-source collection during startup.", "registry.reconstruct()", "3"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._load_installed_plugins", "_disabled_plugins", "lookup", "Skips disabled bundled sources during admission.", "registry.get_enabled()", "3"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._inspect_installed_plugin_admission", "_disabled_plugins", "lookup", "Preserves persisted disabled operator intent during install preflight.", "registry.get()", "5"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._load_installed_plugins", "_v2_plugin_sources", "item_update", "Admits each inspected v2 source.", "registry.install()", "2"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._load_installed_plugins", "_v2_plugin_invoker", "indirect_read", "Binds an installed source before registry activation.", "registry.activate()", "3"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._register_plugin_from_file", "_v2_plugin_sources", "indirect_read", "Obtains the mutable admission source map through a compatibility wrapper.", "registry.get()", "5"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._register_plugin_from_file", "_v2_plugin_sources", "write", "Creates the source map when absent.", "registry.create()", "2"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._register_plugin_from_file", "_plugin_files", "indirect_read", "Obtains the legacy filename map through a compatibility wrapper.", "registry.get()", "5"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._register_plugin_from_file", "_plugin_files", "write", "Creates the filename map when absent.", "registry.create()", "2"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._register_plugin_from_file", "_v2_plugin_invoker", "indirect_read", "Finds the isolated invoker for source registration.", "registry.bind_action()", "7"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._register_plugin_from_file", "_v2_plugin_sources", "item_update", "Stages a replacement source before invoker registration.", "registry.replace()", "5"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._register_plugin_from_file", "_plugin_files", "item_update", "Stages the installed source path with its replacement.", "registry.replace()", "5"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._register_plugin_from_file", "_v2_plugin_sources", "remove", "Rolls back a failed staged source replacement.", "registry.replace()", "5"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._register_plugin_from_file", "_plugin_files", "remove", "Rolls back a failed staged filename replacement.", "registry.replace()", "5"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._register_plugin", "_plugins", "lookup", "Detects an existing legacy instance before overwrite.", "registry.get()", "2"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._register_plugin", "_plugins", "item_update", "Registers a legacy executable plugin instance.", "registry.install()", "2"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._unregister_plugin", "_plugins", "remove", "Removes the legacy executable instance.", "registry.begin_unload()", "7"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._unregister_plugin", "_plugin_files", "remove", "Removes the legacy source-path mapping.", "registry.remove()", "5"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._unregister_plugin", "_v2_plugin_sources", "indirect_read", "Removes v2 source state through a compatibility wrapper.", "registry.remove()", "5"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._unregister_plugin", "_v2_plugin_invoker", "indirect_read", "Unregisters the isolated invoker source.", "registry.complete_unload()", "7"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._iter_hook_plugins", "_plugins", "indirect_read", "Iterates legacy hook providers in priority order.", "registry.snapshot(state='active')", "7"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._get_plugin_for_tool", "_plugins", "iteration", "Finds a legacy tool handler by map or group.", "registry.get_active_tool_owner()", "6"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._get_plugin_for_tool", "_plugins", "lookup", "Finds a legacy tool handler by group key.", "registry.get_active_tool_owner()", "6"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._format_plugin_docs", "_plugins", "indirect_read", "Builds activated-plugin help and presentation metadata.", "registry.snapshot(state='active')", "6"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._install_plugin_from_code", "_plugin_files", "iteration", "Finds the installed ID after atomic source installation.", "registry.get_by_path()", "5"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._cancel_plugin_unload_tasks", "_plugin_unload_tasks", "iteration", "Cancels and reaps deferred unload work.", "registry.complete_unload()", "7"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._cancel_plugin_unload_tasks", "_plugin_unload_tasks", "remove", "Clears reaped unload task ownership.", "registry.complete_unload()", "7"),
    InstalledPluginConsumer("Src/OpenAgentMain.py", "OpenAgent._format_oaplugin_overview", "_plugins", "read", "Renders enabled plugin overview including ID, manifest presentation fields, and tools.", "registry.snapshot(state='active')", "6"),
    InstalledPluginConsumer("Src/OpenAgentMain.py", "OpenAgent._format_oaplugin_overview", "_plugins", "iteration", "Orders overview records and derives the displayed total.", "registry.snapshot(state='active')", "6"),
    InstalledPluginConsumer("Src/OpenAgentMain.py", "OpenAgent._oaplugin_catalog", "_plugins_cache", "read", "Reads the repository catalog cache.", "registry.catalog_snapshot()", "6"),
    InstalledPluginConsumer("Src/OpenAgentMain.py", "OpenAgent._oaplugin_catalog", "_plugins", "lookup", "Marks a catalog entry as installed from the legacy map.", "registry.get()", "6"),
    InstalledPluginConsumer("Src/OpenAgentMain.py", "OpenAgent._oaplugin_catalog", "_v2_plugin_sources", "indirect_read", "Marks a catalog entry as installed from admitted v2 state.", "registry.get()", "6"),
    InstalledPluginConsumer("Src/OpenAgentMain.py", "OpenAgent._oaplugin_manager", "_plugins", "iteration", "Lists installed records for manager/info presentation fields and delete action.", "registry.snapshot()", "6"),
    InstalledPluginConsumer("Src/OpenAgentMain.py", "OpenAgent._oaplugin_uninstall", "_plugin_files", "lookup", "Uses a source path to distinguish bundled disable from removable install.", "registry.get()", "5"),
    InstalledPluginConsumer("Src/OpenAgentMain.py", "OpenAgent._oaplugin_uninstall", "_disabled_plugins", "item_update", "Disables a bundled plugin persistently.", "registry.set_enabled(False)", "7"),
    InstalledPluginConsumer("Src/OpenAgentMain.py", "OpenAgent._oaplugin_uninstall", "_plugins", "read", "Bounds manager pagination after delete or disable.", "registry.snapshot()", "6"),
    InstalledPluginConsumer("Src/OpenAgentMain.py", "OpenAgent.bot_oa", "_background_tool_tasks", "read", "Reports the currently owned background task set.", "registry.own_task()", "7"),
    InstalledPluginConsumer("Src/OpenAgentMain.py", "OpenAgent.on_unload", "_background_tool_tasks", "iteration", "Cancels module-owned background work at shutdown.", "registry.own_task()", "7"),
    InstalledPluginConsumer("Src/OpenAgentMain.py", "OpenAgent.on_unload", "_plugin_unload_tasks", "iteration", "Cancels module-owned plugin teardown work at shutdown.", "registry.complete_unload()", "7"),
    InstalledPluginConsumer("Src/OpenAgentLib/ToolDispatch.py", "_OpenAgentToolRegistryMixin._retired_legacy_names", "_plugins", "read", "Builds the retired legacy executable tool map.", "registry.snapshot(state='active')", "8"),
    InstalledPluginConsumer("Src/OpenAgentLib/ToolDispatch.py", "_OpenAgentRuntimeToolsMixin._active_plugins_prompt", "_plugins", "indirect_read", "Adds active plugin names to the model prompt.", "registry.snapshot(state='active')", "6"),
    InstalledPluginConsumer("Src/OpenAgentLib/ResponseAgent.py", "_OpenAgentResponseMixin.create_background_tool_task", "_background_tool_tasks", "item_update", "Owns a background tool task until completion.", "registry.own_task()", "7"),
    InstalledPluginConsumer("Src/OpenAgentLib/ResponseAgent.py", "_OpenAgentResponseMixin.create_background_tool_task.runner", "_background_tool_tasks", "remove", "Releases a completed background task.", "registry.own_task()", "7"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._schedule_plugin_unload", "_plugin_unload_tasks", "item_update", "Tracks asynchronous plugin teardown until completion.", "registry.own_task()", "7"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._fetch_repo_plugins", "_plugins_cache", "write", "Refreshes the repository catalog cache.", "registry.catalog_snapshot()", "6"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._register_plugin_from_file", "_plugins_cache", "write", "Invalidates the repository catalog cache after source replacement.", "registry.catalog_snapshot()", "5"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentAgentLoopMixin._dispatch_agent_tool_batch", "_v2_source_event", "write", "Binds the source event around a v2 tool batch.", "registry.bind_action()", "7"),
    InstalledPluginConsumer("Src/OpenAgentLib/Lifecycle.py", "_OpenAgentLifecycleMixin.on_unload", "_v2_plugin_invoker", "indirect_read", "Quiesces owned isolated calls before runtime teardown.", "registry.release_task()", "7"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._persist_disabled_plugins", "_disabled_plugins", "persistence_read", "Atomically persists canonical disabled IDs.", "registry.set_enabled(False)", "7"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._set_installed_plugin_enabled", "_disabled_plugins", "item_update", "Commits or rolls back persisted disabled intent.", "registry.set_enabled()", "7"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._set_installed_plugin_enabled", "_plugin_files", "item_update", "Synchronizes executable source paths with state transitions.", "registry.activate()", "7"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._set_installed_plugin_enabled", "_v2_plugin_invoker", "indirect_read", "Quiesces and rebinds isolated source execution.", "registry.own_task()", "7"),
    InstalledPluginConsumer("Src/OpenAgentLib/Plugin/PluginsEngine.py", "_OpenAgentPluginSkillMixin._set_installed_plugin_enabled", "_v2_plugin_sources", "item_update", "Synchronizes admitted sources with enable state.", "registry.activate()", "7"),
    InstalledPluginConsumer("Src/OpenAgentMain.py", "OpenAgent._oaplugin_uninstall", "_v2_plugin_invoker", "indirect_read", "Quiesces isolated invocations before removal.", "registry.remove_active()", "7"),
)
