# SPDX-License-Identifier: MIT
"""Concrete native v2 services backed by the OpenAgent runtime."""

from __future__ import annotations

import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .AgentRuntime import search_tool_docs
from .ToolKernel import ToolRegistry

JsonObject = Mapping[str, Any]


class RuntimeNativeSystemServices:
    """Adapt OpenAgent's concrete state and services to native v2 operations.

    Each public method maps one frozen canonical system operation.  This keeps
    v2 execution independent of the legacy attrs/body tool registries while
    preserving their externally visible result text and state changes.
    """

    def __init__(self, app: Any) -> None:
        self._app = app
        self._registry: ToolRegistry | None = None

    def bind_registry(self, registry: ToolRegistry) -> None:
        """Supply the final registry used by the utility documentation tools."""

        self._registry = registry

    @staticmethod
    def _result(value: Any) -> dict[str, str]:
        return {"result": str(value)}

    def _source_event(self) -> Any | None:
        return self._app._v2_source_event

    def _chat_id(self) -> int | None:
        source_event = self._source_event()
        if source_event is None:
            return None
        chat_id = self._app._event_chat_id(source_event)
        return int(chat_id) if chat_id is not None else None

    def _active_session(self, chat_id: int | None) -> Any | None:
        if chat_id is None:
            return None
        active_id = self._app._active_session.get(chat_id)
        return self._app._sessions.get(active_id) if active_id else None

    def _native_tool_docs(self) -> dict[str, dict[str, str]]:
        if self._registry is None:
            raise RuntimeError("native service registry has not been bound")
        return {
            spec.canonical_id: {
                "tool": spec.canonical_id,
                "desc": spec.description,
                "args": "JSON object matching the declared schema",
                "source": spec.source_family,
            }
            for spec in self._registry.specs()
        }

    @staticmethod
    def _todo_target(arguments: JsonObject) -> dict[str, str]:
        value = str(arguments.get("id", "") or "")
        return {"id": value}

    async def _prune_context(self, arguments: JsonObject) -> dict[str, str]:
        targets = {
            str(item).strip().lower()
            for item in arguments.get("targets", ["all"])
            if str(item).strip()
        }
        if "all" in targets:
            targets.update({"history", "tools", "tool_memory", "runtime_comments"})
        keep = max(0, int(arguments.get("keep", 0) or 0))
        changed: list[str] = []
        chat_id = self._chat_id()
        session = self._active_session(chat_id)

        if "history" in targets:
            if session is None:
                changed.append("history:no active session")
            elif keep:
                del session.messages[:-keep]
                self._app._touch_session(session)
                changed.append(f"history:{len(session.messages)} kept")
            else:
                session.messages.clear()
                self._app._touch_session(session)
                changed.append("history:0 kept")

        if {"tools", "tool_trace", "tool_outputs"} & targets:
            if session is None:
                changed.append("tools:no active session")
            else:
                before = len(session.messages)
                session.messages = [
                    message
                    for message in session.messages
                    if "OpenAgent tool trace:" not in str(message.get("content", ""))
                    and "Tool <" not in str(message.get("content", ""))
                ]
                self._app._touch_session(session)
                changed.append(f"tools:{before - len(session.messages)} removed")

        if {"tool_memory", "memory"} & targets:
            if chat_id is None:
                changed.append("tool_memory:no chat")
            else:
                removed = len(self._app._tool_memory.pop(chat_id, []))
                changed.append(f"tool_memory:{removed} removed")

        if {"runtime_comments", "comments"} & targets:
            token = self._app._placeholder_context.get("cancel_token")
            if token:
                removed = len(self._app._runtime_comments.pop(str(token), []))
            else:
                removed = sum(
                    len(items) for items in self._app._runtime_comments.values()
                )
                self._app._runtime_comments.clear()
            changed.append(f"runtime_comments:{removed} removed")

        return self._result(
            "Context prune complete: "
            + ("; ".join(changed) if changed else "nothing matched")
        )

    async def code_attach_result(self, arguments: JsonObject) -> JsonObject:
        del arguments
        latest = self._app._last_generated_file
        if not latest:
            return self._result("No generated file is available to attach")
        filename = self._app._safe_generated_filename(
            str(latest.get("name") or "generated.py")
        )
        content = str(latest.get("content") or "")
        if not content:
            return self._result("Generated file is empty")
        source_event = self._source_event()
        if source_event is None:
            return self._result(
                f"Generated file ready: {filename} ({len(content)} chars)"
            )
        try:
            data = io.BytesIO(content.encode("utf-8"))
            data.name = filename
            await self._app.client.send_file(
                self._app._event_chat_id(source_event) or source_event,
                data,
                caption=f"Generated file: {filename}",
            )
        except Exception as exc:
            return self._result(f"Attach failed: {exc}")
        return self._result(f"Generated file attached: {filename}")

    async def code_choose_filename(self, arguments: JsonObject) -> JsonObject:
        filename = self._app._safe_generated_filename(str(arguments["name"]))
        self._app._last_generated_file = {"name": filename, "content": ""}
        return self._result(filename)

    async def code_generate_file(self, arguments: JsonObject) -> JsonObject:
        filename = self._app._safe_generated_filename(str(arguments["path"]))
        content = str(arguments["content"])
        if not content.strip():
            return self._result("File content is required in tool body")
        self._app._last_generated_file = {"name": filename, "content": content}
        return self._result(
            f"Generated file prepared: {filename} ({len(content)} chars)"
        )

    async def code_generate_mcub_module(self, arguments: JsonObject) -> JsonObject:
        filename = self._app._safe_generated_filename(str(arguments["name"]))
        if not filename.endswith(".py"):
            filename = self._app._safe_generated_filename(f"{Path(filename).stem}.py")
        content = str(arguments["content"])
        if not content.strip():
            return self._result("MCUB module code is required in tool body")
        self._app._last_generated_file = {"name": filename, "content": content}
        return self._result(f"MCUB module prepared: {filename} ({len(content)} chars)")

    async def code_read_docs(self, arguments: JsonObject) -> JsonObject:
        del arguments
        return self._result(await self._app._fetch_mcub_docs())

    async def context_clear(self, arguments: JsonObject) -> JsonObject:
        del arguments
        chat_id = self._chat_id()
        if chat_id is not None:
            session = self._app._get_active_session(chat_id)
            session.messages.clear()
            self._app._touch_session(session)
            self._app._tool_memory.pop(chat_id, None)
        return self._result("Context cleared")

    async def context_discard(self, arguments: JsonObject) -> JsonObject:
        return await self._prune_context(arguments)

    async def context_media_context(self, arguments: JsonObject) -> JsonObject:
        del arguments
        source_event = self._source_event()
        if source_event is None:
            return self._result("No reply/media context available")
        reply_context, _attachments = await self._app._reply_context(source_event)
        return self._result(reply_context or "No reply/media context available")

    async def context_prune(self, arguments: JsonObject) -> JsonObject:
        return await self._prune_context(arguments)

    async def context_regenerate(self, arguments: JsonObject) -> JsonObject:
        del arguments
        return self._result(
            "Use the regenerate button under the last OpenAgent response"
        )

    async def context_remember(self, arguments: JsonObject) -> JsonObject:
        chat_id = self._chat_id()
        if chat_id is None:
            return self._result("No chat context available")
        self._app._remember_context(chat_id, "Memory note", str(arguments["note"]))
        return self._result("Remembered in current chat context")

    async def context_reply_context(self, arguments: JsonObject) -> JsonObject:
        del arguments
        source_event = self._source_event()
        if source_event is None:
            return self._result("No reply/media context available")
        reply_context, _attachments = await self._app._reply_context(source_event)
        return self._result(reply_context or "No reply/media context available")

    async def context_tool_output(self, arguments: JsonObject) -> JsonObject:
        chat_id = self._chat_id()
        latest = bool(arguments.get("latest", False))
        selector = str(arguments.get("path", "") or "")
        path = (
            self._app._latest_tool_output_path(chat_id)
            if latest
            else self._app._resolve_tool_output_path(selector, chat_id)
        )
        if path is None:
            return self._result(
                'No saved tool output found. Pass path="..." from the tool trace, or latest="true".'
            )
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return self._result(f"Failed to read saved tool output: {exc}")
        mode = str(arguments.get("mode", "head") or "head")
        limit = max(0, int(arguments.get("limit", 12000) or 0))
        offset = max(0, int(arguments.get("offset", 0) or 0))
        if mode == "all":
            shown = text[offset:] if offset else text
            truncated = False
        elif mode == "tail":
            shown = text[-limit:] if limit else ""
            truncated = len(text) > len(shown)
        else:
            shown = text[offset : offset + limit] if limit else ""
            truncated = len(text) > offset + len(shown)
        header = (
            f"Saved tool output: {path}\n"
            f"Total chars: {len(text)}\n"
            f"Mode: {mode}; offset: {offset}; returned chars: {len(shown)}"
        )
        if truncated:
            header += (
                '\n[truncated: re-call context.tool_output with mode="tail", '
                'mode="all", larger limit, or offset]'
            )
        return self._result(f"{header}\n\n{shown}")

    async def skill_save(self, arguments: JsonObject) -> JsonObject:
        saved = self._app._save_skill(str(arguments["name"]), str(arguments["content"]))
        return self._result(self._app.strings("skill_saved", name=saved))

    async def skills_activate(self, arguments: JsonObject) -> JsonObject:
        query = str(arguments.get("query", "") or "")
        return self._result(self._app._activate_skill_text(query))

    async def skills_export_md(self, arguments: JsonObject) -> JsonObject:
        name = str(arguments.get("name", "") or "")
        if not name:
            return self._result(self._app.strings("skill_name_required"))
        path = self._app._find_skill_path(name)
        if not path.exists():
            return self._result(self._app.strings("skill_not_found"))
        return self._result(path.read_text(encoding="utf-8", errors="replace")[:12000])

    async def skills_import_md(self, arguments: JsonObject) -> JsonObject:
        saved = self._app._save_skill(str(arguments["name"]), str(arguments["content"]))
        return self._result(self._app.strings("skill_saved", name=saved))

    async def skills_install(self, arguments: JsonObject) -> JsonObject:
        name = str(arguments.get("name", "") or "")
        if not name:
            return self._result(self._app.strings("skill_name_required"))
        saved = await self._app._install_repo_skill(name)
        return self._result(self._app.strings("skill_installed", name=saved))

    async def skills_list(self, arguments: JsonObject) -> JsonObject:
        del arguments
        names = [
            self._app._skill_name_from_path(path) for path in self._app._list_skills()
        ]
        return self._result("\n".join(names) or self._app.strings("skills_empty"))

    async def skills_read(self, arguments: JsonObject) -> JsonObject:
        name = str(arguments.get("name", "") or "")
        if not name:
            return self._result(self._app.strings("skill_name_required"))
        path = self._app._find_skill_path(name)
        if not path.exists():
            return self._result(self._app.strings("skill_not_found"))
        return self._result(path.read_text(encoding="utf-8", errors="replace")[:12000])

    async def skills_repo_list(self, arguments: JsonObject) -> JsonObject:
        del arguments
        return self._result(await self._app._format_skill_repo_list())

    async def skills_save_from_ai(self, arguments: JsonObject) -> JsonObject:
        saved = self._app._save_skill(str(arguments["name"]), str(arguments["content"]))
        return self._result(self._app.strings("skill_saved", name=saved))

    async def thinking_note(self, arguments: JsonObject) -> JsonObject:
        note = str(arguments["text"]).strip()
        if not note:
            return self._result("Thinking note recorded.")
        return self._result("Thinking note: " + note[:1200])

    async def todo_add(self, arguments: JsonObject) -> JsonObject:
        items = self._app._todo_items()
        text = self._app._todo_parse_html_text(str(arguments["text"]))
        if not text:
            return self._result("todo text is required")
        items.append({"text": text[:500], "status": "pending"})
        await self._app._save_todo_items(items)
        return self._result("TODO item added\n" + self._app._format_todo_placeholder())

    async def todo_clear(self, arguments: JsonObject) -> JsonObject:
        del arguments
        if not self._app._todo_items():
            return self._result("TODO list is already empty")
        await self._app._save_todo_items([])
        return self._result("TODO list cleared")

    async def todo_close(self, arguments: JsonObject) -> JsonObject:
        items = self._app._todo_items()
        index, error = self._app._todo_target_index(
            items, self._todo_target(arguments), ""
        )
        if index is None:
            return self._result(error)
        items[index]["status"] = "closed"
        await self._app._save_todo_items(items)
        return self._result(
            f"TODO closed: {items[index]['text']}\n"
            + self._app._format_todo_placeholder()
        )

    async def todo_closeall(self, arguments: JsonObject) -> JsonObject:
        del arguments
        items = self._app._todo_items()
        if not items:
            return self._result("TODO list is empty")
        for item in items:
            item["status"] = "closed"
        await self._app._save_todo_items(items)
        return self._result(
            "All TODO items closed\n" + self._app._format_todo_placeholder()
        )

    async def todo_current(self, arguments: JsonObject) -> JsonObject:
        del arguments
        items = self._app._todo_items()
        index, error = self._app._todo_target_index(items, {}, "")
        if index is None:
            return self._result(error)
        for item_index, item in enumerate(items):
            if item.get("status") == "open" and item_index != index:
                item["status"] = "pending"
        items[index]["status"] = "open"
        await self._app._save_todo_items(items)
        return self._result(
            f"Current TODO: {items[index]['text']}\n"
            + self._app._format_todo_placeholder()
        )

    async def todo_delete(self, arguments: JsonObject) -> JsonObject:
        items = self._app._todo_items()
        index, error = self._app._todo_target_index(
            items, self._todo_target(arguments), ""
        )
        if index is None:
            return self._result(error)
        removed = items.pop(index)
        await self._app._save_todo_items(items)
        return self._result(
            f"TODO deleted: {removed['text']}\n" + self._app._format_todo_placeholder()
        )

    async def todo_edit(self, arguments: JsonObject) -> JsonObject:
        items = self._app._todo_items()
        index, error = self._app._todo_target_index(
            items, self._todo_target(arguments), ""
        )
        if index is None:
            return self._result(error)
        text = self._app._todo_parse_html_text(str(arguments["text"]))
        if not text:
            return self._result("new todo text is empty")
        items[index]["text"] = text[:500]
        if "status" in arguments:
            items[index]["status"] = self._app._todo_normalize_status(
                str(arguments["status"])
            )
        await self._app._save_todo_items(items)
        return self._result(
            f"TODO updated: {items[index]['text']}\n"
            + self._app._format_todo_placeholder()
        )

    async def utility_agent_log(self, arguments: JsonObject) -> JsonObject:
        del arguments
        return self._result(
            "Agent log is shown under the final answer when tools are used"
        )

    async def utility_error_file(self, arguments: JsonObject) -> JsonObject:
        del arguments
        return self._result("Errors are reported through the MCUB kernel error handler")

    async def utility_list_tools(self, arguments: JsonObject) -> JsonObject:
        del arguments
        docs = self._native_tool_docs()
        groups: dict[str, list[str]] = {}
        for tool_name in sorted(docs):
            groups.setdefault(tool_name.split(".", 1)[0], []).append(tool_name)
        lines = ["Available tools by category:"]
        for group in sorted(groups):
            names = groups[group]
            lines.append(f"\n{group} ({len(names)}):")
            lines.extend(f"  - {name}: {docs[name]['desc']}" for name in names)
        lines.append(
            "\nTip: call utility.tool_help with a tool name for arguments and output."
        )
        return self._result("\n".join(lines)[:9000])

    async def utility_placeholders(self, arguments: JsonObject) -> JsonObject:
        del arguments
        return self._result(self._app._format_placeholders())

    async def utility_plugin_docs(self, arguments: JsonObject) -> JsonObject:
        query = str(arguments.get("plugin", "") or "")
        return self._result(self._app._format_plugin_docs(query or None))

    async def utility_random_template(self, arguments: JsonObject) -> JsonObject:
        del arguments
        return self._result(self._app._thinking_text())

    async def utility_search_tool(self, arguments: JsonObject) -> JsonObject:
        query = str(arguments.get("query", "") or "")
        if not query:
            return self._result(
                "Specify what the tool should do, e.g. query='run a shell command'"
            )
        docs = self._native_tool_docs()
        matches = search_tool_docs(query, docs, limit=8)
        if not matches:
            return self._result(f"No tools matched: {query}")
        lines = [f"Tool matches for: {query}"]
        for name in matches:
            lines.append(f"{name}: {docs[name]['desc']}")
        return self._result("\n\n".join(lines))

    async def utility_token_usage(self, arguments: JsonObject) -> JsonObject:
        del arguments
        usage = self._app._last_token_usage
        return self._result(
            "\n".join(f"{key}: {value}" for key, value in usage.items())
        )

    async def utility_tool_help(self, arguments: JsonObject) -> JsonObject:
        query = str(arguments["tool"]).strip().lower()
        docs = self._native_tool_docs()
        if query not in docs:
            available = ", ".join(sorted(docs))
            return self._result(
                f"No documentation found for '{query}'. Available tools: {available}"
            )
        entry = docs[query]
        return self._result(
            f"{query}\nsource: {entry['source']}\ndesc: {entry['desc']}\n"
            f"args: {entry['args']}"
        )


__all__ = ["RuntimeNativeSystemServices"]
