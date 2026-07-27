# SPDX-License-Identifier: MIT
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from core.lib.types import Event


class OpenAgentContextService:
    """Conversation-history helpers that do not depend on MCUB objects."""

    _VALID_ROLES = frozenset({"system", "user", "assistant"})
    _COMPACTED_PREFIX = "Compacted previous OpenAgent session context:"
    _TOOL_OUTPUT_SPILL_HINT = "Full tool output saved to file:"

    _COMPACTION_SYSTEM_PROMPT = (
        "You compact an OpenAgent chat session. Read the full prior context and "
        "write a concise continuity summary that lets the assistant continue work "
        "without needing the omitted messages. Preserve: user goals, decisions, "
        "constraints, files changed/read, commands run, test results, current TODOs, "
        "open questions, and important warnings. Do not invent facts. Do not include "
        "irrelevant chatter. Output plain text markdown only."
    )

    @staticmethod
    def history_message(role: str, content: Any, limit: int = 12000) -> dict[str, str]:
        text = str(content or "")
        if len(text) > limit:
            text = text[:limit] + "\n...[truncated]"
        return {"role": role, "content": text}

    @classmethod
    def normalize_history_message(
        cls,
        item: dict[str, Any],
        *,
        limit: int = 12000,
    ) -> dict[str, str] | None:
        """Return a provider-safe history message or None for empty/bad entries."""
        if not isinstance(item, dict):
            return None
        role = str(item.get("role") or "assistant").strip().lower()
        if role not in cls._VALID_ROLES:
            role = "assistant"
        content = str(item.get("content") or "").strip()
        if not content:
            return None
        return cls.history_message(role, content, limit=limit)

    def context_entries(
        self,
        prompt: str,
        answer: str,
        tool_trace: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        entries = [self.history_message("user", prompt, limit=8000)]
        for item in tool_trace or []:
            normalized = self.normalize_history_message(item, limit=6000)
            if normalized:
                entries.append(normalized)
        entries.append(self.history_message("assistant", answer, limit=8000))
        return entries

    @classmethod
    def trim_history(
        cls,
        history: list[dict[str, str]],
        context_turns: int,
    ) -> list[dict[str, str]]:
        """Keep the newest user turns while preserving an existing compaction summary."""
        if context_turns <= 0:
            return []

        normalized = [
            item
            for item in (cls.normalize_history_message(item) for item in history)
            if item
        ]
        if not normalized:
            return []

        prefix: list[dict[str, str]] = []
        body = normalized
        first = normalized[0]
        if first["role"] == "system" and first["content"].startswith(
            cls._COMPACTED_PREFIX
        ):
            prefix = [first]
            body = normalized[1:]

        start = 0
        seen_user_turns = 0
        for index in range(len(body) - 1, -1, -1):
            if body[index]["role"] == "user":
                seen_user_turns += 1
                if seen_user_turns >= context_turns:
                    start = index
                    break

        kept = body[start:]
        max_messages = max(context_turns * 6, context_turns * 2)
        if len(kept) > max_messages:
            kept = kept[-max_messages:]
        return [*prefix, *kept]

    @classmethod
    def split_old_and_recent_turns(
        cls,
        history: list[dict[str, str]],
        keep_turns: int,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Split history so recent keeps full user turns, including tool traces."""
        normalized = [
            item
            for item in (cls.normalize_history_message(item) for item in history)
            if item
        ]
        if keep_turns <= 0 or not normalized:
            return normalized, []

        start = len(normalized)
        seen_user_turns = 0
        for index in range(len(normalized) - 1, -1, -1):
            if normalized[index]["role"] == "user":
                seen_user_turns += 1
                start = index
                if seen_user_turns >= keep_turns:
                    break
        return normalized[:start], normalized[start:]

    @staticmethod
    def clean_thinking_notes(thinking_notes: list[str] | None) -> list[str]:
        return [
            str(item).strip() for item in (thinking_notes or []) if str(item).strip()
        ]

    @staticmethod
    def history_chars(history: list[dict[str, str]]) -> int:
        return sum(len(str(item.get("content", ""))) for item in history)

    @staticmethod
    def format_history_for_compaction(history: list[dict[str, str]]) -> str:
        parts = []
        for index, item in enumerate(history, 1):
            role = str(item.get("role", "unknown"))
            content = str(item.get("content", ""))
            parts.append(f"[{index}] {role}:\n{content}")
        return "\n\n".join(parts)

    def compaction_system_prompt(self) -> str:
        return self._COMPACTION_SYSTEM_PROMPT

    @staticmethod
    def safe_tool_name(tool_name: str) -> str:
        name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(tool_name or "tool")).strip("-.")
        return name[:80] or "tool"

    @classmethod
    def format_spilled_tool_call(
        cls,
        *,
        tool_name: str,
        attrs_raw: str,
        body: str,
        output: str,
        file_path: Path,
        preview_chars: int,
    ) -> str:
        output_text = str(output or "")
        if preview_chars <= 0:
            preview = "[preview disabled]"
        elif len(output_text) <= preview_chars * 2:
            preview = output_text
        else:
            preview = (
                output_text[:preview_chars].rstrip()
                + "\n...[middle omitted; full output is in the file above]...\n"
                + output_text[-preview_chars:].lstrip()
            )
        return (
            f"Tool <{tool_name}> call:\n"
            f"attrs: {attrs_raw or '-'}\n"
            f"body: {body or '-'}\n"
            f"output:\n"
            f"[{cls._TOOL_OUTPUT_SPILL_HINT} {file_path}]\n"
            f"[full output chars: {len(output_text)}]\n"
            'To inspect it, call <context.tool_output path="..."> with this path.\n\n'
            f"Preview:\n{preview}"
        )


class _OpenAgentContextMixin:
    """Conversation context, compaction, tool memory and config helpers."""

    def _context_service(self) -> OpenAgentContextService:
        service = getattr(self, "_context_service_instance", None)
        if not isinstance(service, OpenAgentContextService):
            service = OpenAgentContextService()
            self._context_service_instance = service
        return service

    def _request_label(
        self,
        *,
        elapsed: float | None = None,
        thinking_notes: list[str] | None = None,
    ) -> str:
        return self._render_template(
            str(
                self.config.get("request_label", "")
                or self.strings("request_label_default")
            ),
            elapsed=elapsed,
            thinking_notes=thinking_notes,
        )

    def _response_label(
        self,
        *,
        elapsed: float | None = None,
        thinking_notes: list[str] | None = None,
    ) -> str:
        return self._render_template(
            str(
                self.config.get("response_label", "")
                or self.strings("response_label_default")
            ),
            elapsed=elapsed,
            thinking_notes=thinking_notes,
        )

    def _history_message(
        self, role: str, content: Any, limit: int = 12000
    ) -> dict[str, str]:
        return self._context_service().history_message(role, content, limit)

    def _tool_trace_inline_max_chars(self) -> int:
        return max(0, int(self.config.get("tool_trace_inline_max_chars", 6000) or 0))

    def _tool_output_spill_dir(self, chat_id: int | None) -> Path:
        chat_part = str(int(chat_id)) if chat_id else "global"
        return Path(self._workspace_dir()) / "openagent_tool_outputs" / chat_part

    def _tool_output_spill_root(self) -> Path:
        return Path(self._workspace_dir()) / "openagent_tool_outputs"

    def _spill_tool_call_output(
        self,
        chat_id: int | None,
        tool_name: str,
        content: str,
    ) -> Path:
        directory = self._tool_output_spill_dir(chat_id)
        directory.mkdir(parents=True, exist_ok=True)
        safe_name = self._context_service().safe_tool_name(tool_name)
        timestamp = int(time.time() * 1000)
        path = directory / f"{timestamp}-{safe_name}.txt"
        path.write_text(content, encoding="utf-8")
        return path.resolve()

    def _format_tool_call_for_context(
        self,
        chat_id: int | None,
        tool_name: str,
        attrs_raw: str,
        body: str,
        output: str,
    ) -> str:
        full = (
            f"Tool <{tool_name}> call:\n"
            f"attrs: {attrs_raw or '-'}\n"
            f"body: {body or '-'}\n"
            f"output:\n{output}"
        )
        inline_max = self._tool_trace_inline_max_chars()
        if inline_max <= 0 or len(full) <= inline_max:
            return full

        try:
            file_path = self._spill_tool_call_output(chat_id, tool_name, full)
        except OSError as exc:
            self.log.warning(f"OpenAgent tool output spill failed: {exc}")
            return self._history_message("assistant", full, limit=inline_max)["content"]

        preview_chars = max(0, min(2500, inline_max // 2))
        return self._context_service().format_spilled_tool_call(
            tool_name=tool_name,
            attrs_raw=attrs_raw,
            body=body,
            output=output,
            file_path=file_path,
            preview_chars=preview_chars,
        )

    def _tool_output_int_arg(
        self,
        attrs: dict[str, str],
        key: str,
        default: int,
        *,
        minimum: int = 0,
        maximum: int = 200000,
    ) -> int:
        try:
            value = int(str(attrs.get(key, default)).strip())
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(value, maximum))

    def _latest_tool_output_path(self, chat_id: int | None) -> Path | None:
        roots = []
        if chat_id is not None:
            roots.append(self._tool_output_spill_dir(chat_id))
        roots.append(self._tool_output_spill_root())

        candidates: list[Path] = []
        for root in roots:
            if root.exists():
                candidates.extend(
                    path for path in root.rglob("*.txt") if path.is_file()
                )
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _resolve_tool_output_path(
        self,
        selector: str,
        chat_id: int | None,
    ) -> Path | None:
        selector = str(selector or "").strip().strip("\"'")
        if not selector:
            return self._latest_tool_output_path(chat_id)

        root = self._tool_output_spill_root().resolve()
        candidate = Path(selector).expanduser()
        if not candidate.is_absolute():
            chat_dir = (
                self._tool_output_spill_dir(chat_id) if chat_id is not None else root
            )
            direct = chat_dir / candidate
            candidate = direct if direct.exists() else root / candidate
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            return None
        if resolved.is_file():
            return resolved

        fallback_name = Path(selector).name
        if not fallback_name:
            return None
        matches = sorted(
            root.rglob(fallback_name),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return matches[0] if matches else None

    async def _read_tool_output_registry_tool(
        self,
        attrs: dict[str, str],
        body: str,
        source_event: Any | None = None,
    ) -> str:
        chat_id = self._event_chat_id(source_event)
        selector = (
            attrs.get("path")
            or attrs.get("file")
            or attrs.get("id")
            or attrs.get("name")
            or body.strip()
        )
        latest = str(attrs.get("latest", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        path = (
            self._latest_tool_output_path(chat_id)
            if latest
            else self._resolve_tool_output_path(selector, chat_id)
        )
        if path is None:
            return 'No saved tool output found. Pass path="..." from the tool trace, or latest="true".'

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"Failed to read saved tool output: {exc}"
        mode = str(attrs.get("mode") or "head").strip().lower()
        limit = self._tool_output_int_arg(attrs, "limit", 12000, minimum=0)
        offset = self._tool_output_int_arg(attrs, "offset", 0, minimum=0)

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
            header += '\n[truncated: re-call context.tool_output with mode="tail", mode="all", larger limit, or offset]'
        return f"{header}\n\n{shown}"

    def _remember_context(
        self,
        chat_id: int | None,
        prompt: str,
        answer: str,
        tool_trace: list[dict[str, str]] | None = None,
        thinking_notes: list[str] | None = None,
    ) -> None:
        if not chat_id or not self.config["context_enabled"]:
            return
        session = self._get_active_session(int(chat_id))
        history = session.messages
        entries = self._context_service().context_entries(prompt, answer, tool_trace)
        session.thinking_notes = self._context_service().clean_thinking_notes(
            thinking_notes
        )
        history.extend(entries)
        context_turns = int(self.config["context_turns"])
        session.messages = self._context_service().trim_history(history, context_turns)
        if context_turns <= 0:
            session.thinking_notes = []
        self._touch_session(session)
        self._schedule_auto_name_session(session)

    def _history_for_chat(self, chat_id: int | None) -> list[dict[str, str]]:
        if not chat_id or not self.config["context_enabled"]:
            return []
        context_turns = int(self.config.get("context_turns", 10))
        return self._context_service().trim_history(
            self._get_active_session(int(chat_id)).messages,
            context_turns,
        )

    def _history_chars(self, history: list[dict[str, str]]) -> int:
        return self._context_service().history_chars(history)

    def _format_history_for_compaction(self, history: list[dict[str, str]]) -> str:
        return self._context_service().format_history_for_compaction(history)

    def _compaction_system_prompt(self) -> str:
        return self._context_service().compaction_system_prompt()

    async def _compact_chat_history_if_needed(
        self,
        chat_id: int | None,
        provider: str,
        api_key: str,
    ) -> bool:
        if not chat_id or not bool(self.config.get("context_enabled", True)):
            return False
        if not bool(self.config.get("context_compaction_enabled", True)):
            return False

        _compact_session = self._get_active_session(int(chat_id))
        history = _compact_session.messages
        threshold = int(self.config.get("context_compaction_chars", 18000) or 18000)
        if not history or self._history_chars(history) <= threshold:
            return False

        keep_turns = max(
            0, int(self.config.get("context_compaction_keep_turns", 2) or 2)
        )
        (
            old_history,
            recent_history,
        ) = self._context_service().split_old_and_recent_turns(
            history,
            keep_turns,
        )
        if not old_history:
            return False

        max_chars = max(threshold * 2, threshold + 4000)
        compact_input = self._format_history_for_compaction(old_history)
        if len(compact_input) > max_chars:
            compact_input = compact_input[-max_chars:]

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._compaction_system_prompt()},
            {
                "role": "user",
                "content": (
                    "Compact this OpenAgent session context. The assistant will continue "
                    "after your summary, with the newest turns kept separately.\n\n"
                    f"{compact_input}"
                ),
            },
        ]
        max_tokens = int(self.config.get("context_compaction_max_tokens", 900) or 900)
        try:
            if provider in ("openai", "openrouter", "groq", "deepseek", "xai", "other"):
                summary = await self._ask_openai_compatible(
                    provider,
                    messages,
                    api_key,
                    max_tokens_override=max_tokens,
                )
            elif provider == "google":
                summary = await self._ask_google(
                    messages,
                    api_key,
                    max_tokens_override=max_tokens,
                )
            else:
                return False
        except Exception as exc:
            self.log.warning(f"OpenAgent context compaction failed: {exc}")
            return False

        summary = (summary or "").strip()
        if not summary:
            return False

        _compact_session.messages = [
            {
                "role": "system",
                "content": "Compacted previous OpenAgent session context:\n"
                + summary[-12000:],
            },
            *recent_history,
        ]
        self._touch_session(_compact_session)
        return True

    def _tool_memory_note(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        if not text:
            return ""
        max_chars = int(self.config.get("tool_memory_max_chars", 500) or 500)
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        return text

    def _remember_tool_output(
        self, chat_id: int | None, tool_name: str, output: str
    ) -> None:
        if not chat_id or not bool(self.config.get("tool_memory_enabled", False)):
            return
        note = self._tool_memory_note(output)
        if not note:
            return
        memory = self._tool_memory.setdefault(int(chat_id), [])
        memory.append(f"{tool_name}: {note}")
        max_items = int(self.config.get("tool_memory_items", 20) or 20)
        if max_items <= 0:
            memory.clear()
        else:
            del memory[:-max_items]

    def _tool_memory_prompt(self, chat_id: int | None) -> str:
        if not chat_id or not bool(self.config.get("tool_memory_enabled", False)):
            return ""
        notes = self._tool_memory.get(int(chat_id), [])
        if not notes:
            return ""
        return "Recent tool memory:\n" + "\n".join(
            f"- {line}"
            for line in notes[-int(self.config.get("tool_memory_items", 20) or 20) :]
        )

    def _base_url(self, provider: str) -> str:
        if provider == "other":
            return str(self.config.get("custom_base_url", "") or "").strip().rstrip("/")
        return self.BASE_URLS[provider].rstrip("/")

    def _args_raw(self, event: Event) -> str:
        return self.args_raw(event).strip()

    def _invalidate_config_caches(self, key: str | None = None) -> None:
        if key in {None, "todo_status_emojis"}:
            self._todo_status_map_raw = None
            self._todo_status_map_cache = None
        if key in {None, "tool_status_emojis"}:
            self._tool_status_emojis_raw = None
            self._tool_status_emojis_cache = None

    async def _set_config_value(self, key: str, value: Any) -> None:
        self.config[key] = value
        self._invalidate_config_caches(key)
        await self.save_config()


__all__ = [
    "_OpenAgentContextMixin",
    "OpenAgentContextService",
]
