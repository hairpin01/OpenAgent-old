"""Self-Managed Context plugin for OpenAgent.

Features:
- Inspect current context size and estimated tokens.
- On every tool call (in on_after_call hook):
    Checks if active context window exceeds the configured limit (or safe threshold).
    If limit is exceeded, automatically appends context size info and recommendations
    into the tool's output to warn the model and suggest context management steps.
- Summarize / compress history:
    'summary' / 'compress_history' action to summarize or condense older dialog turns
    into a concise summary block, keeping recent turns intact.
- Prune tools / truncate history actions.
"""

from __future__ import annotations

import json
from typing import Any

from ..core import AgentPlugin, ToolContext


def _estimate_tokens(text: str) -> int:
    """Rough estimation: ~4 chars per token."""
    return max(1, len(text) // 4)


def _calculate_total_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += _estimate_tokens(content)
        # Account for tool_calls if any
        calls = m.get("tool_calls")
        if isinstance(calls, list):
            for c in calls:
                func = c.get("function") or {}
                args = str(func.get("arguments") or "")
                name = str(func.get("name") or "")
                total += _estimate_tokens(name) + _estimate_tokens(args)
    return total


class SelfManagedContextPlugin(AgentPlugin):
    """Plugin providing Self-Managed Context capabilities."""

    name = "manage_context"
    description = (
        "Manage, compress, and optimize working context and message history. "
        "Allows inspecting context size, generating summaries of older dialogue to save tokens, "
        "pruning large past tool outputs, and dropping obsolete turns."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "summary", "prune_tools", "truncate_history"],
                "description": (
                    "Action to perform: 'status' to view context size; "
                    "'summary' to compress older dialog turns into a concise summary; "
                    "'prune_tools' to compress old tool outputs in working context; "
                    "'truncate_history' to drop oldest N dialog turns."
                ),
            },
            "summary_text": {
                "type": "string",
                "description": (
                    "Optional custom summary text to replace compressed older turns with. "
                    "If not provided, a default structured summary of the pruned turns is created."
                ),
            },
            "keep_last_turns": {
                "type": "integer",
                "description": (
                    "For 'summary': number of most recent dialog turns (user/assistant exchanges) "
                    "to keep uncompressed (default 2)."
                ),
            },
            "keep_last_tools": {
                "type": "integer",
                "description": (
                    "For 'prune_tools': number of most recent tool outputs to keep intact "
                    "(default 2). Older ones are replaced with brief placeholders."
                ),
            },
            "drop_turns": {
                "type": "integer",
                "description": "For 'truncate_history': number of oldest message turns to drop.",
            },
        },
        "required": ["action"],
    }

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> str:
        action = args.get("action", "status")
        messages: list[dict[str, Any]] = ctx.state.get("active_messages") or []

        if action == "status":
            total_chars = sum(len(str(m.get("content") or "")) for m in messages)
            tool_msg_count = sum(1 for m in messages if m.get("role") == "tool")
            est_tokens = _calculate_total_tokens(messages)
            window_limit = int(ctx.state.get("context_window") or 0)

            return json.dumps(
                {
                    "ok": True,
                    "total_messages": len(messages),
                    "tool_messages": tool_msg_count,
                    "estimated_tokens": est_tokens,
                    "context_window_limit": window_limit,
                    "is_near_or_exceeded": (window_limit > 0 and est_tokens >= window_limit * 0.8),
                    "total_chars": total_chars,
                },
                ensure_ascii=False,
            )

        if action == "summary":
            # Compress older dialogue turns into a summary message
            keep = int(args.get("keep_last_turns", 2))
            keep = max(1, keep)
            custom_summary = args.get("summary_text")

            # Collect indices of dialog messages (exclude system at 0)
            dialog_indices = [
                i for i in range(1, len(messages))
                if messages[i].get("role") in ("user", "assistant")
            ]

            # Require at least keep + 1 turns to compress
            if len(dialog_indices) <= keep:
                return json.dumps(
                    {
                        "ok": True,
                        "message": "Not enough history to compress into summary.",
                        "current_turns": len(dialog_indices),
                    },
                    ensure_ascii=False,
                )

            # Messages to summarize: up to the last `keep` messages
            split_idx = dialog_indices[-keep]
            to_summarize = messages[1:split_idx]

            if not to_summarize:
                return json.dumps(
                    {
                        "ok": True,
                        "message": "No earlier messages to summarize.",
                    },
                    ensure_ascii=False,
                )

            if custom_summary:
                summary_content = f"[Context Summary of earlier turns]:\n{custom_summary}"
            else:
                summary_lines = []
                for m in to_summarize:
                    role = m.get("role", "unknown")
                    content = str(m.get("content") or "")
                    if role == "tool":
                        continue
                    # Short excerpt
                    excerpt = (content[:150] + "...") if len(content) > 150 else content
                    summary_lines.append(f"- {role.capitalize()}: {excerpt}")
                summary_content = (
                    f"[Summary of {len(to_summarize)} earlier messages]:\n"
                    + "\n".join(summary_lines[:15])
                )

            # If there was an earlier summary system message right after system prompt, merge/replace it
            del messages[1:split_idx]
            messages.insert(1, {"role": "system", "content": summary_content})

            new_tokens = _calculate_total_tokens(messages)
            return json.dumps(
                {
                    "ok": True,
                    "summarized_messages_count": len(to_summarize),
                    "remaining_messages": len(messages),
                    "estimated_tokens_now": new_tokens,
                },
                ensure_ascii=False,
            )

        if action == "prune_tools":
            keep = int(args.get("keep_last_tools", 2))
            keep = max(0, keep)
            tool_indices = [
                i for i, m in enumerate(messages) if m.get("role") == "tool"
            ]
            to_prune = tool_indices[:-keep] if keep > 0 else tool_indices
            pruned_count = 0
            saved_chars = 0

            for idx in to_prune:
                content = str(messages[idx].get("content") or "")
                if len(content) > 150:
                    tool_name = messages[idx].get("tool_name", "tool")
                    saved_chars += len(content) - 100
                    messages[idx]["content"] = (
                        f"[{tool_name} output truncated to save context; original {len(content)} chars]"
                    )
                    pruned_count += 1

            return json.dumps(
                {
                    "ok": True,
                    "pruned_tool_messages": pruned_count,
                    "saved_chars_estimate": saved_chars,
                    "remaining_messages": len(messages),
                },
                ensure_ascii=False,
            )

        if action == "truncate_history":
            drop = int(args.get("drop_turns", 1))
            drop = max(0, drop)
            dropped = 0
            # Keep system prompt (index 0) intact
            idx = 1
            while idx < len(messages) and dropped < drop:
                messages.pop(idx)
                dropped += 1

            return json.dumps(
                {
                    "ok": True,
                    "dropped_messages": dropped,
                    "remaining_messages": len(messages),
                },
                ensure_ascii=False,
            )

        return json.dumps({"ok": False, "error": f"Unknown action: {action}"}, ensure_ascii=False)

    async def on_before_call(
        self, name: str, args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any] | str | None:
        return None

    async def on_after_call(
        self,
        name: str,
        args: dict[str, Any],
        result: str,
        ctx: ToolContext,
        messages: list[dict[str, Any]],
    ) -> str:
        ctx.state["active_messages"] = messages
        if not messages:
            return result

        # Compute current estimated token count
        est_tokens = _calculate_total_tokens(messages)
        window_limit = int(ctx.state.get("context_window") or 0)

        # Default fallback threshold if context_window is 0: e.g. 16k tokens or 64k chars
        has_exceeded = False
        limit_desc = ""

        if window_limit > 0:
            if est_tokens >= window_limit:
                has_exceeded = True
                limit_desc = f"{est_tokens}/{window_limit} tokens (exceeded 100% of context window)"
            elif est_tokens >= int(window_limit * 0.85):
                has_exceeded = True
                limit_desc = f"{est_tokens}/{window_limit} tokens (reached {int(est_tokens / window_limit * 100)}% of context window)"
        else:
            # Fallback limit check when context_window is 0 (unspecified)
            # Warn if tokens > 16000 (~64000 chars)
            if est_tokens > 16000:
                has_exceeded = True
                limit_desc = f"~{est_tokens} tokens (high context usage, default threshold 16k tokens)"

        if has_exceeded:
            warning_msg = (
                f"\n\n[CONTEXT WARNING & RECOMMENDATION]\n"
                f"- Context size: {limit_desc}\n"
                f"- Recommendations: Call 'manage_context' with action='summary' to compress older dialogue history, "
                f"or action='prune_tools' to compress older tool outputs, to prevent context overflow."
            )
            return result + warning_msg

        return result


PLUGIN = SelfManagedContextPlugin()
TOOL = PLUGIN
