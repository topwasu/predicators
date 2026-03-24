"""Shared markdown log formatting for agent session conversations.

Converts the ``List[Dict[str, Any]]`` collected by response parsers into
a human-readable markdown document.  Used by
``LocalSandboxSessionManager._flush_log`` and
``docker_agent_runner._flush_log``.
"""
import json
from typing import Any, Dict, List, Optional


def format_conversation_markdown(
    collected: List[Dict[str, Any]],
    title: str = "Query",
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Format collected response messages as a markdown document.

    Args:
        collected: List of parsed message dicts (from ``response_parser``).
        title: Heading for the document (e.g. "Docker Query").
        meta: Optional metadata dict with keys like ``query_number``,
            ``timestamp``, ``session_id``, ``query`` (the prompt text).

    Returns:
        A complete markdown string.
    """
    lines: List[str] = []
    lines.append(f"# {title}\n")

    if meta:
        for key in ("query_number", "timestamp", "session_id"):
            val = meta.get(key)
            if val is not None:
                label = key.replace("_", " ").title()
                lines.append(f"- **{label}:** {val}")
        lines.append("")

    lines.append("## Prompt\n")
    lines.append(meta.get("query", "") if meta else "")
    lines.append("\n")
    lines.append("## Conversation\n")

    turn_num = 0
    for entry in collected:
        etype = entry.get("type", "")

        if etype == "assistant":
            turn_num += 1
            if turn_num > 1:
                lines.append("---\n")
            lines.append(f"### Turn {turn_num}\n")
            for block in entry.get("content", []):
                _format_assistant_block(block, lines)

        elif etype == "user":
            for block in entry.get("content", []):
                _format_user_block(block, lines)

        elif etype == "result":
            turns = entry.get("num_turns", "?")
            cost = entry.get("total_cost_usd")
            cost_str = f"${cost:.2f}" if cost is not None else "?"
            lines.append(f"---\n\n**Result:** {turns} turns, {cost_str}\n")

        elif etype == "error":
            lines.append(f"**Error:** {entry.get('error', '')}\n")

    return "\n".join(lines)


def _format_assistant_block(block: Dict[str, Any], lines: List[str]) -> None:
    """Append markdown for a single assistant content block."""
    btype = block.get("type", "")

    if btype == "ThinkingBlock":
        thinking = block.get("thinking", "")
        if thinking:
            lines.append("*[thinking]*")
            for tline in thinking.splitlines():
                lines.append(f"> {tline}")
            lines.append("")
    elif btype == "text":
        lines.append(f"**Assistant:** {block.get('text', '')}\n")
    elif btype == "tool_use":
        name = block.get("name", "?")
        tool_id = block.get("id", "")
        inp = block.get("input", {})
        lines.append(f"**Tool Call:** `{name}` (id: `{tool_id}`)")
        lines.append("```json")
        lines.append(json.dumps(inp, indent=2, default=str))
        lines.append("```\n")
    else:
        _format_unknown_block(block, lines)


def _format_user_block(block: Dict[str, Any], lines: List[str]) -> None:
    """Append markdown for a single user content block."""
    btype = block.get("type", "")

    if btype == "tool_result":
        tool_use_id = block.get("tool_use_id", "")
        content = block.get("content")
        is_error = block.get("is_error", False)
        label = "Tool Error" if is_error else "Tool Result"
        lines.append(f"**{label}** (tool_use_id: `{tool_use_id}`):")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "image":
                        mime = item.get("mimeType", "image/*")
                        lines.append(f"*[image: {mime}]*\n")
                    else:
                        lines.append("```")
                        lines.append(item.get("text", str(item)))
                        lines.append("```")
                else:
                    lines.append("```")
                    lines.append(str(item))
                    lines.append("```")
        elif content is not None:
            lines.append("```")
            lines.append(str(content))
            lines.append("```")
        lines.append("")
    elif btype == "text":
        lines.append(f"**User:** {block.get('text', '')}\n")
    else:
        _format_unknown_block(block, lines)


def _format_unknown_block(block: Dict[str, Any], lines: List[str]) -> None:
    """Append markdown for an unknown block type."""
    btype = block.get("type", "unknown")
    lines.append(f"**{btype}:**")
    extra = {k: v for k, v in block.items() if k != "type" and v is not None}
    if extra:
        lines.append("```json")
        lines.append(json.dumps(extra, indent=2, default=str))
        lines.append("```")
    lines.append("")
