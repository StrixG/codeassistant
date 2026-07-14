"""Tool Registry + Executor for DeepSeek function calling.

One registry holds every tool the model may call: the local ``rag_search``
plus the MCP git tools (proxied through the MCP client). The Executor takes
a tool call from the model, finds the tool, optionally asks the user to
confirm, runs it, and returns the result string.

Human-in-the-loop: a tool with ``requires_confirmation=True`` is not run
until the injected ``confirm`` callback returns True. No such tool is used
in the real flow yet, so a dummy ``git_push`` exercises the mechanism.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from assistant.core.rag import RagSearcher

log = logging.getLogger("assistant.tools")

Handler = Callable[..., str]
# Returns True to allow a confirmation-gated tool to run.
Confirm = Callable[[str], bool]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema for the arguments object
    handler: Handler
    requires_confirmation: bool = False

    def openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def openai_tools(self) -> list[dict]:
        return [t.openai_schema() for t in self._tools.values()]


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, confirm: Confirm) -> None:
        self._registry = registry
        self._confirm = confirm

    def execute(self, name: str, arguments: dict[str, Any] | None) -> str:
        args = arguments or {}
        try:
            tool = self._registry.get(name)
        except KeyError as e:
            return f"Error: {e}"

        if tool.requires_confirmation:
            preview = f"Tool '{name}' wants to run with arguments: {json.dumps(args, ensure_ascii=False)}"
            if not self._confirm(preview):
                log.info("tool %s cancelled by user", name)
                return f"Cancelled by user: '{name}' was not executed."

        try:
            return tool.handler(**args)
        except TypeError as e:
            return f"Error: bad arguments for '{name}': {e}"
        except Exception as e:  # tool failures are reported to the model, not fatal
            log.warning("tool %s failed: %s", name, e)
            return f"Error running '{name}': {e}"


# --- tool definitions ----------------------------------------------------

_RAG_PARAMS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query for the docs."},
        "top_k": {
            "type": "integer",
            "description": "How many chunks to return (default 5).",
            "default": 5,
        },
    },
    "required": ["query"],
}


def make_rag_tool(searcher: RagSearcher, source_sink: list[dict]) -> Tool:
    """rag_search: query Chroma, record sources for citation, return text."""

    def handler(query: str, top_k: int = 5) -> str:
        hits = searcher.search(query, top_k=top_k)
        if not hits:
            return "No matching documentation found."
        lines: list[str] = []
        for h in hits:
            src = h.file_path + (f" :: {h.heading_path}" if h.heading_path else "")
            # Record for the CLI's Sources block (dedup by file+heading).
            entry = {
                "file_path": h.file_path,
                "heading_path": h.heading_path,
                "git_sha": h.git_sha,
            }
            if entry not in source_sink:
                source_sink.append(entry)
            lines.append(f"[SOURCE: {src}]\n{h.text}")
        return "\n\n---\n\n".join(lines)

    return Tool(
        name="rag_search",
        description=(
            "Search the Element Android documentation and return the most "
            "relevant chunks with their source file and heading path."
        ),
        parameters=_RAG_PARAMS,
        handler=handler,
    )


def make_git_push_tool() -> Tool:
    """Dummy confirmation-gated tool. Does nothing but log; proves the hook."""

    def handler(**kwargs: Any) -> str:
        log.info("git_push invoked (dummy) with %s", kwargs)
        return "git_push (dummy) executed: no-op."

    return Tool(
        name="git_push",
        description="(Demo) Push commits to remote. Requires user confirmation.",
        parameters={"type": "object", "properties": {}},
        handler=handler,
        requires_confirmation=True,
    )


def _mcp_tool(mcp_client, name: str, description: str, parameters: dict) -> Tool:
    def handler(**kwargs: Any) -> str:
        return mcp_client.call_tool(name, kwargs)

    return Tool(name=name, description=description, parameters=parameters, handler=handler)


def build_registry(searcher: RagSearcher, mcp_client, source_sink: list[dict]) -> ToolRegistry:
    """Assemble the full registry: rag_search + MCP git tools + dummy git_push."""
    reg = ToolRegistry()
    reg.register(make_rag_tool(searcher, source_sink))
    for t in mcp_client.tools:
        schema = t.inputSchema or {"type": "object", "properties": {}}
        reg.register(_mcp_tool(mcp_client, t.name, t.description or t.name, schema))
    reg.register(make_git_push_tool())
    return reg
