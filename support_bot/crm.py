"""Async CRM access for the bot, over the MCP stdio server.

Everything the bot knows about users and tickets goes through here, and
here it goes through MCP — the bot never opens users.json/tickets.json
itself. McpClient is synchronous (it marshals onto its own loop in a
background thread), so every call is pushed to a worker thread: aiogram
runs one event loop for all chats, and a blocking MCP round-trip would
stall every other user's message.

Lookups return None when the tool reports the record simply isn't there;
anything else — a broken argument, a dead transport — raises
CrmUnavailable, which handlers turn into an apology to the user.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from typing import Any

from assistant.core.mcp_client import McpClient


class CrmUnavailable(RuntimeError):
    """The CRM could not answer: bad arguments, or MCP itself is down."""


async def _call(mcp: McpClient, tool: str, arguments: dict) -> str:
    try:
        return await asyncio.to_thread(mcp.call_tool, tool, arguments)
    except (RuntimeError, concurrent.futures.TimeoutError) as e:  # transport failure or timeout from McpClient.call_tool
        raise CrmUnavailable(f"MCP call {tool} failed: {e}") from e


async def _call_json(mcp: McpClient, tool: str, arguments: dict) -> Any:
    raw = await _call(mcp, tool, arguments)
    if raw.startswith("Error:"):
        raise CrmUnavailable(raw)
    return json.loads(raw)


async def _lookup(mcp: McpClient, tool: str, arguments: dict) -> dict | None:
    """Like _call_json, but a tool-level error means "not found", not a fault."""
    raw = await _call(mcp, tool, arguments)
    if raw.startswith("Error:"):
        return None
    return json.loads(raw)


async def find_user_by_telegram_id(mcp: McpClient, telegram_id: int) -> dict | None:
    return await _lookup(mcp, "find_user_by_telegram_id", {"telegram_id": str(telegram_id)})


async def get_user(mcp: McpClient, user_id: str) -> dict | None:
    return await _lookup(mcp, "get_user", {"user_id": user_id})


async def bind_telegram_user(mcp: McpClient, user_id: str, telegram_id: int) -> dict:
    return await _call_json(
        mcp, "bind_telegram_user", {"user_id": user_id, "telegram_id": str(telegram_id)}
    )


async def list_open_tickets(mcp: McpClient, user_id: str) -> list[dict]:
    return await _call_json(mcp, "list_tickets", {"user_id": user_id, "status": "open"})


async def update_ticket(mcp: McpClient, ticket_id: str, *, status: str, note: str) -> dict:
    return await _call_json(
        mcp, "update_ticket", {"ticket_id": ticket_id, "status": status, "note": note}
    )
