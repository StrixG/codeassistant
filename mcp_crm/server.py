"""MCP stdio server exposing the mock support CRM.

Runs as a separate process, mirroring ``assistant/mcp_server/server.py``.
The CRM data directory is loaded from config at startup and captured in
the tool closures — the LLM passes ids and free-form arguments, never a
path, so it can never point the tools at a different data dir.

Run standalone:  python -m mcp_crm.server
The support assistant CLI spawns this over stdio via the shared
``assistant.core.mcp_client.McpClient``.
"""

from __future__ import annotations

import json
import logging
import os
import sys

from mcp import StdioServerParameters
from mcp.server.fastmcp import FastMCP

# Quiet the per-request INFO chatter so it doesn't pollute the CLI's stderr.
logging.getLogger("mcp").setLevel(logging.WARNING)

from assistant.config import Config
from mcp_crm import crm_store
from mcp_crm.crm_store import CrmError

_cfg = Config.load(require_api_key=False)
_DATA_DIR = _cfg.support_data_dir

mcp = FastMCP("element-crm")


@mcp.tool()
def get_user(user_id: str) -> str:
    """Return the CRM profile for a user (id, name, email, platform,
    app_version, plan, signup_date) as JSON.

    Call this first in any support conversation to know who you're
    talking to — the app_version is often the root cause of login
    problems that look identical to a 2FA failure.
    """
    try:
        return json.dumps(crm_store.get_user(_DATA_DIR, user_id), ensure_ascii=False)
    except CrmError as e:
        return f"Error: {e}"


@mcp.tool()
def list_tickets(user_id: str, status: str = "") -> str:
    """List support tickets for a user as JSON, optionally filtered by
    status (one of: open, pending, resolved). Pass an empty status to get
    all of the user's tickets.

    Use this to check whether the user already has a relevant open ticket
    before answering — if so, reference its ticket id in your reply.
    """
    try:
        tickets = crm_store.list_tickets(_DATA_DIR, user_id, status or None)
        return json.dumps(tickets, ensure_ascii=False)
    except CrmError as e:
        return f"Error: {e}"


@mcp.tool()
def get_ticket(ticket_id: str) -> str:
    """Return full details of one ticket by id (status, priority, subject,
    description, message history) as JSON."""
    try:
        return json.dumps(crm_store.get_ticket(_DATA_DIR, ticket_id), ensure_ascii=False)
    except CrmError as e:
        return f"Error: {e}"


@mcp.tool()
def update_ticket(ticket_id: str, status: str = "", note: str = "") -> str:
    """Update a ticket's status (open/pending/resolved) and/or append a
    support note to its history. Returns the updated ticket as JSON.

    Only call this after the user has explicitly confirmed the action —
    e.g. after they agree the ticket can be closed because the issue is
    resolved.
    """
    try:
        updated = crm_store.update_ticket(
            _DATA_DIR, ticket_id, status=status or None, note=note or None
        )
        return json.dumps(updated, ensure_ascii=False)
    except CrmError as e:
        return f"Error: {e}"


def default_server_params() -> StdioServerParameters:
    """Params to spawn this project's CRM MCP server over stdio."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_crm.server"],
        env=os.environ.copy(),
    )


def main() -> None:
    mcp.run()  # stdio transport by default


if __name__ == "__main__":
    main()
