"""Tests for the bot's async CRM layer over MCP.

A fake stands in for McpClient: the real one spawns a stdio subprocess,
which these tests have no need for — only the call_tool contract matters
(returns a string, "Error: ..." for a tool-level failure, raises
RuntimeError when the transport itself is down).
"""

from __future__ import annotations

import json

import pytest

from support_bot import crm
from support_bot.crm import CrmUnavailable


class FakeMcp:
    def __init__(self, responses: dict[str, str | Exception]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, name: str, arguments: dict) -> str:
        self.calls.append((name, arguments))
        value = self.responses[name]
        if isinstance(value, Exception):
            raise value
        return value


USER_JSON = json.dumps({"id": "user-1", "name": "Анна", "telegram_id": "111"}, ensure_ascii=False)


@pytest.mark.asyncio
async def test_find_user_by_telegram_id_returns_profile():
    mcp = FakeMcp({"find_user_by_telegram_id": USER_JSON})

    user = await crm.find_user_by_telegram_id(mcp, 111)

    assert user["id"] == "user-1"
    assert mcp.calls == [("find_user_by_telegram_id", {"telegram_id": "111"})]


@pytest.mark.asyncio
async def test_find_user_by_telegram_id_returns_none_when_unbound():
    mcp = FakeMcp({"find_user_by_telegram_id": "Error: no user bound to telegram_id: '111'"})

    assert await crm.find_user_by_telegram_id(mcp, 111) is None


@pytest.mark.asyncio
async def test_find_user_by_telegram_id_raises_when_mcp_is_down():
    mcp = FakeMcp({"find_user_by_telegram_id": RuntimeError("stdio closed")})

    with pytest.raises(CrmUnavailable):
        await crm.find_user_by_telegram_id(mcp, 111)


@pytest.mark.asyncio
async def test_get_user_returns_none_for_unknown_id():
    mcp = FakeMcp({"get_user": "Error: unknown user_id: 'user-999'"})

    assert await crm.get_user(mcp, "user-999") is None


@pytest.mark.asyncio
async def test_bind_telegram_user_passes_string_id():
    mcp = FakeMcp({"bind_telegram_user": USER_JSON})

    user = await crm.bind_telegram_user(mcp, "user-1", 111)

    assert user["id"] == "user-1"
    assert mcp.calls == [("bind_telegram_user", {"user_id": "user-1", "telegram_id": "111"})]


@pytest.mark.asyncio
async def test_bind_telegram_user_raises_on_error_string():
    mcp = FakeMcp({"bind_telegram_user": "Error: unknown user_id: 'user-999'"})

    with pytest.raises(CrmUnavailable):
        await crm.bind_telegram_user(mcp, "user-999", 111)


@pytest.mark.asyncio
async def test_list_open_tickets_filters_by_status():
    tickets = json.dumps([{"id": "ticket-1001", "status": "open"}])
    mcp = FakeMcp({"list_tickets": tickets})

    result = await crm.list_open_tickets(mcp, "user-1")

    assert [t["id"] for t in result] == ["ticket-1001"]
    assert mcp.calls == [("list_tickets", {"user_id": "user-1", "status": "open"})]


@pytest.mark.asyncio
async def test_update_ticket_sends_status_and_note():
    mcp = FakeMcp({"update_ticket": json.dumps({"id": "ticket-1001", "status": "resolved"})})

    updated = await crm.update_ticket(mcp, "ticket-1001", status="resolved", note="Готово")

    assert updated["status"] == "resolved"
    assert mcp.calls == [
        ("update_ticket", {"ticket_id": "ticket-1001", "status": "resolved", "note": "Готово"})
    ]
