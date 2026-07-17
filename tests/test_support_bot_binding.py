"""Tests for binding a Telegram account to a CRM user.

Drives the binding functions directly with a fake MCP client — no
Telegram API and no MCP subprocess involved.
"""

from __future__ import annotations

import json

import pytest

from support_bot.binding import PendingBindings, try_bind


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


def test_pending_bindings_tracks_waiting_chats():
    pending = PendingBindings()
    assert not pending.is_waiting(111)

    pending.mark_waiting(111)
    assert pending.is_waiting(111)
    assert not pending.is_waiting(222)

    pending.clear(111)
    assert not pending.is_waiting(111)


def test_pending_bindings_clear_is_idempotent():
    pending = PendingBindings()
    pending.clear(111)  # must not raise on an id that was never waiting
    assert not pending.is_waiting(111)


@pytest.mark.asyncio
async def test_try_bind_success_binds_and_stops_waiting():
    mcp = FakeMcp({"get_user": USER_JSON, "bind_telegram_user": USER_JSON})
    pending = PendingBindings()
    pending.mark_waiting(111)

    reply, user = await try_bind(mcp, pending, 111, "user-1")

    assert user["id"] == "user-1"
    assert "Анна" in reply
    assert not pending.is_waiting(111)
    assert [c[0] for c in mcp.calls] == ["get_user", "bind_telegram_user"]


@pytest.mark.asyncio
async def test_try_bind_unknown_user_keeps_waiting():
    mcp = FakeMcp({"get_user": "Error: unknown user_id: 'user-999'"})
    pending = PendingBindings()
    pending.mark_waiting(111)

    reply, user = await try_bind(mcp, pending, 111, "user-999")

    assert user is None
    assert "не найден" in reply.lower()
    assert pending.is_waiting(111)  # still in binding mode, can retry
    assert [c[0] for c in mcp.calls] == ["get_user"]  # no bind attempted


@pytest.mark.asyncio
async def test_try_bind_escapes_user_text_in_the_error_reply():
    # The reply is sent with parse_mode=HTML, so whatever the person typed
    # must come back escaped or the message breaks.
    mcp = FakeMcp({"get_user": "Error: unknown user_id: '<b>'"})
    pending = PendingBindings()
    pending.mark_waiting(111)

    reply, _ = await try_bind(mcp, pending, 111, "<b>")

    assert "&lt;b&gt;" in reply
    assert "<b>" not in reply


@pytest.mark.asyncio
async def test_try_bind_escapes_html_special_chars_in_the_bound_name():
    # The success reply is also sent with parse_mode=HTML, so CRM data
    # interpolated into it must come back escaped just like the error path.
    bound_json = json.dumps(
        {"id": "user-1", "name": "<b>Анна</b> & Co", "telegram_id": "111"}, ensure_ascii=False
    )
    mcp = FakeMcp({"get_user": bound_json, "bind_telegram_user": bound_json})
    pending = PendingBindings()
    pending.mark_waiting(111)

    reply, user = await try_bind(mcp, pending, 111, "user-1")

    assert "&lt;b&gt;Анна&lt;/b&gt; &amp; Co" in reply
    assert "<b>Анна</b>" not in reply


@pytest.mark.asyncio
async def test_try_bind_strips_whitespace_from_user_id():
    mcp = FakeMcp({"get_user": USER_JSON, "bind_telegram_user": USER_JSON})
    pending = PendingBindings()
    pending.mark_waiting(111)

    await try_bind(mcp, pending, 111, "  user-1  ")

    assert mcp.calls[0] == ("get_user", {"user_id": "user-1"})
