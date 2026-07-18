"""Tests for the mock-CRM read/write helpers behind the MCP tools.

Runs against temp JSON files, the same way ``test_repo_tools.py`` runs the
git tools against a temp repo — no MCP transport involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_crm import crm_store
from mcp_crm.crm_store import CrmError


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    users = [
        {
            "id": "user-1",
            "name": "Анна Смирнова",
            "email": "anna@example.com",
            "platform": "Android",
            "app_version": "1.6.20",
            "plan": "free",
            "signup_date": "2023-03-11",
            "telegram_id": "111",
        },
        {
            "id": "user-2",
            "name": "Дмитрий Волков",
            "email": "d.volkov@example.com",
            "platform": "Android",
            "app_version": "1.4.2",
            "plan": "pro",
            "signup_date": "2022-11-02",
            "telegram_id": None,
        },
    ]
    tickets = [
        {
            "id": "ticket-1001",
            "user_id": "user-1",
            "status": "open",
            "priority": "high",
            "subject": "Не могу войти",
            "description": "2FA код всегда неверный",
            "history": [{"author": "user", "timestamp": "2026-07-14T09:12:00", "text": "Помогите"}],
        },
        {
            "id": "ticket-1002",
            "user_id": "user-1",
            "status": "resolved",
            "priority": "low",
            "subject": "Старый вопрос",
            "description": "Уже решено",
            "history": [],
        },
    ]
    (tmp_path / "users.json").write_text(json.dumps(users, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "tickets.json").write_text(json.dumps(tickets, ensure_ascii=False), encoding="utf-8")
    return tmp_path


def test_get_user_found(data_dir):
    user = crm_store.get_user(data_dir, "user-1")
    assert user["name"] == "Анна Смирнова"
    assert user["app_version"] == "1.6.20"


def test_get_user_unknown_raises(data_dir):
    with pytest.raises(CrmError):
        crm_store.get_user(data_dir, "user-999")


def test_list_tickets_all(data_dir):
    tickets = crm_store.list_tickets(data_dir, "user-1")
    assert {t["id"] for t in tickets} == {"ticket-1001", "ticket-1002"}


def test_list_tickets_filtered_by_status(data_dir):
    tickets = crm_store.list_tickets(data_dir, "user-1", status="open")
    assert [t["id"] for t in tickets] == ["ticket-1001"]


def test_list_tickets_other_user_is_empty(data_dir):
    assert crm_store.list_tickets(data_dir, "user-2") == []


def test_get_ticket_found(data_dir):
    t = crm_store.get_ticket(data_dir, "ticket-1001")
    assert t["subject"] == "Не могу войти"


def test_get_ticket_unknown_raises(data_dir):
    with pytest.raises(CrmError):
        crm_store.get_ticket(data_dir, "ticket-9999")


def test_update_ticket_status_persists_to_disk(data_dir):
    updated = crm_store.update_ticket(data_dir, "ticket-1001", status="resolved")
    assert updated["status"] == "resolved"

    # Re-read from disk with a fresh load — the write must be durable.
    reloaded = crm_store.get_ticket(data_dir, "ticket-1001")
    assert reloaded["status"] == "resolved"


def test_update_ticket_note_appends_to_history(data_dir):
    before = len(crm_store.get_ticket(data_dir, "ticket-1001")["history"])
    updated = crm_store.update_ticket(data_dir, "ticket-1001", note="Закрыто по подтверждению")
    assert len(updated["history"]) == before + 1
    assert updated["history"][-1]["text"] == "Закрыто по подтверждению"
    assert updated["history"][-1]["author"] == "support"


def test_update_ticket_invalid_status_raises(data_dir):
    with pytest.raises(CrmError):
        crm_store.update_ticket(data_dir, "ticket-1001", status="closed")


def test_update_ticket_unknown_id_raises(data_dir):
    with pytest.raises(CrmError):
        crm_store.update_ticket(data_dir, "ticket-9999", status="resolved")


def test_update_ticket_does_not_touch_other_tickets(data_dir):
    crm_store.update_ticket(data_dir, "ticket-1001", status="resolved")
    other = crm_store.get_ticket(data_dir, "ticket-1002")
    assert other["status"] == "resolved"  # unchanged, was already resolved
    assert other["history"] == []


def test_find_user_by_telegram_id_found(data_dir):
    user = crm_store.find_user_by_telegram_id(data_dir, "111")
    assert user["id"] == "user-1"


def test_find_user_by_telegram_id_unknown_raises(data_dir):
    with pytest.raises(CrmError):
        crm_store.find_user_by_telegram_id(data_dir, "999")


def test_find_user_by_telegram_id_ignores_unbound_users(data_dir):
    # user-2 has telegram_id None — a None lookup must not match it.
    with pytest.raises(CrmError):
        crm_store.find_user_by_telegram_id(data_dir, None)


def test_bind_telegram_user_persists_to_disk(data_dir):
    updated = crm_store.bind_telegram_user(data_dir, "user-2", "222")
    assert updated["telegram_id"] == "222"

    reloaded = crm_store.find_user_by_telegram_id(data_dir, "222")
    assert reloaded["id"] == "user-2"


def test_bind_telegram_user_overwrites_existing_binding(data_dir):
    crm_store.bind_telegram_user(data_dir, "user-2", "111")
    assert crm_store.find_user_by_telegram_id(data_dir, "111")["id"] == "user-2"


def test_bind_telegram_user_unknown_user_raises(data_dir):
    with pytest.raises(CrmError):
        crm_store.bind_telegram_user(data_dir, "user-999", "333")


def test_bind_telegram_user_does_not_touch_other_users(data_dir):
    crm_store.bind_telegram_user(data_dir, "user-2", "222")
    assert crm_store.get_user(data_dir, "user-1")["telegram_id"] == "111"
