"""Tests for support_assistant.chat's pure context-building helpers.

No MCP, no Chroma, no network — just the text assembly and the
SUGGEST_CLOSE marker parsing.
"""

from __future__ import annotations

from assistant.core.rag import SearchHit
from support_assistant.chat import build_context_block, extract_close_suggestion


def _user() -> dict:
    return {
        "id": "user-1",
        "name": "Анна Смирнова",
        "email": "anna@example.com",
        "platform": "Android",
        "app_version": "1.6.20",
        "plan": "free",
        "signup_date": "2023-03-11",
    }


def _ticket() -> dict:
    return {
        "id": "ticket-1001",
        "status": "open",
        "priority": "high",
        "subject": "Не могу войти",
        "description": "2FA код всегда неверный",
        "history": [
            {"author": "user", "timestamp": "2026-07-14T09:12:00", "text": "Помогите пожалуйста"}
        ],
    }


def _hit() -> SearchHit:
    return SearchHit(
        file_path="faq.md",
        heading_path="Двухфакторная аутентификация (2FA) > Код 2FA всегда неверный",
        git_sha="",
        text="Проверьте автоматическую синхронизацию времени на телефоне.",
        distance=0.1,
        source="faq",
    )


def test_context_includes_profile_fields():
    block = build_context_block(_user(), [], [])
    assert "Анна Смирнова" in block
    assert "1.6.20" in block
    assert "free" in block


def test_context_includes_ticket_id_and_last_message():
    block = build_context_block(_user(), [_ticket()], [])
    assert "ticket-1001" in block
    assert "Не могу войти" in block
    assert "Помогите пожалуйста" in block


def test_context_no_tickets_says_so():
    block = build_context_block(_user(), [], [])
    assert "Открытых тикетов у пользователя нет." in block


def test_context_includes_rag_chunk_source_and_text():
    block = build_context_block(_user(), [], [_hit()])
    assert "faq.md" in block
    assert "Код 2FA всегда неверный" in block
    assert "синхронизацию времени" in block


def test_context_no_hits_says_so():
    block = build_context_block(_user(), [], [])
    assert "Релевантных фрагментов базы знаний не найдено." in block


def test_extract_close_suggestion_parses_trailing_marker():
    answer = "Проблема решена, обновите приложение.\nSUGGEST_CLOSE: ticket-1001"
    cleaned, ticket_id = extract_close_suggestion(answer)
    assert ticket_id == "ticket-1001"
    assert "SUGGEST_CLOSE" not in cleaned
    assert cleaned == "Проблема решена, обновите приложение."


def test_extract_close_suggestion_absent_marker_returns_none():
    answer = "Просто ответ без предложения закрыть тикет."
    cleaned, ticket_id = extract_close_suggestion(answer)
    assert ticket_id is None
    assert cleaned == answer


def test_extract_close_suggestion_trims_whitespace():
    answer = "Ответ.\n\nSUGGEST_CLOSE:   ticket-42  \n"
    cleaned, ticket_id = extract_close_suggestion(answer)
    assert ticket_id == "ticket-42"
    assert cleaned == "Ответ."
