"""Tests for support_assistant.chat.answer_message: the one-DeepSeek-call
turn logic, including graceful degradation when DeepSeek is unavailable.

The LLM is faked (no network); RAG hits and CRM data are passed in
directly, matching how ``run_chat`` calls this function after MCP/RAG
already returned their results.
"""

from __future__ import annotations

import types

import pytest

from assistant.core.llm import LlmError
from support_assistant.chat import answer_message


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


def _open_ticket(ticket_id: str = "ticket-1001") -> dict:
    return {
        "id": ticket_id,
        "status": "open",
        "priority": "high",
        "subject": "Не могу войти",
        "description": "2FA код всегда неверный",
        "history": [],
    }


def _fake_response(content: str):
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice])


class _FakeLlmDown:
    def chat(self, messages, tools=None, **kwargs):
        raise LlmError("DeepSeek unavailable after retry: timeout")


class _FakeLlmOk:
    def __init__(self, content: str) -> None:
        self._content = content

    def chat(self, messages, tools=None, **kwargs):
        return _fake_response(self._content)


def test_answer_message_degrades_gracefully_when_llm_down():
    result = answer_message(
        "Почему не работает авторизация?",
        user=_user(),
        tickets=[_open_ticket()],
        hits=[],
        llm=_FakeLlmDown(),
    )
    assert result.llm_ok is False
    assert result.ticket_suggested is None
    assert "недоступен" in result.answer.lower()


def test_answer_message_no_exception_propagates_when_llm_down():
    # The whole point: a down LLM must produce an answer, not raise.
    try:
        answer_message(
            "Вопрос", user=_user(), tickets=[], hits=[], llm=_FakeLlmDown()
        )
    except LlmError:
        pytest.fail("LlmError must be caught inside answer_message, not propagated")


def test_answer_message_honours_close_suggestion_for_open_ticket():
    llm = _FakeLlmOk("Обновите время на телефоне, это решит проблему.\nSUGGEST_CLOSE: ticket-1001")
    result = answer_message(
        "Почему не работает 2FA?",
        user=_user(),
        tickets=[_open_ticket("ticket-1001")],
        hits=[],
        llm=llm,
    )
    assert result.ticket_suggested == "ticket-1001"
    assert "SUGGEST_CLOSE" not in result.answer
    assert result.llm_ok is True


def test_answer_message_ignores_close_suggestion_for_unknown_ticket():
    # Model hallucinates a ticket id that isn't actually open for this user.
    llm = _FakeLlmOk("Вот ответ.\nSUGGEST_CLOSE: ticket-9999")
    result = answer_message(
        "Вопрос", user=_user(), tickets=[_open_ticket("ticket-1001")], hits=[], llm=llm
    )
    assert result.ticket_suggested is None


def test_answer_message_plain_answer_without_marker():
    llm = _FakeLlmOk("Element поддерживает Android 7.0 и новее.")
    result = answer_message("Какие версии Android поддерживаются?", user=_user(), tickets=[], hits=[], llm=llm)
    assert result.ticket_suggested is None
    assert result.answer == "Element поддерживает Android 7.0 и новее."
