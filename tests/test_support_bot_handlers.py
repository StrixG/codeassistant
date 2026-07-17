"""Tests for the bot's question flow and ticket-close keyboard.

answer_question is the piece with logic in it — it gathers CRM and RAG
context and hands it to the reused answer_message core. The aiogram
handler wrappers around it are trivial and are exercised by hand.
"""

from __future__ import annotations

import json
import types

import pytest

from assistant.core.rag import SearchHit
from support_bot.handlers import answer_question, close_keyboard


class FakeMcp:
    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, name: str, arguments: dict) -> str:
        self.calls.append((name, arguments))
        return self.responses[name]


class FakeRag:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 4) -> list[SearchHit]:
        self.queries.append(query)
        return self.hits


class FakeLlm:
    """Same shape as _FakeLlmOk in test_support_chat_graceful.py, but it
    also records the prompts it was handed so the tests can assert what
    reached the model."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.messages: list[list[dict]] = []

    def chat(self, messages, tools=None, **kwargs):
        self.messages.append(messages)
        message = types.SimpleNamespace(content=self._content)
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])


USER = {"id": "user-1", "name": "Анна", "app_version": "1.4.2", "telegram_id": "111"}
OPEN_TICKET = {
    "id": "ticket-1001",
    "user_id": "user-1",
    "status": "open",
    "priority": "high",
    "subject": "Не могу войти",
    "description": "2FA код всегда неверный",
    "history": [],
}


@pytest.mark.asyncio
async def test_answer_question_puts_crm_and_rag_into_the_answer():
    mcp = FakeMcp({"list_tickets": json.dumps([OPEN_TICKET], ensure_ascii=False)})
    rag = FakeRag(
        [
            SearchHit(
                file_path="faq.md",
                heading_path="Вход",
                git_sha="abc123",
                text="Обновите приложение",
                distance=0.1,
            )
        ]
    )
    llm = FakeLlm("Обновите приложение до 1.6.20.")

    result = await answer_question(
        "Не могу войти", telegram_id=111, user=USER, mcp=mcp, rag=rag, llm=llm
    )

    assert result.answer == "Обновите приложение до 1.6.20."
    assert result.llm_ok
    # The user's profile and open ticket reached the prompt.
    prompt = llm.messages[0][1]["content"]
    assert "1.4.2" in prompt
    assert "ticket-1001" in prompt
    assert "Обновите приложение" in prompt
    assert rag.queries == ["Не могу войти"]


@pytest.mark.asyncio
async def test_answer_question_surfaces_close_suggestion():
    mcp = FakeMcp({"list_tickets": json.dumps([OPEN_TICKET], ensure_ascii=False)})
    rag = FakeRag([])
    llm = FakeLlm("Готово, проблема решена.\nSUGGEST_CLOSE: ticket-1001")

    result = await answer_question(
        "Всё заработало", telegram_id=111, user=USER, mcp=mcp, rag=rag, llm=llm
    )

    assert result.ticket_suggested == "ticket-1001"
    assert "SUGGEST_CLOSE" not in result.answer


@pytest.mark.asyncio
async def test_answer_question_ignores_close_suggestion_for_unknown_ticket():
    mcp = FakeMcp({"list_tickets": json.dumps([], ensure_ascii=False)})
    rag = FakeRag([])
    llm = FakeLlm("Готово.\nSUGGEST_CLOSE: ticket-9999")

    result = await answer_question(
        "Всё заработало", telegram_id=111, user=USER, mcp=mcp, rag=rag, llm=llm
    )

    assert result.ticket_suggested is None


def test_close_keyboard_encodes_ticket_id():
    kb = close_keyboard("ticket-1001")

    data = [b.callback_data for row in kb.inline_keyboard for b in row]
    assert data == ["ticket_close:yes:ticket-1001", "ticket_close:no:ticket-1001"]
