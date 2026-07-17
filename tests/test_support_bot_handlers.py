"""Tests for the bot's question flow and ticket-close keyboard.

answer_question is the piece with logic in it — it gathers CRM and RAG
context and hands it to the reused answer_message core. The aiogram
handler wrappers around it are trivial and are exercised by hand.
"""

from __future__ import annotations

import json
import types

import pytest
from aiogram.exceptions import TelegramBadRequest

from assistant.core.rag import SearchHit
from support_bot.handlers import (
    CRM_DOWN_MESSAGE,
    UNEXPECTED_ERROR_MESSAGE,
    answer_question,
    close_keyboard,
    handle_ticket_close,
    handle_unexpected_error,
    render_answer,
    split_answer_into_chunks,
)


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


# --- render_answer / split_answer_into_chunks -------------------------------
# Pure helpers, no aiogram types, so they are tested directly.


def test_render_answer_returns_normal_text_unchanged():
    assert render_answer("Обновите приложение до 1.6.20.") == "Обновите приложение до 1.6.20."


def test_render_answer_falls_back_on_empty_string():
    assert render_answer("") == "Готово."


def test_render_answer_falls_back_on_whitespace_only():
    assert render_answer("   \n\t  ") == "Готово."


def test_split_answer_into_chunks_short_text_stays_one_chunk():
    text = "Короткий ответ."
    assert split_answer_into_chunks(text) == [text]


def test_split_answer_into_chunks_splits_long_text_on_newline_boundary():
    line = "Строка ответа с содержимым.\n"
    text = line * 200  # well over 4096 chars, newline boundaries throughout

    chunks = split_answer_into_chunks(text)

    assert len(chunks) > 1
    assert all(len(c) <= 4096 for c in chunks)
    assert "".join(chunks) == text


def test_split_answer_into_chunks_hard_cuts_a_single_overlong_line():
    text = "a" * 9000  # one line, no newline anywhere to split on

    chunks = split_answer_into_chunks(text)

    assert len(chunks) == 3
    assert all(len(c) <= 4096 for c in chunks)
    assert "".join(chunks) == text


# --- handle_ticket_close resilience -----------------------------------------
# The callback spinner must be dismissed no matter what happens to the
# message edit afterwards (old message, InaccessibleMessage, CRM down).


class FakeEditableMessage:
    def __init__(self, *, fail_edit: bool = False) -> None:
        self.fail_edit = fail_edit
        self.edited: list[str] = []
        self.answered: list[str] = []

    async def edit_text(self, text, **kwargs):
        if self.fail_edit:
            raise TelegramBadRequest(method=None, message="message to edit not found")
        self.edited.append(text)

    async def answer(self, text, **kwargs):
        self.answered.append(text)


class FakeInaccessibleMessage:
    """Stand-in for aiogram's InaccessibleMessage: no edit_text at all."""

    def __init__(self) -> None:
        self.answered: list[str] = []

    async def answer(self, text, **kwargs):
        self.answered.append(text)


class FakeCallbackQuery:
    def __init__(self, data: str, message) -> None:
        self.data = data
        self.message = message
        self.answer_calls: list[tuple] = []

    async def answer(self, *args, **kwargs):
        self.answer_calls.append((args, kwargs))


@pytest.mark.asyncio
async def test_handle_ticket_close_decline_answers_callback_and_edits_message():
    message = FakeEditableMessage()
    callback = FakeCallbackQuery("ticket_close:no:ticket-1001", message)

    await handle_ticket_close(callback, mcp=None)

    assert callback.answer_calls  # spinner dismissed
    assert message.edited == ["Хорошо, тикет ticket-1001 остаётся открытым."]


@pytest.mark.asyncio
async def test_handle_ticket_close_answers_callback_even_when_edit_fails():
    message = FakeEditableMessage(fail_edit=True)
    callback = FakeCallbackQuery("ticket_close:no:ticket-1001", message)

    await handle_ticket_close(callback, mcp=None)

    assert callback.answer_calls  # spinner still dismissed
    assert message.answered == ["Хорошо, тикет ticket-1001 остаётся открытым."]  # fallback


@pytest.mark.asyncio
async def test_handle_ticket_close_survives_inaccessible_message():
    message = FakeInaccessibleMessage()  # no edit_text -> would be AttributeError
    callback = FakeCallbackQuery("ticket_close:no:ticket-1001", message)

    await handle_ticket_close(callback, mcp=None)

    assert callback.answer_calls
    assert message.answered == ["Хорошо, тикет ticket-1001 остаётся открытым."]


@pytest.mark.asyncio
async def test_handle_ticket_close_resolves_ticket_and_edits_message():
    message = FakeEditableMessage()
    callback = FakeCallbackQuery("ticket_close:yes:ticket-1001", message)
    mcp = FakeMcp({"update_ticket": json.dumps({"id": "ticket-1001", "status": "resolved"})})

    await handle_ticket_close(callback, mcp=mcp)

    assert callback.answer_calls
    assert message.edited == ["Тикет ticket-1001 закрыт. Спасибо!"]
    assert [c[0] for c in mcp.calls] == ["update_ticket"]


@pytest.mark.asyncio
async def test_handle_ticket_close_crm_down_still_dismisses_spinner():
    message = FakeEditableMessage(fail_edit=True)
    callback = FakeCallbackQuery("ticket_close:yes:ticket-1001", message)
    mcp = FakeMcp({"update_ticket": "Error: mcp down"})

    await handle_ticket_close(callback, mcp=mcp)

    assert callback.answer_calls
    assert message.answered == [CRM_DOWN_MESSAGE]


# --- handle_unexpected_error -------------------------------------------------


@pytest.mark.asyncio
async def test_handle_unexpected_error_answers_the_message_update():
    message = FakeEditableMessage()
    event = types.SimpleNamespace(
        exception=RuntimeError("boom"),
        update=types.SimpleNamespace(message=message, callback_query=None),
    )

    result = await handle_unexpected_error(event)

    assert result is True
    assert message.answered == [UNEXPECTED_ERROR_MESSAGE]


@pytest.mark.asyncio
async def test_handle_unexpected_error_dismisses_callback_query_spinner():
    callback = FakeCallbackQuery("ticket_close:yes:ticket-1001", FakeEditableMessage())
    event = types.SimpleNamespace(
        exception=RuntimeError("boom"),
        update=types.SimpleNamespace(message=None, callback_query=callback),
    )

    result = await handle_unexpected_error(event)

    assert result is True
    assert callback.answer_calls
    assert callback.answer_calls[0][0][0] == UNEXPECTED_ERROR_MESSAGE
