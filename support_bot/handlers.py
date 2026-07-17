"""aiogram handlers: /start, questions, and the ticket-close callback.

The answering core is support_assistant.chat.answer_message, reused as
is: MCP and RAG run deterministically before the model, then one DeepSeek
call per message. Everything blocking (MCP, Chroma, DeepSeek) is pushed
off the event loop with to_thread so one slow answer doesn't hold up
every other chat.

Dependencies (mcp, rag, llm, pending) arrive from the dispatcher's
workflow_data — see support_bot/bot.py, where they are registered under
these exact names.
"""

from __future__ import annotations

import asyncio
import logging

from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from assistant.core.llm import DeepSeekClient
from assistant.core.mcp_client import McpClient
from assistant.core.rag import RagSearcher
from support_assistant.chat import SupportTurnResult, answer_message
from support_bot import crm
from support_bot.binding import BINDING_PROMPT, PendingBindings, try_bind
from support_bot.crm import CrmUnavailable

log = logging.getLogger(__name__)

router = Router()

# Only the binding messages are HTML — they format user ids as <code>.
# The bot has no global parse_mode on purpose: answers come from the model
# and any stray "<" in them would break HTML parsing and drop the message.
HTML = "HTML"

CRM_DOWN_MESSAGE = (
    "Извините, сервис поддержки сейчас недоступен. "
    "Попробуйте, пожалуйста, ещё раз через пару минут."
)
UNEXPECTED_ERROR_MESSAGE = "Извините, что-то пошло не так. Попробуйте ещё раз чуть позже."

CLOSE_NOTE = "Закрыто ассистентом поддержки после подтверждения пользователем."


def close_keyboard(ticket_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"✅ Закрыть тикет {ticket_id}",
                    callback_data=f"ticket_close:yes:{ticket_id}",
                ),
                InlineKeyboardButton(
                    text="Не сейчас", callback_data=f"ticket_close:no:{ticket_id}"
                ),
            ]
        ]
    )


async def answer_question(
    question: str,
    *,
    telegram_id: int,
    user: dict,
    mcp: McpClient,
    rag: RagSearcher,
    llm: DeepSeekClient,
) -> SupportTurnResult:
    """Gather CRM + RAG context for one question and run the shared core."""
    tickets, hits = await asyncio.gather(
        crm.list_open_tickets(mcp, user["id"]),
        asyncio.to_thread(rag.search, question, 4),
    )

    result = await asyncio.to_thread(
        answer_message, question, user=user, tickets=tickets, hits=hits, llm=llm
    )

    # Same debug trail the CLI printed, now in the bot's log rather than
    # in the user's chat.
    chunks = "; ".join(
        h.file_path + (f" :: {h.heading_path}" if h.heading_path else "") for h in hits
    )
    log.info(
        "tg=%s user=%s tickets=%d rag=[%s] llm_ok=%s",
        telegram_id,
        user["id"],
        len(tickets),
        chunks or "-",
        result.llm_ok,
    )
    return result


@router.message(CommandStart())
async def handle_start(
    message: Message,
    mcp: McpClient,
    pending: PendingBindings,
) -> None:
    telegram_id = message.from_user.id
    try:
        user = await crm.find_user_by_telegram_id(mcp, telegram_id)
    except CrmUnavailable as e:
        log.warning("CRM unavailable on /start for tg=%s: %s", telegram_id, e)
        await message.answer(CRM_DOWN_MESSAGE)
        return

    if user is None:
        pending.mark_waiting(telegram_id)
        await message.answer(BINDING_PROMPT, parse_mode=HTML)
        return

    await message.answer(
        f"Здравствуйте, {user.get('name')}! Я ассистент поддержки Element для Android.\n\n"
        "Опишите вашу проблему — отвечу с учётом вашего профиля и открытых тикетов."
    )


@router.message(F.text)
async def handle_text(
    message: Message,
    mcp: McpClient,
    rag: RagSearcher,
    llm: DeepSeekClient,
    pending: PendingBindings,
) -> None:
    telegram_id = message.from_user.id
    try:
        if pending.is_waiting(telegram_id):
            reply, _ = await try_bind(mcp, pending, telegram_id, message.text)
            await message.answer(reply, parse_mode=HTML)
            return

        user = await crm.find_user_by_telegram_id(mcp, telegram_id)
        if user is None:
            pending.mark_waiting(telegram_id)
            await message.answer(BINDING_PROMPT, parse_mode=HTML)
            return
    except CrmUnavailable as e:
        log.warning("CRM unavailable for tg=%s: %s", telegram_id, e)
        await message.answer(CRM_DOWN_MESSAGE)
        return

    await message.bot.send_chat_action(message.chat.id, "typing")
    result = await answer_question(
        message.text, telegram_id=telegram_id, user=user, mcp=mcp, rag=rag, llm=llm
    )
    await message.answer(result.answer)

    if result.ticket_suggested:
        await message.answer(
            f"Похоже, проблема из тикета {result.ticket_suggested} решена. Закрыть его?",
            reply_markup=close_keyboard(result.ticket_suggested),
        )


@router.callback_query(F.data.startswith("ticket_close:"))
async def handle_ticket_close(callback: CallbackQuery, mcp: McpClient) -> None:
    _, decision, ticket_id = callback.data.split(":", 2)

    if decision == "no":
        await callback.message.edit_text(f"Хорошо, тикет {ticket_id} остаётся открытым.")
        await callback.answer()
        return

    try:
        await crm.update_ticket(mcp, ticket_id, status="resolved", note=CLOSE_NOTE)
    except CrmUnavailable as e:
        log.warning("failed to close %s: %s", ticket_id, e)
        await callback.message.edit_text(CRM_DOWN_MESSAGE)
        await callback.answer()
        return

    await callback.message.edit_text(f"Тикет {ticket_id} закрыт. Спасибо!")
    await callback.answer()


@router.errors()
async def handle_unexpected_error(event) -> bool:
    """Log anything that escaped a handler and tell the user, without dying."""
    log.exception("unhandled error in handler", exc_info=event.exception)
    message = getattr(event.update, "message", None)
    if message is not None:
        await message.answer(UNEXPECTED_ERROR_MESSAGE)
    return True
