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
import contextlib
import logging
from collections.abc import AsyncIterator

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramBadRequest
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

# Telegram rejects a message over 4096 characters, and message.answer("")
# with an empty string, so long/empty model answers need a bit of massaging
# before they go out. Both helpers are pure (str/list in, str/list out) so
# they are unit-tested without any aiogram types.
TELEGRAM_MESSAGE_LIMIT = 4096
EMPTY_ANSWER_FALLBACK = "Готово."

# Telegram's "typing…" indicator auto-clears after ~5s, so a single
# send_chat_action goes dark long before a slow answer (MCP + RAG +
# DeepSeek) is ready — and the chat looks frozen. Re-send it a little
# under that window, in the background, for as long as we're working.
TYPING_REFRESH_SECONDS = 4.0


@contextlib.asynccontextmanager
async def keep_typing(bot: Bot, chat_id: int) -> AsyncIterator[None]:
    """Hold the "typing…" indicator up for the whole ``with`` body.

    A background task re-sends the chat action every
    ``TYPING_REFRESH_SECONDS`` until the block exits. The indicator is
    cosmetic, so any send failure is logged and swallowed — it must never
    break the actual answer. CancelledError is not an Exception subclass,
    so the ``except`` below lets clean cancellation on exit through.
    """

    async def loop() -> None:
        while True:
            try:
                await bot.send_chat_action(chat_id, "typing")
            except Exception as e:  # cosmetic only — never break the answer
                log.warning("failed to send typing action to chat=%s: %s", chat_id, e)
            await asyncio.sleep(TYPING_REFRESH_SECONDS)

    task = asyncio.create_task(loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


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


def render_answer(answer: str) -> str:
    """Substitute a short fallback for an empty/whitespace model answer.

    answer_message can return "" when the model's whole reply was a
    SUGGEST_CLOSE marker (stripped out by extract_close_suggestion); an
    empty message.answer("") call is a Telegram 400. The fallback reads
    naturally right before the close-ticket prompt that usually follows it.
    """
    stripped = answer.strip()
    return stripped if stripped else EMPTY_ANSWER_FALLBACK


def split_answer_into_chunks(answer: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Split ``answer`` into chunks Telegram will accept (<= ``limit`` chars).

    Prefers cutting on a newline near the limit so paragraphs stay whole;
    a single line longer than ``limit`` has no such boundary and is hard-cut.
    Every character of ``answer`` ends up in exactly one chunk, in order.
    """
    if len(answer) <= limit:
        return [answer]

    chunks: list[str] = []
    remaining = answer
    while len(remaining) > limit:
        window = remaining[:limit]
        newline_idx = window.rfind("\n")
        # Cut right after the newline so it stays with the chunk before it;
        # a newline at index 0 (or none at all) isn't a useful boundary.
        cut = newline_idx + 1 if newline_idx > 0 else limit
        chunks.append(remaining[:cut])
        remaining = remaining[cut:]
    if remaining:
        chunks.append(remaining)
    return chunks


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

    async with keep_typing(message.bot, message.chat.id):
        result = await answer_question(
            message.text, telegram_id=telegram_id, user=user, mcp=mcp, rag=rag, llm=llm
        )
    for chunk in split_answer_into_chunks(render_answer(result.answer)):
        await message.answer(chunk)

    if result.ticket_suggested:
        await message.answer(
            f"Похоже, проблема из тикета {result.ticket_suggested} решена. Закрыть его?",
            reply_markup=close_keyboard(result.ticket_suggested),
        )


async def _show_on_callback_message(callback: CallbackQuery, text: str) -> None:
    """Best-effort display of ``text`` on the ticket-close callback's message.

    Called after callback.answer() has already dismissed the spinner, so
    nothing here may raise: the message can be too old to edit or already
    identical (TelegramBadRequest), or it can be an InaccessibleMessage,
    which has no edit_text at all (AttributeError). Fall back to a fresh
    reply, and if even that fails, just log it — the spinner is already gone.
    """
    try:
        await callback.message.edit_text(text)
        return
    except (TelegramBadRequest, AttributeError) as e:
        log.warning("failed to edit ticket-close message: %s", e)

    try:
        await callback.message.answer(text)
    except TelegramBadRequest as e:
        log.warning("failed to send ticket-close fallback message: %s", e)


@router.callback_query(F.data.startswith("ticket_close:"))
async def handle_ticket_close(callback: CallbackQuery, mcp: McpClient) -> None:
    _, decision, ticket_id = callback.data.split(":", 2)
    # Dismiss the spinner first, before touching the message or the CRM —
    # everything past this point is best-effort, the spinner must not hang.
    await callback.answer()

    if decision == "no":
        await _show_on_callback_message(callback, f"Хорошо, тикет {ticket_id} остаётся открытым.")
        return

    try:
        await crm.update_ticket(mcp, ticket_id, status="resolved", note=CLOSE_NOTE)
    except CrmUnavailable as e:
        log.warning("failed to close %s: %s", ticket_id, e)
        await _show_on_callback_message(callback, CRM_DOWN_MESSAGE)
        return

    await _show_on_callback_message(callback, f"Тикет {ticket_id} закрыт. Спасибо!")


@router.errors()
async def handle_unexpected_error(event) -> bool:
    """Log anything that escaped a handler and tell the user, without dying."""
    log.exception("unhandled error in handler", exc_info=event.exception)
    message = getattr(event.update, "message", None)
    if message is not None:
        await message.answer(UNEXPECTED_ERROR_MESSAGE)
        return True

    # A callback_query update has no .message to fall back on that's safe
    # to send to — answering the callback is the one thing that must always
    # happen, or the user is left staring at a spinning button.
    callback_query = getattr(event.update, "callback_query", None)
    if callback_query is not None:
        await callback_query.answer(UNEXPECTED_ERROR_MESSAGE, show_alert=True)
    return True
