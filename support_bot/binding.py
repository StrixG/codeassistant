"""Binding a Telegram account to a CRM user.

The CLI took the user id as --user; a Telegram chat has to say who it is
some other way. On first contact the bot asks for the CRM user id, checks
it exists, and stores the binding in the CRM itself — from then on the
telegram_id alone identifies the person.

Which chats are mid-binding is in-memory state: there are eight mock
users, and after a restart find_user_by_telegram_id finds anyone already
bound, so there is nothing worth persisting here.

No authentication: anyone who knows a user id can bind it to themselves.
Fine for a mock CRM demo, not for anything real.
"""

from __future__ import annotations

import html

from assistant.core.mcp_client import McpClient
from support_bot import crm

BINDING_PROMPT = (
    "Здравствуйте! Не вижу вас в системе поддержки Element.\n\n"
    "Отправьте, пожалуйста, ваш идентификатор пользователя "
    "(например, <code>user-1</code>), чтобы я связал этот чат с вашим профилем."
)


class PendingBindings:
    """Telegram chats that have been asked for a CRM user id."""

    def __init__(self) -> None:
        self._waiting: set[int] = set()

    def mark_waiting(self, telegram_id: int) -> None:
        self._waiting.add(telegram_id)

    def is_waiting(self, telegram_id: int) -> bool:
        return telegram_id in self._waiting

    def clear(self, telegram_id: int) -> None:
        self._waiting.discard(telegram_id)


async def try_bind(
    mcp: McpClient,
    pending: PendingBindings,
    telegram_id: int,
    text: str,
) -> tuple[str, dict | None]:
    """Treat ``text`` as a CRM user id and bind it to ``telegram_id``.

    Returns the reply to send and the bound profile, or ``(reply, None)``
    if the id is unknown — in which case the chat stays in binding mode
    so the person can try again.
    """
    user_id = text.strip()
    user = await crm.get_user(mcp, user_id)
    if user is None:
        # The id came from the chat, so it goes back out escaped — the
        # binding replies are sent with parse_mode=HTML.
        return (
            f"Пользователь <code>{html.escape(user_id)}</code> не найден. "
            "Проверьте идентификатор и отправьте ещё раз (например, <code>user-1</code>).",
            None,
        )

    bound = await crm.bind_telegram_user(mcp, user_id, telegram_id)
    pending.clear(telegram_id)
    return (
        f"Готово, {bound.get('name')}! Этот чат связан с профилем "
        f"<code>{bound['id']}</code>.\n\nЗадайте ваш вопрос — я отвечу с учётом "
        "вашего профиля и открытых тикетов.",
        bound,
    )
