"""Pure read/write helpers over the mock-CRM JSON files.

Mirrors ``assistant/mcp_server/repo_tools.py``: no MCP here, so it is
unit-testable on its own against a temp data dir. The data dir is passed
in by the caller from config, never taken from an LLM argument.

Files: ``users.json`` (read and written by ``bind_telegram_user``) and
``tickets.json`` (read and written by ``update_ticket``).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class CrmError(ValueError):
    """Raised for missing users/tickets or bad update arguments."""


def _load_json(path: Path) -> list[dict]:
    if not path.is_file():
        raise CrmError(f"data file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _users_path(data_dir: Path) -> Path:
    return data_dir / "users.json"


def _tickets_path(data_dir: Path) -> Path:
    return data_dir / "tickets.json"


def load_users(data_dir: Path) -> list[dict]:
    return _load_json(_users_path(data_dir))


def load_tickets(data_dir: Path) -> list[dict]:
    return _load_json(_tickets_path(data_dir))


def save_tickets(data_dir: Path, tickets: list[dict]) -> None:
    _tickets_path(data_dir).write_text(
        json.dumps(tickets, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_users(data_dir: Path, users: list[dict]) -> None:
    _users_path(data_dir).write_text(
        json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_user(data_dir: Path, user_id: str) -> dict:
    for u in load_users(data_dir):
        if u["id"] == user_id:
            return u
    raise CrmError(f"unknown user_id: {user_id!r}")


def find_user_by_telegram_id(data_dir: Path, telegram_id: str) -> dict:
    """Return the CRM profile bound to a Telegram account id.

    Users with no binding yet carry ``telegram_id: null``; a ``None``
    lookup must not match them, hence the explicit guard.
    """
    if telegram_id is None:
        raise CrmError("telegram_id is required")
    for u in load_users(data_dir):
        if u.get("telegram_id") == telegram_id:
            return u
    raise CrmError(f"no user bound to telegram_id: {telegram_id!r}")


def bind_telegram_user(data_dir: Path, user_id: str, telegram_id: str) -> dict:
    """Bind a Telegram account id to a CRM user, returning the updated profile.

    Rebinding an id that already points at another user just overwrites it —
    this is a mock CRM, not an auth system.
    """
    users = load_users(data_dir)
    bound: dict | None = None
    for u in users:
        # Clear any existing binding of this telegram_id
        if u.get("telegram_id") == telegram_id:
            u["telegram_id"] = None
        # Find and bind the target user
        if u["id"] == user_id:
            u["telegram_id"] = telegram_id
            bound = u
    if bound is None:
        raise CrmError(f"unknown user_id: {user_id!r}")

    save_users(data_dir, users)
    return bound


def list_tickets(data_dir: Path, user_id: str, status: str | None = None) -> list[dict]:
    tickets = [t for t in load_tickets(data_dir) if t["user_id"] == user_id]
    if status:
        tickets = [t for t in tickets if t["status"] == status]
    return tickets


def get_ticket(data_dir: Path, ticket_id: str) -> dict:
    for t in load_tickets(data_dir):
        if t["id"] == ticket_id:
            return t
    raise CrmError(f"unknown ticket_id: {ticket_id!r}")


_VALID_STATUSES = {"open", "pending", "resolved"}


def update_ticket(
    data_dir: Path,
    ticket_id: str,
    *,
    status: str | None = None,
    note: str | None = None,
) -> dict:
    """Update a ticket's status and/or append a note to its history.

    Returns the updated ticket. Raises ``CrmError`` for an unknown ticket
    id or an invalid status.
    """
    if status is not None and status not in _VALID_STATUSES:
        raise CrmError(f"invalid status {status!r}, must be one of {sorted(_VALID_STATUSES)}")

    tickets = load_tickets(data_dir)
    updated: dict | None = None
    for t in tickets:
        if t["id"] == ticket_id:
            if status is not None:
                t["status"] = status
            if note:
                t["history"].append(
                    {
                        "author": "support",
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "text": note,
                    }
                )
            updated = t
            break
    if updated is None:
        raise CrmError(f"unknown ticket_id: {ticket_id!r}")

    save_tickets(data_dir, tickets)
    return updated
