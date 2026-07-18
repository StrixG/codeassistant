# Telegram Support Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Заменить CLI-интерфейс ассистента поддержки на Telegram-бота, переиспользуя существующее ядро (RAG + MCP-CRM + один вызов DeepSeek) без изменений.

**Architecture:** Новый пакет `support_bot/` — тонкая async-обвязка на aiogram 3 поверх `support_assistant.chat.answer_message`. Один долгоживущий `McpClient` на весь процесс бота, поднимается в startup-хуке. Все синхронные вызовы (MCP, RAG, DeepSeek) уходят в `asyncio.to_thread`, чтобы не блокировать event loop. Резолв пользователя — по `telegram_id` через два новых MCP-тула, инвариант «CRM только через MCP» сохраняется.

**Tech Stack:** Python 3.10+, aiogram 3 (long polling), MCP Python SDK (stdio), Chroma, pytest + pytest-asyncio.

Спека: `docs/superpowers/specs/2026-07-17-telegram-support-bot-design.md`

## Global Constraints

- Python `>=3.10` (см. `pyproject.toml`), синтаксис `from __future__ import annotations` в каждом новом модуле — как во всех существующих.
- Никакой модуль, кроме `assistant/config.py`, не читает `os.environ` напрямую.
- Бот никогда не открывает `users.json`/`tickets.json` напрямую — только через MCP-тулы `mcp_crm.server`.
- Один вызов DeepSeek на одно сообщение пользователя (без агентного tool-calling цикла) — реализовано в `answer_message`, не менять.
- Язык интерфейса бота и всех сообщений пользователю — русский.
- Ядро `support_assistant/chat.py` (`answer_message`, `build_context_block`, `extract_close_suggestion`, `SYSTEM_PROMPT`) переиспользуется импортом и **не модифицируется**.
- `TELEGRAM_BOT_TOKEN` в `Config` имеет дефолт `""` (не `required=True`): `Config.load()` общий для основного ассистента и `mcp_crm.server`, они не должны падать без токена. Проверку делает `support_bot/bot.py` при старте.
- Тесты: pytest, temp-директории для данных, без сети и без реального Telegram API.

---

### Task 1: CRM-функции резолва и биндинга по telegram_id

**Files:**
- Modify: `mcp_crm/crm_store.py` (добавить две функции после `get_user`)
- Modify: `tests/test_crm_store.py` (фикстура `data_dir` + новые тесты)

**Interfaces:**
- Consumes: `crm_store.load_users`, `crm_store._users_path`, `crm_store.CrmError` (существуют).
- Produces:
  - `crm_store.find_user_by_telegram_id(data_dir: Path, telegram_id: str) -> dict` — профиль или `CrmError`.
  - `crm_store.bind_telegram_user(data_dir: Path, user_id: str, telegram_id: str) -> dict` — обновлённый профиль или `CrmError`.
  - `crm_store.save_users(data_dir: Path, users: list[dict]) -> None`.

- [ ] **Step 1: Написать падающие тесты**

Сначала добавить `telegram_id` в фикстуру. В `tests/test_crm_store.py` заменить словарь `user-1` внутри фикстуры `data_dir` на:

```python
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
```

и словарь `user-2` на:

```python
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
```

Затем в конец файла добавить:

```python
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
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

Run: `pytest tests/test_crm_store.py -v`
Expected: 6 новых тестов FAIL с `AttributeError: module 'mcp_crm.crm_store' has no attribute 'find_user_by_telegram_id'` (и то же для `bind_telegram_user`). Существующие 12 тестов — PASS.

- [ ] **Step 3: Реализовать минимально**

В `mcp_crm/crm_store.py` после `save_tickets` добавить:

```python
def save_users(data_dir: Path, users: list[dict]) -> None:
    _users_path(data_dir).write_text(
        json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8"
    )
```

После `get_user` добавить:

```python
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
        if u["id"] == user_id:
            u["telegram_id"] = telegram_id
            bound = u
            break
    if bound is None:
        raise CrmError(f"unknown user_id: {user_id!r}")

    save_users(data_dir, users)
    return bound
```

- [ ] **Step 4: Запустить тесты, убедиться что проходят**

Run: `pytest tests/test_crm_store.py -v`
Expected: PASS, 18 passed.

- [ ] **Step 5: Коммит**

```bash
git add mcp_crm/crm_store.py tests/test_crm_store.py
git commit -m "feat(crm): look up and bind CRM users by telegram_id"
```

---

### Task 2: MCP-тулы find_user_by_telegram_id и bind_telegram_user

**Files:**
- Modify: `mcp_crm/server.py` (два новых `@mcp.tool()` после `get_user`)
- Modify: `data/support/users.json` (поле `telegram_id` во всех 8 записях)

**Interfaces:**
- Consumes: `crm_store.find_user_by_telegram_id`, `crm_store.bind_telegram_user` (Task 1).
- Produces: MCP-тулы `find_user_by_telegram_id(telegram_id: str) -> str` и `bind_telegram_user(user_id: str, telegram_id: str) -> str` — JSON-строка профиля либо строка `Error: ...`, как у существующих тулов.

Тула здесь тестируется не юнит-тестом (транспорт MCP в тестах не поднимается — так же, как для существующих `get_user`/`update_ticket`), а ручной проверкой списка тулов: логика уже покрыта тестами Task 1.

- [ ] **Step 1: Добавить telegram_id в mock-CRM**

В `data/support/users.json` добавить `"telegram_id": null` последним полем каждой из 8 записей. Например, `user-1` становится:

```json
  {
    "id": "user-1",
    "name": "Анна Смирнова",
    "email": "anna.smirnova@example.com",
    "platform": "Android",
    "app_version": "1.6.20",
    "plan": "free",
    "signup_date": "2023-03-11",
    "telegram_id": null
  },
```

То же самое для `user-2` … `user-8`. Значения заполняются только через `/start`-биндинг в боте — руками ничего не прописываем.

- [ ] **Step 2: Добавить два MCP-тула**

В `mcp_crm/server.py` после функции `get_user` добавить:

```python
@mcp.tool()
def find_user_by_telegram_id(telegram_id: str) -> str:
    """Return the CRM profile bound to a Telegram account id as JSON.

    Used by the Telegram bot to work out who is writing to it before
    answering. Returns an error string if no user is bound to that id —
    the caller is expected to ask the person for their CRM user id and
    then call bind_telegram_user.
    """
    try:
        user = crm_store.find_user_by_telegram_id(_DATA_DIR, telegram_id)
        return json.dumps(user, ensure_ascii=False)
    except CrmError as e:
        return f"Error: {e}"


@mcp.tool()
def bind_telegram_user(user_id: str, telegram_id: str) -> str:
    """Bind a Telegram account id to a CRM user. Returns the updated
    profile as JSON.

    Call this once the person has told you their CRM user id and it has
    been verified with get_user. Rebinding overwrites any previous
    binding.
    """
    try:
        user = crm_store.bind_telegram_user(_DATA_DIR, user_id, telegram_id)
        return json.dumps(user, ensure_ascii=False)
    except CrmError as e:
        return f"Error: {e}"
```

- [ ] **Step 3: Проверить, что сервер поднимается и отдаёт 6 тулов**

Run:

```bash
python - <<'PY'
from assistant.core.mcp_client import McpClient
from mcp_crm.server import default_server_params

with McpClient(default_server_params()) as mcp:
    print(sorted(t.name for t in mcp.tools))
    print(mcp.call_tool("find_user_by_telegram_id", {"telegram_id": "424242"}))
PY
```

Expected:
```
['bind_telegram_user', 'find_user_by_telegram_id', 'get_ticket', 'get_user', 'list_tickets', 'update_ticket']
Error: no user bound to telegram_id: '424242'
```

- [ ] **Step 4: Убедиться, что существующие тесты не сломаны**

Run: `pytest tests/test_crm_store.py tests/test_support_context.py tests/test_support_chat_graceful.py -v`
Expected: PASS, всё зелёное.

- [ ] **Step 5: Коммит**

```bash
git add mcp_crm/server.py data/support/users.json
git commit -m "feat(mcp-crm): expose telegram lookup and binding tools"
```

---

### Task 3: Конфиг и зависимости бота

**Files:**
- Modify: `assistant/config.py` (поле `telegram_bot_token`)
- Modify: `.env.example`
- Modify: `pyproject.toml` (пакет `support_bot*`, зависимость `aiogram`)
- Test: `tests/test_config_telegram.py` (создать)

**Interfaces:**
- Produces: `Config.telegram_bot_token: str` (дефолт `""`, читается из `TELEGRAM_BOT_TOKEN`).

- [ ] **Step 1: Написать падающий тест**

Создать `tests/test_config_telegram.py`:

```python
"""Config must surface the Telegram bot token without making it mandatory.

``Config.load()`` is shared with the main assistant and with
``mcp_crm.server``; requiring the bot token there would break both.
"""

from __future__ import annotations

from assistant.config import Config


def test_telegram_token_read_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TARGET_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")

    cfg = Config.load()

    assert cfg.telegram_bot_token == "123:ABC"


def test_telegram_token_defaults_to_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("TARGET_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    cfg = Config.load()

    assert cfg.telegram_bot_token == ""
```

- [ ] **Step 2: Запустить тест, убедиться что падает**

Run: `pytest tests/test_config_telegram.py -v`
Expected: FAIL с `AttributeError: 'Config' object has no attribute 'telegram_bot_token'`.

- [ ] **Step 3: Добавить поле в Config**

В `assistant/config.py` после `support_data_dir` в теле датакласса добавить:

```python
    # Telegram support bot (Day 34). Defaulted to "" rather than required:
    # Config.load() is shared with the main assistant and mcp_crm.server,
    # neither of which needs a bot token. support_bot/bot.py checks it.
    telegram_bot_token: str = ""
```

И в `load()`, после строки `support_data_dir=...`:

```python
            telegram_bot_token=_get("TELEGRAM_BOT_TOKEN", ""),
```

- [ ] **Step 4: Запустить тест, убедиться что проходит**

Run: `pytest tests/test_config_telegram.py -v`
Expected: PASS, 2 passed.

- [ ] **Step 5: Дописать .env.example**

В конец `.env.example` добавить:

```
# --- Telegram support bot (Day 34) ---
# Токен от @BotFather; нужен только для support_bot, не для остальных команд.
# TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
```

- [ ] **Step 6: Добавить aiogram и пакет support_bot в pyproject**

В `pyproject.toml` заменить строку с `mcp = [...]` на:

```toml
mcp = ["mcp>=1.2", "prompt_toolkit>=3.0"]
bot = ["aiogram>=3.13"]
```

и строку `include = [...]` на:

```toml
include = ["assistant*", "mcp_crm*", "support_assistant*", "support_bot*"]
```

- [ ] **Step 7: Установить и проверить импорт**

Run: `pip install -e ".[mcp,bot,dev]" && python -c "import aiogram; print(aiogram.__version__)"`
Expected: версия 3.x, например `3.13.1`.

- [ ] **Step 8: Коммит**

```bash
git add assistant/config.py .env.example pyproject.toml tests/test_config_telegram.py
git commit -m "feat(config): add TELEGRAM_BOT_TOKEN and aiogram dependency"
```

---

### Task 4: Async-слой доступа к CRM через MCP

**Files:**
- Create: `support_bot/__init__.py`
- Create: `support_bot/crm.py`
- Test: `tests/test_support_bot_crm.py` (создать)

**Interfaces:**
- Consumes: `assistant.core.mcp_client.McpClient` (метод `call_tool(name, arguments) -> str`), MCP-тулы из Task 2.
- Produces (всё в `support_bot/crm.py`):
  - `class CrmUnavailable(RuntimeError)`
  - `async def find_user_by_telegram_id(mcp, telegram_id: int) -> dict | None`
  - `async def get_user(mcp, user_id: str) -> dict | None`
  - `async def bind_telegram_user(mcp, user_id: str, telegram_id: int) -> dict`
  - `async def list_open_tickets(mcp, user_id: str) -> list[dict]`
  - `async def update_ticket(mcp, ticket_id: str, *, status: str, note: str) -> dict`

Примечание к спеке: спека перечисляла три файла в `support_bot/` (`bot.py`, `handlers.py`, `binding.py`). Слой `crm.py` выделен отдельно, чтобы хендлеры не занимались ни `to_thread`, ни разбором JSON, ни различением «не найдено» и «MCP упала» — одна ответственность на файл.

Ключевое различение: тулы возвращают строку `Error: ...` и для «не найдено», и для сломанных аргументов, а `McpClient.call_tool` бросает `RuntimeError` при отказе транспорта. Функции-lookup (`find_user_by_telegram_id`, `get_user`) отдают `None` на `Error:` — это штатное «нет такого»; всё остальное поднимается как `CrmUnavailable`.

- [ ] **Step 1: Написать падающие тесты**

Создать `tests/test_support_bot_crm.py`:

```python
"""Tests for the bot's async CRM layer over MCP.

A fake stands in for McpClient: the real one spawns a stdio subprocess,
which these tests have no need for — only the call_tool contract matters
(returns a string, "Error: ..." for a tool-level failure, raises
RuntimeError when the transport itself is down).
"""

from __future__ import annotations

import json

import pytest

from support_bot import crm
from support_bot.crm import CrmUnavailable


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


@pytest.mark.asyncio
async def test_find_user_by_telegram_id_returns_profile():
    mcp = FakeMcp({"find_user_by_telegram_id": USER_JSON})

    user = await crm.find_user_by_telegram_id(mcp, 111)

    assert user["id"] == "user-1"
    assert mcp.calls == [("find_user_by_telegram_id", {"telegram_id": "111"})]


@pytest.mark.asyncio
async def test_find_user_by_telegram_id_returns_none_when_unbound():
    mcp = FakeMcp({"find_user_by_telegram_id": "Error: no user bound to telegram_id: '111'"})

    assert await crm.find_user_by_telegram_id(mcp, 111) is None


@pytest.mark.asyncio
async def test_find_user_by_telegram_id_raises_when_mcp_is_down():
    mcp = FakeMcp({"find_user_by_telegram_id": RuntimeError("stdio closed")})

    with pytest.raises(CrmUnavailable):
        await crm.find_user_by_telegram_id(mcp, 111)


@pytest.mark.asyncio
async def test_get_user_returns_none_for_unknown_id():
    mcp = FakeMcp({"get_user": "Error: unknown user_id: 'user-999'"})

    assert await crm.get_user(mcp, "user-999") is None


@pytest.mark.asyncio
async def test_bind_telegram_user_passes_string_id():
    mcp = FakeMcp({"bind_telegram_user": USER_JSON})

    user = await crm.bind_telegram_user(mcp, "user-1", 111)

    assert user["id"] == "user-1"
    assert mcp.calls == [("bind_telegram_user", {"user_id": "user-1", "telegram_id": "111"})]


@pytest.mark.asyncio
async def test_bind_telegram_user_raises_on_error_string():
    mcp = FakeMcp({"bind_telegram_user": "Error: unknown user_id: 'user-999'"})

    with pytest.raises(CrmUnavailable):
        await crm.bind_telegram_user(mcp, "user-999", 111)


@pytest.mark.asyncio
async def test_list_open_tickets_filters_by_status():
    tickets = json.dumps([{"id": "ticket-1001", "status": "open"}])
    mcp = FakeMcp({"list_tickets": tickets})

    result = await crm.list_open_tickets(mcp, "user-1")

    assert [t["id"] for t in result] == ["ticket-1001"]
    assert mcp.calls == [("list_tickets", {"user_id": "user-1", "status": "open"})]


@pytest.mark.asyncio
async def test_update_ticket_sends_status_and_note():
    mcp = FakeMcp({"update_ticket": json.dumps({"id": "ticket-1001", "status": "resolved"})})

    updated = await crm.update_ticket(mcp, "ticket-1001", status="resolved", note="Готово")

    assert updated["status"] == "resolved"
    assert mcp.calls == [
        ("update_ticket", {"ticket_id": "ticket-1001", "status": "resolved", "note": "Готово"})
    ]
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

Run: `pytest tests/test_support_bot_crm.py -v`
Expected: FAIL со сборкой — `ModuleNotFoundError: No module named 'support_bot'`.

- [ ] **Step 3: Создать пакет и слой доступа**

Создать `support_bot/__init__.py`:

```python
"""Telegram front-end for the Element support assistant."""
```

Создать `support_bot/crm.py`:

```python
"""Async CRM access for the bot, over the MCP stdio server.

Everything the bot knows about users and tickets goes through here, and
here it goes through MCP — the bot never opens users.json/tickets.json
itself. McpClient is synchronous (it marshals onto its own loop in a
background thread), so every call is pushed to a worker thread: aiogram
runs one event loop for all chats, and a blocking MCP round-trip would
stall every other user's message.

Lookups return None when the tool reports the record simply isn't there;
anything else — a broken argument, a dead transport — raises
CrmUnavailable, which handlers turn into an apology to the user.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from assistant.core.mcp_client import McpClient


class CrmUnavailable(RuntimeError):
    """The CRM could not answer: bad arguments, or MCP itself is down."""


async def _call(mcp: McpClient, tool: str, arguments: dict) -> str:
    try:
        return await asyncio.to_thread(mcp.call_tool, tool, arguments)
    except RuntimeError as e:  # transport failure from McpClient.call_tool
        raise CrmUnavailable(f"MCP call {tool} failed: {e}") from e


async def _call_json(mcp: McpClient, tool: str, arguments: dict) -> Any:
    raw = await _call(mcp, tool, arguments)
    if raw.startswith("Error:"):
        raise CrmUnavailable(raw)
    return json.loads(raw)


async def _lookup(mcp: McpClient, tool: str, arguments: dict) -> dict | None:
    """Like _call_json, but a tool-level error means "not found", not a fault."""
    raw = await _call(mcp, tool, arguments)
    if raw.startswith("Error:"):
        return None
    return json.loads(raw)


async def find_user_by_telegram_id(mcp: McpClient, telegram_id: int) -> dict | None:
    return await _lookup(mcp, "find_user_by_telegram_id", {"telegram_id": str(telegram_id)})


async def get_user(mcp: McpClient, user_id: str) -> dict | None:
    return await _lookup(mcp, "get_user", {"user_id": user_id})


async def bind_telegram_user(mcp: McpClient, user_id: str, telegram_id: int) -> dict:
    return await _call_json(
        mcp, "bind_telegram_user", {"user_id": user_id, "telegram_id": str(telegram_id)}
    )


async def list_open_tickets(mcp: McpClient, user_id: str) -> list[dict]:
    return await _call_json(mcp, "list_tickets", {"user_id": user_id, "status": "open"})


async def update_ticket(mcp: McpClient, ticket_id: str, *, status: str, note: str) -> dict:
    return await _call_json(
        mcp, "update_ticket", {"ticket_id": ticket_id, "status": status, "note": note}
    )
```

- [ ] **Step 4: Запустить тесты, убедиться что проходят**

Run: `pytest tests/test_support_bot_crm.py -v`
Expected: PASS, 8 passed.

Если тесты падают с `async def functions are not natively supported`, добавить в `pyproject.toml` секцию:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

и перезапустить.

- [ ] **Step 5: Коммит**

```bash
git add support_bot/__init__.py support_bot/crm.py tests/test_support_bot_crm.py pyproject.toml
git commit -m "feat(bot): add async CRM access layer over MCP"
```

---

### Task 5: Логика привязки Telegram-аккаунта к CRM-юзеру

**Files:**
- Create: `support_bot/binding.py`
- Test: `tests/test_support_bot_binding.py` (создать)

**Interfaces:**
- Consumes: `support_bot.crm.find_user_by_telegram_id`, `crm.get_user`, `crm.bind_telegram_user`, `crm.CrmUnavailable` (Task 4).
- Produces (всё в `support_bot/binding.py`):
  - `class PendingBindings` с методами `mark_waiting(telegram_id: int) -> None`, `is_waiting(telegram_id: int) -> bool`, `clear(telegram_id: int) -> None`
  - `BINDING_PROMPT: str`
  - `async def try_bind(mcp, pending: PendingBindings, telegram_id: int, text: str) -> tuple[str, dict | None]` — возвращает `(текст ответа, привязанный профиль или None)`

`PendingBindings` — in-memory `set`: юзеров восемь, состояние живёт ровно столько же, сколько процесс бота, переживать рестарт ему незачем (после рестарта `find_user_by_telegram_id` и так найдёт уже привязанного пользователя).

- [ ] **Step 1: Написать падающие тесты**

Создать `tests/test_support_bot_binding.py`:

```python
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
async def test_try_bind_strips_whitespace_from_user_id():
    mcp = FakeMcp({"get_user": USER_JSON, "bind_telegram_user": USER_JSON})
    pending = PendingBindings()
    pending.mark_waiting(111)

    await try_bind(mcp, pending, 111, "  user-1  ")

    assert mcp.calls[0] == ("get_user", {"user_id": "user-1"})
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

Run: `pytest tests/test_support_bot_binding.py -v`
Expected: FAIL с `ModuleNotFoundError: No module named 'support_bot.binding'`.

- [ ] **Step 3: Реализовать биндинг**

Создать `support_bot/binding.py`:

```python
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
```

- [ ] **Step 4: Запустить тесты, убедиться что проходят**

Run: `pytest tests/test_support_bot_binding.py -v`
Expected: PASS, 6 passed.

- [ ] **Step 5: Коммит**

```bash
git add support_bot/binding.py tests/test_support_bot_binding.py
git commit -m "feat(bot): bind telegram accounts to CRM users"
```

---

### Task 6: Хендлеры бота

**Files:**
- Create: `support_bot/handlers.py`
- Test: `tests/test_support_bot_handlers.py` (создать)

**Interfaces:**
- Consumes: `support_bot.crm` (Task 4), `support_bot.binding.PendingBindings`, `binding.try_bind`, `binding.BINDING_PROMPT` (Task 5), `support_assistant.chat.answer_message` и `SupportTurnResult` (существуют, не менять), `assistant.core.rag.RagSearcher`, `assistant.core.llm.DeepSeekClient`.
- Produces:
  - `router: aiogram.Router` — с тремя хендлерами.
  - `def close_keyboard(ticket_id: str) -> InlineKeyboardMarkup`
  - `async def answer_question(question, *, telegram_id, user, mcp, rag, llm) -> SupportTurnResult`

Хендлеры получают зависимости (`mcp`, `rag`, `llm`, `pending`) через `workflow_data` диспетчера aiogram: имена параметров хендлера совпадают с ключами, которые Task 7 положит в `dp`. Тестами покрывается `answer_question` и `close_keyboard` — фактическая логика; сами `@router.message`-обёртки остаются тривиальными и проверяются ручным прогоном в Task 7.

- [ ] **Step 1: Написать падающие тесты**

Создать `tests/test_support_bot_handlers.py`:

```python
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
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

Run: `pytest tests/test_support_bot_handlers.py -v`
Expected: FAIL с `ModuleNotFoundError: No module named 'support_bot.handlers'`.

- [ ] **Step 3: Реализовать хендлеры**

Создать `support_bot/handlers.py`:

```python
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
```

- [ ] **Step 4: Запустить тесты, убедиться что проходят**

Run: `pytest tests/test_support_bot_handlers.py -v`
Expected: PASS, 4 passed.

- [ ] **Step 5: Коммит**

```bash
git add support_bot/handlers.py tests/test_support_bot_handlers.py
git commit -m "feat(bot): add start, question and ticket-close handlers"
```

---

### Task 7: Точка входа бота и запуск

**Files:**
- Create: `support_bot/bot.py`
- Modify: `README_SUPPORT.md`

**Interfaces:**
- Consumes: `support_bot.handlers.router` (Task 6), `support_bot.binding.PendingBindings` (Task 5), `Config.telegram_bot_token` (Task 3), `mcp_crm.server.default_server_params` (существует).
- Produces: `python -m support_bot.bot` — работающий бот.

- [ ] **Step 1: Написать точку входа**

Создать `support_bot/bot.py`:

```python
"""Telegram entry point for the support assistant.

Run:  python -m support_bot.bot

One McpClient (one mcp_crm.server subprocess) serves every chat: the CRM
tools are stateless JSON reads and writes, so a single process keeps up,
and handlers push their calls to worker threads anyway. The client is
started before polling and stopped after it, so the subprocess never
outlives the bot.

The RAG index and the DeepSeek client are likewise built once and shared
— loading the embedding model per message would cost seconds each time.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace as replace_cfg

from aiogram import Bot, Dispatcher

from assistant.config import Config
from assistant.core.llm import DeepSeekClient
from assistant.core.mcp_client import McpClient
from assistant.core.rag import RagSearcher
from mcp_crm.server import default_server_params as crm_server_params
from support_bot.binding import PendingBindings
from support_bot.handlers import router

log = logging.getLogger(__name__)


def _build_rag(cfg: Config) -> RagSearcher:
    """Point the shared searcher at the support collection, as the CLI does."""
    support_cfg = replace_cfg(cfg, chroma_collection=cfg.support_chroma_collection)
    rag = RagSearcher(support_cfg)
    if rag.count() == 0:
        raise RuntimeError(
            "Индекс support_kb пуст. Сначала: python -m support_assistant.index_support_kb"
        )
    return rag


async def run_bot() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = Config.load()
    if not cfg.telegram_bot_token:
        log.error("TELEGRAM_BOT_TOKEN не задан. Добавьте его в .env (см. .env.example).")
        return 1

    try:
        rag = _build_rag(cfg)
    except Exception as e:
        log.error("%s", e)
        return 1

    llm = DeepSeekClient(cfg)
    pending = PendingBindings()

    bot = Bot(token=cfg.telegram_bot_token)
    dp = Dispatcher()
    dp.include_router(router)

    mcp = McpClient(crm_server_params()).start()
    # Names here must match the handler parameter names in handlers.py.
    dp["mcp"] = mcp
    dp["rag"] = rag
    dp["llm"] = llm
    dp["pending"] = pending

    log.info("Бот запущен, MCP-тулы: %s", ", ".join(sorted(t.name for t in mcp.tools)))
    try:
        await dp.start_polling(bot)
    finally:
        mcp.stop()
        await bot.session.close()
        log.info("Бот остановлен.")
    return 0


def main() -> int:
    try:
        return asyncio.run(run_bot())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Проверить, что бот стартует и падает понятно без токена**

Run: `TELEGRAM_BOT_TOKEN= python -m support_bot.bot; echo "exit=$?"`
Expected:
```
... ERROR support_bot.bot: TELEGRAM_BOT_TOKEN не задан. Добавьте его в .env (см. .env.example).
exit=1
```

- [ ] **Step 3: Ручной прогон с настоящим ботом**

Предусловие: токен от [@BotFather](https://t.me/BotFather) прописан в `.env` как `TELEGRAM_BOT_TOKEN`, индекс построен (`python -m support_assistant.index_support_kb`).

Run: `python -m support_bot.bot`

Проверить в Telegram по шагам:
1. `/start` → бот просит идентификатор (профиль ещё не привязан).
2. Отправить `user-999` → «Пользователь user-999 не найден», бот всё ещё ждёт id.
3. Отправить `user-2` → «Готово, Дмитрий Волков! Этот чат связан с профилем user-2».
4. Спросить «Не могу войти, 2FA код не подходит» → осмысленный ответ со ссылкой на открытый тикет пользователя; в логе бота видна строка `tg=... user=user-2 tickets=... rag=[...]`.
5. Написать «Спасибо, всё заработало» → ответ + сообщение с кнопками закрытия тикета.
6. Нажать «✅ Закрыть тикет …» → сообщение меняется на «Тикет … закрыт. Спасибо!», кнопки исчезают; в `data/support/tickets.json` у тикета `status: "resolved"` и новая запись в `history`.
7. Повторить `/start` → бот здоровается по имени, идентификатор больше не спрашивает.
8. `Ctrl+C` → в логе «Бот остановлен», процесс `mcp_crm.server` не остаётся висеть (`pgrep -f mcp_crm.server` пусто).

- [ ] **Step 4: Прогнать весь набор тестов**

Run: `pytest -v`
Expected: PASS, всё зелёное — ни один существующий тест не задет.

- [ ] **Step 5: Обновить README_SUPPORT.md**

В `README_SUPPORT.md`:

1. В разделе «Новые файлы» после блока `support_assistant/...` добавить:

```
support_bot/crm.py               async-доступ к CRM через MCP (to_thread + разбор JSON)
support_bot/binding.py           привязка telegram_id к CRM-профилю
support_bot/handlers.py          хендлеры aiogram: /start, вопрос, закрытие тикета
support_bot/bot.py               точка входа Telegram-бота (long polling)

tests/test_support_bot_crm.py       тесты async-слоя CRM на fake-MCP
tests/test_support_bot_binding.py   тесты привязки telegram_id → user_id
tests/test_support_bot_handlers.py  тесты сборки контекста вопроса и клавиатуры
tests/test_config_telegram.py       тесты чтения TELEGRAM_BOT_TOKEN
```

2. В разделе «Запуск с нуля» после шага 4 добавить:

````
# 5. Telegram-бот (вместо CLI): токен от @BotFather в .env как TELEGRAM_BOT_TOKEN
pip install -e ".[mcp,bot,dev]"
python -m support_bot.bot
````

и уточнить шаг 1: `pip install -e ".[mcp,bot,dev]"`.

3. В таблицу «Переменные окружения» добавить строку:

```
| `TELEGRAM_BOT_TOKEN` | для бота | — | токен от @BotFather; нужен только `support_bot`, остальные команды работают без него |
```

4. В конце раздела «Архитектура» добавить абзац:

```
Telegram-бот (`support_bot`) — альтернативный фронтенд к тому же ядру:
`answer_message` из `support_assistant.chat` переиспользуется как есть,
поэтому пайплайн (MCP → RAG → один вызов DeepSeek) идентичен CLI. Отличий
два: пользователь определяется по `telegram_id` через MCP-тулы
`find_user_by_telegram_id`/`bind_telegram_user` вместо флага `--user`, а
закрытие тикета подтверждается inline-кнопкой вместо ввода `y/N`. Один
`McpClient` и один `RagSearcher` живут на весь процесс бота и делятся
между всеми чатами; блокирующие вызовы уходят в `asyncio.to_thread`,
чтобы медленный ответ одному пользователю не задерживал остальных.
```

- [ ] **Step 6: Коммит**

```bash
git add support_bot/bot.py README_SUPPORT.md
git commit -m "feat(bot): add telegram entry point and document the bot"
```
