# Telegram-бот поддержки Element Android

Дата: 2026-07-17

## Цель

Заменить CLI-интерфейс `support_assistant.chat` на Telegram-бота — тот же
RAG + MCP-CRM + DeepSeek пайплайн, но реалистичный канал доставки для
конечного пользователя вместо терминала. Ядро (`answer_message`,
`build_context_block`, `extract_close_suggestion` в
`support_assistant/chat.py`) переиспользуется без изменений; меняется
только обвязка, которая раньше была REPL-циклом на `prompt_toolkit`.

## Архитектура

```
Telegram ──aiogram──▶ support_bot.bot (long polling)
                          │
              ┌───────────┼─────────────────┐
              ▼           ▼                 ▼
     find_user_by_       RagSearcher    DeepSeekClient
     telegram_id /        (support_kb)   (тот же chat())
     bind_telegram_user
     (через McpClient,
      один на весь бот)
              │
              ▼
     mcp_crm.server (MCP stdio) ── data/support/users.json + tickets.json
```

Один долгоживущий `McpClient` поднимается в `on_startup` и глушится в
`on_shutdown` — как в CLI, но живёт на весь процесс бота, а не на одну
сессию одного пользователя. Все синхронные вызовы (`McpClient.call_tool`,
`RagSearcher.search`, `DeepSeekClient.chat`, `answer_message`)
оборачиваются в `asyncio.to_thread(...)` внутри хендлеров aiogram —
иначе они блокируют единственный event loop, и другие пользователи бота
встают в очередь на секунды на каждый DeepSeek/MCP-вызов.

Библиотека: **aiogram 3**, long polling (без вебхука — не нужен для
демо/локального запуска).

## Новые и изменённые файлы

### `mcp_crm` (расширение существующего MCP-сервера)

- `crm_store.py`: две новые чистые функции, юнит-тестируемые как
  остальные:
  - `find_user_by_telegram_id(data_dir, telegram_id) -> dict` — поиск по
    полю `telegram_id`, `CrmError` если не найден.
  - `bind_telegram_user(data_dir, user_id, telegram_id) -> dict` —
    записывает `telegram_id` в запись юзера в `users.json`, возвращает
    обновлённый профиль. `CrmError`, если `user_id` не существует.
    Перезаписывает без проверки уникальности — mock CRM, не продовая
    система.
- `server.py`: два новых `@mcp.tool()` поверх этих функций —
  `find_user_by_telegram_id`, `bind_telegram_user` — по образцу
  существующих `get_user`/`update_ticket` (JSON-строка или
  `Error: ...`).

### `data/support/users.json`

Каждой из 8 записей добавляется `"telegram_id": null`. Заполняется
только через `/start`-биндинг в боте, руками не прописывается.

### `support_bot/` (новый пакет)

- `__init__.py`
- `bot.py` — точка входа: `Config.load()`, поднимает общий `McpClient`
  (тот же `mcp_crm.server.default_server_params()`, что и в CLI) и
  `RagSearcher`/`DeepSeekClient` в startup-хуке, регистрирует роутеры,
  `dp.start_polling(bot)`. Запуск: `python -m support_bot.bot`.
- `handlers.py` — три сценария:
  - `/start` — резолвит `telegram_id` через `find_user_by_telegram_id`;
    найден → приветствие с именем; не найден → просит ввести `user_id`.
  - Текстовое сообщение от **непривязанного** чата — трактуется как
    ввод `user_id` для биндинга: `get_user` (валидация) →
    `bind_telegram_user` → подтверждение или сообщение об ошибке
    (неизвестный `user_id`), без падения FSM.
  - Текстовое сообщение от **привязанного** чата — обычный вопрос:
    `get_user` + `list_tickets(status=open)` + `rag.search(top_k=4)`
    параллельно (`asyncio.gather`, каждый в `to_thread`) → `to_thread`
    `answer_message` → ответ текстом. Если есть `ticket_suggested` —
    отдельное сообщение с inline-кнопками
    `✅ Закрыть тикет {id}` / `Не сейчас`.
  - Callback `ticket_close:{yes,no}:{ticket_id}` — `yes` вызывает
    `update_ticket` через MCP и редактирует сообщение на
    "Тикет {id} закрыт", убирая кнопки; `no` просто убирает кнопки.
- `binding.py` — состояние "жду `user_id` от этого чата, чтобы
  привязать": простой in-memory `set[int]` ожидающих telegram_id
  (юзеров всего 8 — полноценный FSM aiogram избыточен).

### Конфигурация

- `TELEGRAM_BOT_TOKEN` — новая обязательная переменная в
  `.env`/`.env.example`, читается через `assistant.config.Config` (как
  всё остальное — бот не трогает `os.environ` напрямую).
- `pyproject.toml`: `support_bot*` в `packages.find.include`, `aiogram`
  в зависимостях.

## Data flow одного сообщения

1. `message.from_user.id` → `to_thread(mcp.call_tool,
   "find_user_by_telegram_id", ...)`.
2. Не найден → режим биндинга (см. выше).
3. Найден → параллельно `get_user`, `list_tickets(status=open)`,
   `rag.search`.
4. `to_thread(answer_message, ...)` — без изменений в самой функции,
   включая graceful-деградацию при `LlmError`.
5. Ответ текстом; при `ticket_suggested` — второе сообщение с
   inline-кнопками.
6. Callback подтверждения → `update_ticket` через MCP, редактирование
   сообщения.

Отладочная информация (какие MCP-тулы вызваны, какие RAG-чанки попали в
контекст — `_print_debug` в CLI) пользователю в Telegram не отправляется;
логируется в stdout процесса бота тем же способом, что сейчас идёт в
консоль CLI.

## Обработка ошибок

- DeepSeek недоступен → `answer_message` уже возвращает дружелюбное
  извинение вместо исключения — реюз без изменений.
- MCP-вызов упал (`RuntimeError`) → пользователю "сервис временно
  недоступен, попробуйте позже" + лог в stdout.
- Неизвестный `user_id` при биндинге → "такой user_id не найден,
  проверьте и попробуйте снова", чат остаётся в режиме ожидания
  биндинга.
- Необработанное исключение в хендлере → aiogram error-хендлер логирует
  и отправляет нейтральное "что-то пошло не так", бот не падает целиком.

## Тесты

- `tests/test_crm_store.py` — новые кейсы для `find_user_by_telegram_id`
  (найден/не найден) и `bind_telegram_user` (успех/неизвестный
  `user_id`), на temp JSON, как существующие тесты.
- `tests/test_support_bot_binding.py` (новый) — юнит-тест логики
  биндинга (текст → `user_id` → успех/ошибка), вызывая функции-обработчики
  напрямую с mock `Message`/`McpClient`, без реального Telegram API.
- Основной диалоговый сценарий (вопрос → ответ) через настоящий
  aiogram update-цикл не тестируется — ядро уже покрыто
  `test_support_context.py`/`test_support_chat_graceful.py`; в боте
  только тонкая обвязка над ним.

## Вне рамок

- Настоящая аутентификация/безопасность биндинга (любой, кто знает
  `user_id`, может привязать его к себе) — приемлемо для mock-демо,
  не для продакшена.
- Webhook-деплой, docker, множественные инстансы бота.
- Локализация интерфейса бота (только русский, как весь остальной
  ассистент).
