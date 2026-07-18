# Ассистент поддержки Element Android (День 33)

Мини-сервис поддержки пользователей мессенджера Element (Android):
отвечает на вопросы по продукту через RAG (FAQ + документация), знает
контекст конкретного пользователя/тикета через отдельный MCP-сервер CRM
и отвечает одним вызовом DeepSeek. Построен поверх существующей
инфраструктуры проекта (индексатор, чанкер, embeddings, DeepSeek-клиент,
MCP-клиент) — ничего из этого не скопировано, всё переиспользуется
импортом.

## Архитектура

```
                         ┌─────────────────────────┐
                         │  support_assistant.chat │   CLI: python -m support_assistant.chat --user user-1
                         │       (REPL-цикл)        │
                         └───────────┬──────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                             │
        ▼                            ▼                             ▼
┌───────────────┐           ┌────────────────┐           ┌──────────────────┐
│  MCP-клиент    │  stdio    │  RagSearcher    │  Chroma   │  DeepSeekClient   │
│ (assistant.core│──────────▶│ (assistant.core │──────────▶│ (assistant.core   │
│  .mcp_client)  │           │      .rag)      │           │      .llm)        │
└───────┬────────┘           └────────┬────────┘           └─────────┬─────────┘
        │ MCP-протокол                │ top-k поиск                  │ один chat()-вызов
        ▼                             ▼                               │ в контексте — профиль,
┌────────────────┐           ┌──────────────────┐                    │ тикеты, чанки RAG
│ mcp_crm.server  │           │ Chroma collection │                   ▼
│ (FastMCP, stdio)│           │   "support_kb"     │           ┌─────────────┐
│  get_user        │          │ (faq.md +          │           │  DeepSeek   │
│  list_tickets     │         │  product_guide.md)  │          │  API        │
│  get_ticket        │        └──────────────────┘             └─────────────┘
│  update_ticket      │
└──────────┬────────────┘
           │ читает/пишет JSON
           ▼
┌────────────────────────┐
│ data/support/users.json │
│ data/support/tickets.json│  ← mock-CRM, единственная точка правды
└────────────────────────┘
```

Ключевой инвариант: **CRM-данные ассистент видит только через MCP**
(`get_user`, `list_tickets`, `get_ticket`, `update_ticket`) — `chat.py`
ни разу не открывает `users.json`/`tickets.json` напрямую. Обновление
тикета (`update_ticket`) тоже идёт через MCP, после подтверждения
пользователем в CLI.

RAG-поиск ограничен отдельной коллекцией Chroma `support_kb`, не
пересекающейся с основной коллекцией `element_docs` (код + доки
Element Android из первого дня проекта) — они физически разные
коллекции в одном и том же `.chroma`.

За один вопрос пользователя ассистент делает **один** вызов DeepSeek
(без агентного tool-calling цикла): MCP и RAG выполняются детерминированно
до вызова модели, их результат просто попадает в промпт.

### Переиспользованные компоненты

| Модуль | Что переиспользуется |
|---|---|
| `assistant.config.Config` | добавлены 2 новых поля (`support_chroma_collection`, `support_data_dir`) с дефолтами — старый код не задет |
| `assistant.core.embeddings.get_embedder` | тот же локальный e5-эмбеддер, что и в основном индексе |
| `assistant.indexer.chunker.chunk_markdown` | то же чанкование по заголовкам markdown |
| `assistant.core.rag.RagSearcher` | тот же класс поиска, направлен на коллекцию `support_kb` через `dataclasses.replace(cfg, chroma_collection=...)` |
| `assistant.core.llm.DeepSeekClient` / `LlmError` | тот же клиент DeepSeek с ретраями |
| `assistant.core.mcp_client.McpClient` | тот же синхронный обёртка над MCP stdio-сессией, теперь подключена к `mcp_crm.server` вместо `assistant.mcp_server.server` |

### Новые файлы

```
data/support/faq.md              15+ вопрос-ответов по Element Android
data/support/product_guide.md    пользовательская документация (RAG-корпус)
data/support/users.json          mock CRM: 8 пользователей
data/support/tickets.json        mock CRM: 13 тикетов

mcp_crm/crm_store.py             чистые read/write функции над JSON (юнит-тестируемые)
mcp_crm/server.py                MCP stdio-сервер (FastMCP): get_user/list_tickets/get_ticket/update_ticket

support_assistant/index_support_kb.py   индексация faq.md + product_guide.md в коллекцию support_kb
support_assistant/chat.py               CLI-чат ассистента поддержки

tests/test_crm_store.py          тесты CRM-тулов на тестовых JSON
tests/test_support_context.py    тесты сборки контекста и парсинга SUGGEST_CLOSE
tests/test_support_chat_graceful.py   graceful-ответ при недоступном DeepSeek

support_bot/crm.py               async-доступ к CRM через MCP (to_thread + разбор JSON)
support_bot/binding.py           привязка telegram_id к CRM-профилю
support_bot/handlers.py          хендлеры aiogram: /start, вопрос, закрытие тикета
support_bot/bot.py               точка входа Telegram-бота (long polling)

tests/test_support_bot_crm.py       тесты async-слоя CRM на fake-MCP
tests/test_support_bot_binding.py   тесты привязки telegram_id → user_id
tests/test_support_bot_handlers.py  тесты сборки контекста вопроса и клавиатуры
tests/test_config_telegram.py       тесты чтения TELEGRAM_BOT_TOKEN
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

## Запуск с нуля

```bash
# 1. Зависимости (mcp-экстра нужна и основному ассистенту, и этому сервису)
pip install -e ".[mcp,bot,dev]"

# 2. .env — если ещё не настроен для основного ассистента
cp .env.example .env
# заполнить TARGET_REPO_PATH (клон element-android) и DEEPSEEK_API_KEY

# 3. Индексация support-базы знаний (отдельно от основного `assistant index`)
python -m support_assistant.index_support_kb

# 4. Запуск чата поддержки для конкретного пользователя
python -m support_assistant.chat --user user-1

# 5. Telegram-бот (вместо CLI): токен от @BotFather в .env как TELEGRAM_BOT_TOKEN
pip install -e ".[mcp,bot,dev]"
python -m support_bot.bot
```

`mcp_crm.server` отдельно руками запускать не нужно — `support_assistant.chat`
поднимает его сам как stdio-подпроцесс через `McpClient`, так же как
основной `assistant chat` поднимает `assistant.mcp_server.server`.

## Переменные окружения

Все — в `.env`, читаются только через `assistant.config.Config` (ничто
другое `os.environ` напрямую не трогает — так уже было заведено в
основном ассистенте, и это правило сохранено).

| Переменная | Обязательна | Умолчание | Назначение |
|---|---|---|---|
| `DEEPSEEK_API_KEY` | да | — | ключ DeepSeek, только из окружения |
| `TARGET_REPO_PATH` | да | — | путь к клону element-android (используется основным ассистентом; support-ассистент его не читает, но `Config.load()` общий и требует эту переменную) |
| `SUPPORT_CHROMA_COLLECTION` | нет | `support_kb` | имя Chroma-коллекции для FAQ/доков поддержки |
| `SUPPORT_DATA_DIR` | нет | `./data/support` | папка с `users.json`/`tickets.json`/`faq.md`/`product_guide.md` |
| `CHROMA_PATH` | нет | `./.chroma` | общий Chroma store и для `element_docs`, и для `support_kb` |
| `DEEPSEEK_MODEL`, `DEEPSEEK_BASE_URL`, `EMBEDDING_MODEL`, `REQUEST_TIMEOUT` | нет | см. `.env.example` | те же, что у основного ассистента |
| `TELEGRAM_BOT_TOKEN` | для бота | — | токен от @BotFather; нужен только `support_bot`, остальные команды работают без него |

## Тесты

```bash
pytest tests/test_crm_store.py tests/test_support_context.py tests/test_support_chat_graceful.py -v
# или весь набор проекта, ничего не задето:
pytest
```
