# Element Android Docs Assistant

CLI-ассистент, отвечающий на вопросы о проекте **Element Android** (Matrix-клиент
на Kotlin). Опирается на документацию проекта через RAG и на живой контекст git
через MCP-сервер. Ответы всегда содержат ссылки на файлы-источники.

## Как это работает

```
вопрос ──▶ статический контекст (ветка + модули)
       ──▶ DeepSeek V4 Pro (function calling, thinking off)
              │  сам решает, что вызвать (до 5 итераций):
              ├─ rag_search      → ChromaDB (локальные эмбеддинги e5)
              ├─ git_* / read_file → MCP-сервер (stdio, только чтение)
       ──▶ ответ + блок «Источники»  +  метрики в metrics.jsonl
```

- **LLM:** `deepseek-v4-pro` через библиотеку `openai` (`base_url=https://api.deepseek.com`),
  thinking mode выключен (RAG не требует CoT; вдвое дешевле и быстрее).
- **Эмбеддинги:** локальные, `sentence-transformers` / `intfloat/multilingual-e5-small`
  (вопросы по-русски, документация по-английски). У DeepSeek нет `/embeddings`.
- **Векторная БД:** ChromaDB (embedded, persist на диск, без Docker).
- **MCP:** официальный Python SDK, транспорт stdio, четыре read-only git-тула.
- **CLI:** `rich` + `prompt_toolkit`. Веба нет.

## Установка

```bash
# 1. Клонировать целевой репозиторий
git clone https://github.com/element-hq/element-android.git

# 2. Виртуальное окружение и зависимости
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. Конфиг
cp .env.example .env
#   заполнить в .env:
#     DEEPSEEK_API_KEY=sk-...           (реальный ключ DeepSeek)
#     TARGET_REPO_PATH=/путь/к/element-android
```

## Использование

```bash
# Индексация документации (инкрементальная; --force для полной)
python -m assistant index
python -m assistant index --force

# Интерактивный REPL
python -m assistant chat
#   /help <вопрос>   — задать вопрос
#   /reindex         — переиндексировать
#   /metrics         — сводка latency / токенов / стоимости
#   /quit            — выход
#   (строка без слэша тоже трактуется как вопрос)

# Прогон эталонных вопросов
python -m assistant eval
```

## Что индексируется

Только документация, не код: `docs/*.md`, корневые `README.md` / `CONTRIBUTING.md`,
`*/README.md` модулей. Пропускаются `docs/images/`, PDF, `.mwb`, исходники.

Чанкинг — по markdown-заголовкам (`#`/`##`/`###`) с сохранением цепочки заголовков
(`heading_path`) в каждом чанке. Лимит чанка — 512 токенов (предел e5-small; больший
чанк молча обрезался бы при эмбеддинге). Инкрементальность — по хешу содержимого
файла (sidecar `index_state.json`).

## MCP-тулы (только чтение)

| Тул | Действие |
|---|---|
| `git_current_branch` | текущая ветка |
| `git_list_files` | список файлов (опц. префикс пути) |
| `git_diff` | `git diff HEAD` |
| `read_file` | чтение файла по пути внутри репо |

Безопасность: путь к репо берётся только из конфига (не из аргументов LLM);
все git-вызовы — `subprocess.run` со списком аргументов, без `shell=True`;
`read_file` отклоняет пути вне репозитория (`../`, абсолютные).

## Human-in-the-loop

У каждого тула есть флаг `requires_confirmation`. Если `True`, executor печатает,
что собирается сделать, и ждёт `y/n`. В боевом потоке таких тулов нет; механизм
демонстрирует фиктивный `git_push` (no-op) и покрыт тестом.

## Надёжность и стоимость

- Retry: 1 повтор при таймауте / 5xx с backoff. Timeout запроса 60 с.
- Fallback: при исчерпании retry — понятное сообщение, без стектрейса.
- Метрики (`metrics.jsonl`) на каждый запрос: latency, токены (вкл. `cached_tokens`),
  вызванные тулы, источники. `/metrics` считает P50/P95, средние токены и стоимость.
- Кэш префикса: системный промпт и статический контекст идут неизменным префиксом
  сообщений — DeepSeek кэширует его автоматически (cache-hit вход в ~120× дешевле).

Тарифы `deepseek-v4-pro` (за 1M токенов): cache-hit вход $0.003625, cache-miss вход
$0.435, выход $0.87.

## Тесты

```bash
.venv/bin/python -m pytest -q
```

Покрыто: чанкер (заголовки в `heading_path`, лимит чанка), индексатор (нет дублей
при повторе), MCP `read_file` (traversal отклоняется), executor (gated-тул не
выполняется без подтверждения), метрики (формула стоимости, перцентили).

## Структура

```
assistant/
  __main__.py          # CLI: index | chat | eval
  config.py            # чтение .env
  eval_runner.py       # прогон eval/questions.yaml
  core/
    llm.py             # DeepSeek клиент, retry, токены
    tools.py           # Tool Registry + Executor + confirmation hook
    rag.py             # поиск по Chroma
    mcp_client.py      # синхронная обёртка над MCP stdio-сессией
    pipeline.py        # цикл tool-use одного запроса
    embeddings.py      # локальные эмбеддинги e5 (query:/passage:)
    metrics.py         # запись и агрегация JSONL, стоимость
  indexer/
    chunker.py         # разбиение markdown по заголовкам
    index.py           # обход файлов, хеши, запись в Chroma
  mcp_server/
    server.py          # MCP stdio-сервер (4 git-тула)
    repo_tools.py      # git/файловые хелперы + защита от traversal
  cli/
    chat.py            # REPL, команды, вывод источников
eval/questions.yaml    # 10 эталонных вопросов
tests/
```
