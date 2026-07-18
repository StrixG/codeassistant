# Сценарий демо для видео — File Agent

Агент запускается из `/Users/obrekht/AI/codeassistant`, целевой репозиторий —
`element-android` (`TARGET_REPO_PATH=/Users/obrekht/MobileProjects/element-android`).
Держи два терминала рядом: **T1** — агент (codeassistant), **T2** —
`element-android` для показа diff/отката.

```bash
# ── T2: element-android, стартовое чистое состояние ──────────────
cd /Users/obrekht/MobileProjects/element-android
git status                       # чисто — нет незакоммиченных правок
```

## Сценарий (а) — чтение, порождает отчёт

Агент ищет использования класса, сам решает какие инструменты звать, пишет отчёт
в корень целевого репо.

```bash
# ── T1: codeassistant ────────────────────────────────────────────
cd /Users/obrekht/AI/codeassistant

python -m file_agent "Найди все места использования класса AnalyticsTracker и создай в корне usage_report.md с таблицей файл/строка/как используется и выводом об архитектуре"
```

В консоли — трасса: `search` → `read_file`(и) → `write_file`. Показать результат:

```bash
# ── T2 ──
git status                       # новый untracked файл usage_report.md
sed -n '1,40p' usage_report.md   # таблица файл/строка + вывод об архитектуре
```

## Сценарий (б) — запись, сначала предпросмотр (`--dry-run`)

Цель — `vector-config/.../Config.kt`: часть публичных флагов задокументирована
KDoc'ом, часть — нет (`ENABLE_LOCATION_SHARING`, `LOCATION_MAP_TILER_KEY`,
`LOW_PRIVACY_LOG_ENABLE`, `ENABLE_STRICT_MODE_LOGS`, `*_ANALYTICS_CONFIG`). Агент
читает файл, находит недокументированные флаги и дописывает KDoc — задача
**аддитивная**, diff гарантированно ненулевой.

`--dry-run`: `write_file`/`edit_file` не пишут на диск, копятся в оверлее, в
конце — единый unified diff. Диск нетронут.

```bash
# ── T1 ──
python -m file_agent "В файле vector-config/src/main/java/im/vector/app/config/Config.kt добавь KDoc-комментарий к каждому публичному флагу, у которого его ещё нет (по образцу соседних задокументированных); выведи список задокументированных флагов" --dry-run
```

Доказать, что диск чист после dry-run:

```bash
# ── T2 ──
git status                       # Config.kt без изменений — dry-run ничего не записал
```

## Сценарий (б) — реальный прогон (без флага)

```bash
# ── T1 ──
python -m file_agent "В файле vector-config/src/main/java/im/vector/app/config/Config.kt добавь KDoc-комментарий к каждому публичному флагу, у которого его ещё нет (по образцу соседних задокументированных); выведи список задокументированных флагов"
```

Показать реальные правки:

```bash
# ── T2: git diff в element-android ───────────────────────────────
CFG=vector-config/src/main/java/im/vector/app/config/Config.kt
git status                       # теперь Config.kt изменён
git diff --stat "$CFG"           # сводка: 1 файл, +N строк KDoc
git diff "$CFG"                  # сами KDoc-блоки над флагами
```

## Откат и повтор — доказательство воспроизводимости

Откатываем правки, состояние снова чистое, запускаем тот же прогон ещё раз.

```bash
# ── T2: откат ────────────────────────────────────────────────────
CFG=vector-config/src/main/java/im/vector/app/config/Config.kt
git checkout -- "$CFG"           # откат правок Config.kt
git status                       # снова чисто

# ── T1: повторный прогон той же цели ─────────────────────────────
cd /Users/obrekht/AI/codeassistant
python -m file_agent "В файле vector-config/src/main/java/im/vector/app/config/Config.kt добавь KDoc-комментарий к каждому публичному флагу, у которого его ещё нет (по образцу соседних задокументированных); выведи список задокументированных флагов"

# ── T2: тот же набор флагов ──────────────────────────────────────
cd /Users/obrekht/MobileProjects/element-android
git diff --stat vector-config/src/main/java/im/vector/app/config/Config.kt
```

## Заметки для съёмки

- **Почему аддитивная задача, а не «cleanup»:** element-android — зрелый,
  вылизанный линтером репо. Cleanup-цели («добавь лицензию где нет», «Log→Timber»)
  уже сделаны везде → 0 файлов, пустой diff (проверено: 0 реальных `Log.d(` во
  всём репо; заголовки стоят на всех tracked `.kt`). Аддитивная цель — «допиши
  KDoc где нет» — гарантирует ненулевой, но ограниченный diff.
- **Воспроизводимость — набор, не байты:** повтор задокументирует **тот же набор
  флагов** (`git diff --stat` идентичен), но текст KDoc генерится LLM → между
  прогонами может отличаться формулировкой. Это фича, не баг: показывает, что
  правит настоящий агент с суждением, а не sed-скрипт. Не обещай на видео
  «байт-в-байт тот же diff».
- **Точка идемпотентности:** цель — «где KDoc нет». На *не откаченном* репо
  повторный прогон тронет 0 флагов (у всех уже есть KDoc) — полезно показать как
  «агент не дублирует». Для повторного diff откатывай `git checkout -- "$CFG"`.
- **Проверка после прогона:** `git diff "$CFG"` — убедись, что KDoc вставлен НАД
  `const val`/`val`, а не разорвал объявление. Если модель ошиблась с уникальным
  `old_text`, `edit_file` отклонит правку — в трассе будет видно.
- **Семантический поиск в (а):** если `AnalyticsTracker` ищется `mode=semantic` —
  нужен индекс: `python -m assistant index`. Без индекса агент откатится на
  текстовый `git grep` — работает всегда, для видео надёжнее.
- **Убрать `usage_report.md`** перед чистой съёмкой: `rm -f usage_report.md` в
  element-android (untracked, `git checkout .` его не тронет).
- **Код возврата:** при исчерпании 15 итераций CLI вернёт `2` — на демо-целях не
  должно случаться.
