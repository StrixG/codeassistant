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

`--dry-run`: `write_file`/`edit_file` не пишут на диск, копятся в оверлее, в
конце — единый unified diff. Диск нетронут.

```bash
# ── T1 ──
python -m file_agent "Добавь лицензионный заголовок во все .kt-файлы vector-config, где его нет; выведи список исправленных" --dry-run
```

Доказать, что диск чист после dry-run:

```bash
# ── T2 ──
git status                       # vector-config без изменений — dry-run ничего не записал
```

## Сценарий (б) — реальный прогон (без флага)

```bash
# ── T1 ──
python -m file_agent "Добавь лицензионный заголовок во все .kt-файлы vector-config, где его нет; выведи список исправленных"
```

Показать реальные правки:

```bash
# ── T2: git diff в element-android ───────────────────────────────
git status                       # теперь .kt-файлы vector-config изменены
git diff --stat vector-config    # сводка: сколько файлов тронуто
git diff vector-config | head -60  # сами заголовки в начале файлов
```

## Откат и повтор — доказательство воспроизводимости

Откатываем правки, состояние снова чистое, запускаем тот же прогон ещё раз —
результат тот же diff.

```bash
# ── T2: откат ────────────────────────────────────────────────────
git checkout .                   # откат правок vector-config
git clean -fd vector-config      # (если агент создавал новые файлы)
git status                       # снова чисто

# ── T1: повторный прогон той же цели ─────────────────────────────
cd /Users/obrekht/AI/codeassistant
python -m file_agent "Добавь лицензионный заголовок во все .kt-файлы vector-config, где его нет; выведи список исправленных"

# ── T2: тот же результат ─────────────────────────────────────────
cd /Users/obrekht/MobileProjects/element-android
git diff --stat vector-config    # тот же набор файлов, что и в первый раз
```

## Заметки для съёмки

- **Точка идемпотентности:** цель — «где заголовка нет». После первого прогона
  повторный на *не откаченном* репо тронет 0 файлов (`edit_file` не найдёт куда
  вставлять) — тоже полезно показать как «агент не дублирует». Но для «того же
  diff» откатывай через `git checkout .` перед повтором.
- **Семантический поиск в (а):** если `AnalyticsTracker` ищется `mode=semantic` —
  нужен индекс: `python -m assistant index`. Без индекса агент откатится на
  текстовый `git grep` — работает всегда, для видео надёжнее.
- **Убрать `usage_report.md`** перед чистой съёмкой: `rm -f usage_report.md` в
  element-android (untracked, `git checkout .` его не тронет).
- **Код возврата:** при исчерпании 15 итераций CLI вернёт `2` — на демо-целях не
  должно случаться.
