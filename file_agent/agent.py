"""The goal-driven agent loop.

The user states a goal ("update the docs", "add the licence header where it's
missing"), not a sequence of steps. The agent decides which file tools to call
and in what order, executes them, feeds the results back, and repeats until it
answers without requesting another tool — capped at :data:`MAX_ITERATIONS`.

Reuses :class:`DeepSeekClient` (function-calling mode), the Tool/Registry/
Executor plumbing and the Chroma code index. This module only adds the loop,
the system prompt and the per-step console trace used for the demo video.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from assistant.config import Config
from assistant.core.llm import DeepSeekClient, usage_dict
from assistant.core.tools import ToolExecutor
from file_agent.tools import FileTools, build_registry

MAX_ITERATIONS = 15

SYSTEM_PROMPT = (
    "Ты ассистент по файлам проекта. Тебе ставят ЦЕЛЬ, а не пошаговые команды. "
    "Сам решай, какие инструменты и в каком порядке вызывать, чтобы её достичь: "
    "ищи (`search`), перечисляй (`list_files`), читай (`read_file`), создавай "
    "(`write_file`) и правь (`edit_file`) файлы. "
    "Работай маленькими проверяемыми шагами: сначала найди и прочитай нужное, "
    "потом меняй. Для правки существующих файлов предпочитай `edit_file` — "
    "`old_text` должен встречаться в файле ровно один раз, иначе добавь контекста. "
    "Не выдумывай пути — проверяй их через `list_files`/`search`. "
    "Когда цель достигнута, ответь БЕЗ вызова инструментов и дай краткий итог: "
    "что сделано и какие файлы затронуты."
)


class _LazyRag:
    """Defers loading the embedding model until semantic search is first used."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._searcher = None

    def search(self, query: str, top_k: int = 5, *, where: dict | None = None):
        if self._searcher is None:
            from assistant.core.rag import RagSearcher

            self._searcher = RagSearcher(self._cfg)
        return self._searcher.search(query, top_k=top_k, where=where)


@dataclass
class AgentResult:
    answer: str
    tools_called: list[str] = field(default_factory=list)
    iterations: int = 0
    hit_limit: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    diff: str = ""  # unified diff of staged changes when --dry-run


def _summarize(output: str) -> str:
    """One-line preview of a tool result for the console trace."""
    first = output.strip().splitlines()[0] if output.strip() else "(empty)"
    if len(first) > 100:
        first = first[:100] + "…"
    return f"{first}  [{len(output)} chars]"


def _fmt_args(args: dict) -> str:
    text = json.dumps(args, ensure_ascii=False)
    return text if len(text) <= 160 else text[:160] + "…"


def run_agent(
    goal: str,
    *,
    cfg: Config,
    dry_run: bool = False,
    llm: DeepSeekClient | None = None,
    tools: FileTools | None = None,
    console: Console | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> AgentResult:
    """Drive the agent toward ``goal``; return the final answer and a trace.

    ``llm`` and ``tools`` are injectable so tests can mock the model and point
    the tools at a scratch directory.
    """
    console = console or Console()
    llm = llm or DeepSeekClient(cfg)
    tools = tools or FileTools(cfg.target_repo_path, searcher=_LazyRag(cfg), dry_run=dry_run)

    registry = build_registry(tools)
    executor = ToolExecutor(registry, confirm=lambda _prompt: True)
    schemas = registry.openai_tools()

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Цель: {goal}"},
    ]
    result = AgentResult(answer="")
    mode = "[yellow]DRY-RUN[/yellow]" if dry_run else "[green]LIVE[/green]"
    console.print(Panel(f"{mode}  цель: {goal}", title="file-agent", border_style="blue"))

    final_msg = None
    for i in range(max_iterations):
        result.iterations = i + 1
        console.print(f"[dim]── итерация {i + 1}/{max_iterations} ──[/dim]")

        resp = llm.chat(messages, schemas)
        u = usage_dict(resp)
        result.prompt_tokens += u["prompt_tokens"]
        result.completion_tokens += u["completion_tokens"]
        result.cached_tokens += u["cached_tokens"]

        msg = resp.choices[0].message
        final_msg = msg
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            break

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result.tools_called.append(name)
            console.print(f"  [cyan]→ {name}[/cyan] {_fmt_args(args)}")
            output = executor.execute(name, args)
            console.print(f"    [dim]↳ {_summarize(output)}[/dim]")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
    else:
        # Loop exhausted without a tool-free answer.
        result.hit_limit = True
        console.print(f"[red]Достигнут лимит итераций ({max_iterations}) — остановка.[/red]")

    result.answer = (final_msg.content if final_msg else "") or ""

    if dry_run and tools.has_pending():
        result.diff = tools.pending_diff()
        console.print(Panel("Предпросмотр изменений (--dry-run, на диск не записано):",
                            border_style="yellow"))
        console.print(Syntax(result.diff, "diff", theme="ansi_dark", word_wrap=True))
    elif dry_run:
        console.print("[dim]--dry-run: изменений не накоплено.[/dim]")

    console.print(Panel(result.answer or "(пустой ответ)", title="Итог", border_style="green"))
    console.print(
        f"[dim]tools={result.tools_called} iterations={result.iterations} "
        f"prompt={result.prompt_tokens} completion={result.completion_tokens} "
        f"cached={result.cached_tokens}[/dim]"
    )
    return result


def build_and_run(goal: str, *, dry_run: bool = False) -> AgentResult:
    """Load config from the environment and run the agent against the target repo."""
    cfg = Config.load()
    if not Path(cfg.target_repo_path).is_dir():
        raise SystemExit(f"TARGET_REPO_PATH is not a directory: {cfg.target_repo_path}")
    return run_agent(goal, cfg=cfg, dry_run=dry_run)
