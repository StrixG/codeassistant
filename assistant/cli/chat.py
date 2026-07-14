"""Interactive REPL: /help, /reindex, /metrics, /quit.

Wires the whole assistant together: RAG searcher, MCP git tools, tool
registry + executor (with a real y/n confirmation prompt), DeepSeek client,
and metrics recording. Every /help answer prints a Sources block.
"""

from __future__ import annotations

from datetime import datetime

from prompt_toolkit import PromptSession
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from assistant.config import Config
from assistant.core import metrics
from assistant.core.llm import DeepSeekClient, LlmError
from assistant.core.mcp_client import McpClient, default_server_params
from assistant.core.pipeline import build_static_context, run_query
from assistant.core.rag import RagSearcher
from assistant.core.tools import ToolExecutor, build_registry

console = Console()

_BANNER = (
    "[bold]Element Android docs assistant[/bold]\n"
    "Commands: [cyan]/help <вопрос>[/cyan]  [cyan]/reindex[/cyan]  "
    "[cyan]/metrics[/cyan]  [cyan]/quit[/cyan]\n"
    "(a bare line without a slash is treated as a question)"
)


def _make_confirm():
    def confirm(prompt: str) -> bool:
        console.print(f"[yellow]{prompt}[/yellow]")
        ans = input("Proceed? [y/N] ").strip().lower()
        return ans in ("y", "yes")

    return confirm


def _print_result(result) -> None:
    console.print(Panel(result.answer or "(пустой ответ)", title="Ответ", border_style="green"))
    if result.sources:
        console.print("[bold]Источники:[/bold]")
        for s in result.sources:
            hp = s.get("heading_path") or "(файл целиком)"
            console.print(f"  • {s['file_path']} → {hp}")
    elif "rag_search" in result.tools_called:
        console.print("[red]Источники: (rag_search вызван, но источников нет — баг)[/red]")
    else:
        console.print("[dim]Источники: (RAG не использовался — ответ из git-тулов)[/dim]")
    console.print(
        f"[dim]tools={result.tools_called} latency={result.latency_ms}ms "
        f"prompt={result.prompt_tokens} completion={result.completion_tokens} "
        f"cached={result.cached_tokens}[/dim]"
    )


def _record(cfg: Config, question: str, result) -> None:
    metrics.record(
        cfg.metrics_path,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "latency_ms": result.latency_ms,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "cached_tokens": result.cached_tokens,
            "tools_called": result.tools_called,
            "sources": [s["file_path"] for s in result.sources],
        },
    )


def _print_metrics(cfg: Config) -> None:
    s = metrics.summarize(cfg.metrics_path)
    if s.get("count", 0) == 0:
        console.print("[dim]Нет записанных метрик.[/dim]")
        return
    t = Table(title="Метрики")
    t.add_column("metric")
    t.add_column("value", justify="right")
    t.add_row("requests", str(s["count"]))
    t.add_row("latency P50", f"{s['latency_p50_ms']} ms")
    t.add_row("latency P95", f"{s['latency_p95_ms']} ms")
    t.add_row("avg prompt tokens", str(s["avg_prompt_tokens"]))
    t.add_row("avg completion tokens", str(s["avg_completion_tokens"]))
    t.add_row("avg cached tokens", str(s["avg_cached_tokens"]))
    t.add_row("avg cost", f"${s['avg_cost_usd']:.6f}")
    t.add_row("total cost", f"${s['total_cost_usd']:.6f}")
    console.print(t)


def run_chat() -> int:
    cfg = Config.load()
    try:
        rag = RagSearcher(cfg)
    except Exception:
        console.print("[red]Индекс не найден. Сначала: python -m assistant index[/red]")
        return 1
    if rag.count() == 0:
        console.print("[red]Индекс пуст. Сначала: python -m assistant index[/red]")
        return 1

    llm = DeepSeekClient(cfg)
    static_context = build_static_context(cfg.target_repo_path)
    source_sink: list[dict] = []

    console.print(Panel(_BANNER, border_style="blue"))
    session: PromptSession = PromptSession()

    with McpClient(default_server_params()) as mcp:
        registry = build_registry(rag, mcp, source_sink)
        executor = ToolExecutor(registry, _make_confirm())

        while True:
            try:
                line = session.prompt("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue

            if line in ("/quit", "/exit"):
                break
            if line == "/metrics":
                _print_metrics(cfg)
                continue
            if line == "/reindex":
                from assistant.indexer.index import run_index

                run_index(force=False)
                rag = RagSearcher(cfg)  # fresh handle sees new data
                registry = build_registry(rag, mcp, source_sink)
                executor = ToolExecutor(registry, _make_confirm())
                continue

            if line.startswith("/help"):
                question = line[len("/help"):].strip()
            elif line.startswith("/"):
                console.print(f"[red]Неизвестная команда: {line.split()[0]}[/red]")
                continue
            else:
                question = line  # bare line = question

            if not question:
                console.print("[dim]Использование: /help <вопрос>[/dim]")
                continue

            try:
                result = run_query(
                    question,
                    llm=llm,
                    registry=registry,
                    executor=executor,
                    source_sink=source_sink,
                    static_context=static_context,
                )
            except LlmError as e:
                console.print(f"[red]Сервис недоступен: {e}. Попробуйте позже.[/red]")
                continue

            _print_result(result)
            _record(cfg, question, result)

    console.print("[dim]Пока.[/dim]")
    return 0
