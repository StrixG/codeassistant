"""Run the eval question set and report per-question outcomes.

For each question: run the full pipeline, then check the expected source
file appears in the answer's sources. The last question has no expected
sources — it must be answered from git tools, not RAG, so success there
means a git_* tool was called and no RAG source was cited.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from assistant.config import Config
from assistant.core import metrics
from assistant.core.llm import DeepSeekClient, LlmError
from assistant.core.mcp_client import McpClient, default_server_params
from assistant.core.pipeline import build_static_context, run_query
from assistant.core.rag import RagSearcher
from assistant.core.tools import ToolExecutor, build_registry

console = Console()

_QUESTIONS_PATH = Path(__file__).resolve().parents[1] / "eval" / "questions.yaml"


def _passed(expect_sources: list[str], result) -> bool:
    got = {s["file_path"] for s in result.sources}
    if expect_sources:
        return all(exp in got for exp in expect_sources)
    # No expected RAG source: must come from git tools, not RAG.
    used_git = any(t.startswith("git_") or t == "read_file" for t in result.tools_called)
    return used_git and not got


def run_eval() -> int:
    cfg = Config.load()
    questions = yaml.safe_load(_QUESTIONS_PATH.read_text(encoding="utf-8"))

    try:
        rag = RagSearcher(cfg)
    except Exception:
        console.print("[red]Индекс не найден. Сначала: python -m assistant index[/red]")
        return 1

    llm = DeepSeekClient(cfg)
    static_context = build_static_context(cfg.target_repo_path)
    source_sink: list[dict] = []

    table = Table(title="Eval")
    table.add_column("#", justify="right")
    table.add_column("вопрос", max_width=40)
    table.add_column("ожид. источник")
    table.add_column("нашёлся?", justify="center")
    table.add_column("latency", justify="right")
    table.add_column("tokens", justify="right")

    passed = 0
    with McpClient(default_server_params()) as mcp:
        registry = build_registry(rag, mcp, source_sink)
        # Auto-decline gated tools during eval (none should be needed).
        executor = ToolExecutor(registry, confirm=lambda _p: False)

        for i, item in enumerate(questions, 1):
            q = item["q"]
            expect = item.get("expect_sources") or []
            try:
                result = run_query(
                    q,
                    llm=llm,
                    registry=registry,
                    executor=executor,
                    source_sink=source_sink,
                    static_context=static_context,
                )
            except LlmError as e:
                table.add_row(str(i), q, ", ".join(expect) or "(git)", "[red]ERR[/red]", "-", str(e)[:20])
                continue

            ok = _passed(expect, result)
            passed += ok
            metrics.record(
                cfg.metrics_path,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "question": q,
                    "latency_ms": result.latency_ms,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "cached_tokens": result.cached_tokens,
                    "tools_called": result.tools_called,
                    "sources": [s["file_path"] for s in result.sources],
                },
            )
            table.add_row(
                str(i),
                q,
                ", ".join(expect) or "(git-тулы)",
                "[green]✓[/green]" if ok else "[red]✗[/red]",
                f"{result.latency_ms}ms",
                f"{result.prompt_tokens}+{result.completion_tokens}",
            )

    console.print(table)
    console.print(f"[bold]Passed {passed}/{len(questions)}[/bold]")
    return 0 if passed == len(questions) else 1
