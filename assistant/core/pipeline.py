"""The single-question pipeline shared by the REPL and the eval runner.

Flow:
1. Assemble static context (current branch + top-level modules) once — it
   is stable, so it rides in the system message as a cache-friendly prefix.
2. Give the model the tools and let it decide what to call, up to 5
   tool-use iterations.
3. Return the answer plus the sources gathered by rag_search, the tools
   called, latency, and token counts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from assistant.core.llm import DeepSeekClient, usage_dict
from assistant.core.tools import ToolExecutor, ToolRegistry
from assistant.mcp_server import repo_tools

MAX_TOOL_ITERATIONS = 5

SYSTEM_PROMPT = (
    "Ты ассистент по проекту Element Android (Matrix-клиент на Kotlin). "
    "Отвечай ТОЛЬКО на основе документации, полученной через `rag_search`, "
    "и фактов из git-тулов. Всегда указывай файлы-источники. "
    "Если в документации ответа нет — так и скажи, не выдумывай."
)


@dataclass
class QueryResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    iterations: int = 0


def build_static_context(repo: Path) -> str:
    try:
        branch = repo_tools.git_current_branch(repo)
    except Exception:
        branch = "(unknown)"
    modules = sorted(
        d.name for d in repo.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    return (
        f"Текущая git-ветка целевого репозитория: {branch}\n"
        f"Модули верхнего уровня: {', '.join(modules)}"
    )


def run_query(
    question: str,
    *,
    llm: DeepSeekClient,
    registry: ToolRegistry,
    executor: ToolExecutor,
    source_sink: list[dict],
    static_context: str,
) -> QueryResult:
    # Stable prefix first (system + static context) -> DeepSeek prefix cache.
    messages: list[dict] = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{static_context}"},
        {"role": "user", "content": question},
    ]
    tools = registry.openai_tools()

    result = QueryResult(answer="")
    source_sink.clear()
    started = time.perf_counter()

    final_msg = None
    for i in range(MAX_TOOL_ITERATIONS):
        result.iterations = i + 1
        resp = llm.chat(messages, tools)
        u = usage_dict(resp)
        result.prompt_tokens += u["prompt_tokens"]
        result.completion_tokens += u["completion_tokens"]
        result.cached_tokens += u["cached_tokens"]

        msg = resp.choices[0].message
        final_msg = msg
        # Append the whole assistant message (keeps tool_calls / any
        # reasoning_content) so the next request is well-formed.
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
            output = executor.execute(name, args)
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": output}
            )

    result.answer = (final_msg.content if final_msg else "") or ""
    result.latency_ms = int((time.perf_counter() - started) * 1000)
    # Snapshot sources collected by rag_search during this query.
    result.sources = list(source_sink)
    return result
