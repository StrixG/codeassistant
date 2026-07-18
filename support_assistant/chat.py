"""Support-assistant CLI for Element Android.

Run:  python -m support_assistant.chat --user user-1

Per message the flow is fixed (no agentic tool-calling loop — a single
DeepSeek call per turn, per spec):

1. Fetch the user's CRM profile and open tickets *through MCP*
   (``mcp_crm.server``, via the shared ``assistant.core.mcp_client``) —
   never by reading the JSON files directly.
2. RAG top-k search over the ``support_kb`` Chroma collection (FAQ +
   product guide), reusing ``assistant.core.rag.RagSearcher``.
3. One DeepSeek call (``assistant.core.llm.DeepSeekClient``) with the
   profile, tickets and retrieved chunks folded into the prompt.
4. If the model judges the open-ticket problem solved, it appends a
   ``SUGGEST_CLOSE: <ticket_id>`` marker; on user confirmation the CLI
   calls ``update_ticket`` through MCP.

After every answer a debug block prints which MCP tools were called and
which RAG chunks made it into context — useful for demoing the pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from dataclasses import replace as replace_cfg

from prompt_toolkit import PromptSession
from rich.console import Console
from rich.panel import Panel

from assistant.config import Config
from assistant.core.llm import DeepSeekClient, LlmError
from assistant.core.mcp_client import McpClient
from assistant.core.rag import RagSearcher, SearchHit
from mcp_crm.server import default_server_params as crm_server_params

console = Console()

SYSTEM_PROMPT = (
    "Ты дружелюбный ассистент поддержки мессенджера Element для Android. "
    "Отвечай на русском языке, коротко и по делу, не выдумывай факты сверх "
    "того, что дано в профиле пользователя, его тикетах и фрагментах базы "
    "знаний ниже. Если причина проблемы видна из профиля пользователя "
    "(например, устаревшая версия приложения) — называй именно её, а не "
    "общие догадки. Если у пользователя есть релевантный открытый тикет, "
    "обязательно сошлись на его номер (например, «тикет ticket-1001»). "
    "Если твой ответ решает проблему, из-за которой открыт тикет, заверши "
    "сообщение ОТДЕЛЬНОЙ последней строкой вида "
    "'SUGGEST_CLOSE: <id тикета>' — эта строка не показывается пользователю "
    "напрямую, она только предлагает закрыть тикет."
)

_CLOSE_MARKER_RE = re.compile(r"\n?SUGGEST_CLOSE:\s*(\S+)\s*$")


@dataclass
class SupportTurnResult:
    answer: str
    ticket_suggested: str | None = None
    rag_hits: list[SearchHit] = field(default_factory=list)
    llm_ok: bool = True


def extract_close_suggestion(answer: str) -> tuple[str, str | None]:
    """Split a trailing ``SUGGEST_CLOSE: <id>`` marker off the model's answer."""
    m = _CLOSE_MARKER_RE.search(answer)
    if not m:
        return answer.strip(), None
    ticket_id = m.group(1).strip()
    cleaned = _CLOSE_MARKER_RE.sub("", answer).strip()
    return cleaned, ticket_id


def build_context_block(user: dict, tickets: list[dict], hits: list[SearchHit]) -> str:
    """Render the CRM profile, open tickets and RAG hits as prompt context."""
    lines = [
        "Профиль пользователя:",
        f"- Имя: {user.get('name')}",
        f"- Email: {user.get('email')}",
        f"- Платформа: {user.get('platform')}, версия приложения: {user.get('app_version')}",
        f"- Тариф: {user.get('plan')}, дата регистрации: {user.get('signup_date')}",
        "",
    ]

    if tickets:
        lines.append("Открытые тикеты пользователя:")
        for t in tickets:
            lines.append(f"- [{t['id']}] (приоритет: {t.get('priority')}) {t.get('subject')}")
            lines.append(f"  Описание: {t.get('description')}")
            history = t.get("history") or []
            if history:
                last = history[-1]
                lines.append(
                    f"  Последнее сообщение ({last.get('timestamp')}, "
                    f"{last.get('author')}): {last.get('text')}"
                )
    else:
        lines.append("Открытых тикетов у пользователя нет.")
    lines.append("")

    if hits:
        lines.append("Фрагменты базы знаний:")
        for h in hits:
            tag = h.file_path + (f" :: {h.heading_path}" if h.heading_path else "")
            lines.append(f"[{tag}]\n{h.text}")
    else:
        lines.append("Релевантных фрагментов базы знаний не найдено.")

    return "\n".join(lines)


def answer_message(
    question: str,
    *,
    user: dict,
    tickets: list[dict],
    hits: list[SearchHit],
    llm: DeepSeekClient,
) -> SupportTurnResult:
    """Build the context, make the one DeepSeek call, and parse the reply.

    Degrades gracefully: if DeepSeek is unavailable (``LlmError``), returns
    a friendly Russian apology instead of raising — the caller never sees
    an exception from a down LLM.
    """
    context = build_context_block(user, tickets, hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\nВопрос пользователя: {question}"},
    ]
    try:
        resp = llm.chat(messages)
    except LlmError as e:
        return SupportTurnResult(
            answer=(
                "Извините, сервис ИИ-ассистента сейчас недоступен. "
                f"Попробуйте, пожалуйста, ещё раз через пару минут. ({e})"
            ),
            llm_ok=False,
        )

    raw = (resp.choices[0].message.content or "").strip()
    cleaned, ticket_id = extract_close_suggestion(raw)
    # Only honour the suggestion if it names a ticket the user actually has
    # open — the model isn't trusted to invent or misquote an id.
    open_ids = {t["id"] for t in tickets}
    if ticket_id not in open_ids:
        ticket_id = None
    return SupportTurnResult(answer=cleaned, ticket_suggested=ticket_id, rag_hits=hits)


def _mcp_json(mcp: McpClient, tool: str, arguments: dict, calls: list[str]) -> dict | list:
    calls.append(tool)
    raw = mcp.call_tool(tool, arguments)
    if raw.startswith("Error:"):
        raise RuntimeError(raw)
    return json.loads(raw)


def _print_debug(mcp_calls: list[str], hits: list[SearchHit]) -> None:
    console.print(f"[dim]MCP-тулзы вызваны: {', '.join(mcp_calls) or '(нет)'}[/dim]")
    if hits:
        chunks = "; ".join(
            h.file_path + (f" :: {h.heading_path}" if h.heading_path else "") for h in hits
        )
        console.print(f"[dim]RAG-чанки в контексте: {chunks}[/dim]")
    else:
        console.print("[dim]RAG-чанки в контексте: (нет)[/dim]")


def run_chat(user_id: str) -> int:
    cfg = Config.load()
    support_cfg = replace_cfg(cfg, chroma_collection=cfg.support_chroma_collection)
    try:
        rag = RagSearcher(support_cfg)
    except Exception:
        console.print(
            "[red]Индекс support_kb не найден. Сначала: "
            "python -m support_assistant.index_support_kb[/red]"
        )
        return 1
    if rag.count() == 0:
        console.print(
            "[red]Индекс support_kb пуст. Сначала: "
            "python -m support_assistant.index_support_kb[/red]"
        )
        return 1

    llm = DeepSeekClient(cfg)
    console.print(
        Panel(
            f"[bold]Ассистент поддержки Element[/bold]\nПользователь: {user_id}\n"
            "Команды: [cyan]/quit[/cyan] для выхода",
            border_style="blue",
        )
    )
    session: PromptSession = PromptSession()

    with McpClient(crm_server_params()) as mcp:
        while True:
            try:
                question = session.prompt("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                continue
            if question in ("/quit", "/exit"):
                break

            mcp_calls: list[str] = []
            try:
                user = _mcp_json(mcp, "get_user", {"user_id": user_id}, mcp_calls)
                tickets = _mcp_json(
                    mcp, "list_tickets", {"user_id": user_id, "status": "open"}, mcp_calls
                )
            except RuntimeError as e:
                console.print(f"[red]CRM недоступна через MCP: {e}[/red]")
                continue

            hits = rag.search(question, top_k=4)

            result = answer_message(question, user=user, tickets=tickets, hits=hits, llm=llm)

            style = "green" if result.llm_ok else "red"
            console.print(Panel(result.answer, title="Ответ", border_style=style))
            _print_debug(mcp_calls, hits)

            if result.ticket_suggested:
                console.print(
                    f"[yellow]Ассистент считает, что тикет "
                    f"{result.ticket_suggested} можно закрыть.[/yellow]"
                )
                ans = input(f"Закрыть тикет {result.ticket_suggested}? [y/N] ").strip().lower()
                if ans in ("y", "yes"):
                    _mcp_json(
                        mcp,
                        "update_ticket",
                        {
                            "ticket_id": result.ticket_suggested,
                            "status": "resolved",
                            "note": "Закрыто ассистентом поддержки после подтверждения пользователем.",
                        },
                        mcp_calls,
                    )
                    console.print("[dim]MCP-тулзы вызваны: update_ticket[/dim]")
                    console.print(f"[green]Тикет {result.ticket_suggested} закрыт.[/green]")

    console.print("[dim]Пока.[/dim]")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="support_assistant.chat", description="Element support assistant")
    parser.add_argument("--user", required=True, help="CRM user id, e.g. user-1")
    args = parser.parse_args()
    return run_chat(args.user)


if __name__ == "__main__":
    raise SystemExit(main())
