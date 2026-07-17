"""Telegram entry point for the support assistant.

Run:  python -m support_bot.bot

One McpClient (one mcp_crm.server subprocess) serves every chat: the CRM
tools are stateless JSON reads and writes, so a single process keeps up,
and handlers push their calls to worker threads anyway. The client is
started before polling and stopped after it, so the subprocess never
outlives the bot.

The RAG index and the DeepSeek client are likewise built once and shared
— loading the embedding model per message would cost seconds each time.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace as replace_cfg

from aiogram import Bot, Dispatcher

from assistant.config import Config
from assistant.core.llm import DeepSeekClient
from assistant.core.mcp_client import McpClient
from assistant.core.rag import RagSearcher
from mcp_crm.server import default_server_params as crm_server_params
from support_bot.binding import PendingBindings
from support_bot.handlers import router

log = logging.getLogger(__name__)


def _build_rag(cfg: Config) -> RagSearcher:
    """Point the shared searcher at the support collection, as the CLI does."""
    support_cfg = replace_cfg(cfg, chroma_collection=cfg.support_chroma_collection)
    rag = RagSearcher(support_cfg)
    if rag.count() == 0:
        raise RuntimeError(
            "Индекс support_kb пуст. Сначала: python -m support_assistant.index_support_kb"
        )
    return rag


async def run_bot() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = Config.load()
    if not cfg.telegram_bot_token:
        log.error("TELEGRAM_BOT_TOKEN не задан. Добавьте его в .env (см. .env.example).")
        return 1

    try:
        rag = _build_rag(cfg)
    except Exception as e:
        log.error("%s", e)
        return 1

    llm = DeepSeekClient(cfg)
    pending = PendingBindings()

    bot = Bot(token=cfg.telegram_bot_token)
    dp = Dispatcher()
    dp.include_router(router)

    mcp = McpClient(crm_server_params()).start()
    # Names here must match the handler parameter names in handlers.py.
    dp["mcp"] = mcp
    dp["rag"] = rag
    dp["llm"] = llm
    dp["pending"] = pending

    log.info("Бот запущен, MCP-тулы: %s", ", ".join(sorted(t.name for t in mcp.tools)))
    try:
        await dp.start_polling(bot)
    finally:
        mcp.stop()
        await bot.session.close()
        log.info("Бот остановлен.")
    return 0


def main() -> int:
    try:
        return asyncio.run(run_bot())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
