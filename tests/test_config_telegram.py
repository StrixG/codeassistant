"""Config must surface the Telegram bot token without making it mandatory.

``Config.load()`` is shared with the main assistant and with
``mcp_crm.server``; requiring the bot token there would break both.
"""

from __future__ import annotations

from assistant.config import Config


def test_telegram_token_read_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TARGET_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")

    cfg = Config.load()

    assert cfg.telegram_bot_token == "123:ABC"


def test_telegram_token_defaults_to_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("TARGET_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    cfg = Config.load()

    assert cfg.telegram_bot_token == ""
