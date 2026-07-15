"""DeepSeek chat client: function calling, retry, token accounting.

Model is ``deepseek-v4-pro`` (never the deprecated ``deepseek-chat`` alias).
Thinking mode is disabled: RAG answers don't need a chain of thought, and
CoT doubles latency and output cost. With thinking off, temperature is
honoured, so we pin it low for factual, repeatable answers.

The system prompt and static context are sent as the stable message prefix
so DeepSeek's automatic prefix caching kicks in — cache-hit input is ~120x
cheaper than cache-miss input.
"""

from __future__ import annotations

import time

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
)

from assistant.config import Config

# Disable V4 thinking mode for this workload.
_THINKING_OFF = {"thinking": {"type": "disabled"}}
_RETRY_BACKOFF_S = 1.5


class LlmError(Exception):
    """Raised after retries are exhausted; carries a user-safe message."""


class DeepSeekClient:
    def __init__(self, cfg: Config) -> None:
        self._client = OpenAI(
            api_key=cfg.deepseek_api_key,
            base_url=cfg.deepseek_base_url,
            timeout=cfg.request_timeout,
        )
        self._model = cfg.deepseek_model

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        response_format: dict | None = None,
        **kwargs,
    ):
        """One chat completion with 1 retry on timeout / 5xx, with backoff.

        ``response_format={"type": "json_object"}`` switches on JSON mode; the
        prompt must mention JSON for the API to accept it. Omitted from the
        request entirely when None, so existing callers send identical bodies.
        """
        params = {
            "model": self._model,
            "messages": messages,
            "tools": tools or None,
            "tool_choice": "auto" if tools else None,
            "temperature": 0.0,
            "extra_body": _THINKING_OFF,
            **kwargs,
        }
        if response_format is not None:
            params["response_format"] = response_format

        last_err: Exception | None = None
        for attempt in range(2):  # 1 initial try + 1 retry
            try:
                return self._client.chat.completions.create(**params)
            except (APITimeoutError, APIConnectionError, InternalServerError) as e:
                last_err = e
            except APIStatusError as e:
                if e.status_code and e.status_code >= 500:
                    last_err = e
                else:
                    # 4xx (bad request, auth, ...) won't fix on retry.
                    raise LlmError(f"DeepSeek request failed: {e.status_code}") from e
            if attempt == 0:
                time.sleep(_RETRY_BACKOFF_S)
        raise LlmError(f"DeepSeek unavailable after retry: {last_err}")


def usage_dict(response) -> dict[str, int]:
    """Extract prompt/completion/cached token counts from a response.

    DeepSeek returns ``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``
    on the usage object; ``cached`` is the hit count.
    """
    u = getattr(response, "usage", None)
    if u is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    extra = getattr(u, "model_extra", None) or {}
    cached = (
        getattr(u, "prompt_cache_hit_tokens", None)
        if getattr(u, "prompt_cache_hit_tokens", None) is not None
        else extra.get("prompt_cache_hit_tokens", 0)
    )
    return {
        "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
        "cached_tokens": int(cached or 0),
    }
