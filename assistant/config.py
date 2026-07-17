"""Central config, loaded from .env.

All secrets and paths live here. Nothing else reads os.environ directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str | None = None, *, required: bool = False) -> str:
    val = os.environ.get(key, default)
    if required and not val:
        raise RuntimeError(
            f"Missing required env var {key}. Copy .env.example to .env and fill it in."
        )
    return val or ""


@dataclass(frozen=True)
class Config:
    # DeepSeek
    deepseek_api_key: str
    deepseek_model: str
    deepseek_base_url: str
    request_timeout: int

    # Embeddings / vector store
    embedding_model: str
    chroma_path: Path
    chroma_collection: str

    # Target repo
    target_repo_path: Path
    target_default_branch: str

    # Ops
    metrics_path: Path

    # Support assistant (Day 33): separate RAG collection + mock-CRM JSON dir.
    # Defaulted so existing Config(...) call sites (tests, day-31/32 code)
    # keep working unchanged.
    support_chroma_collection: str = "support_kb"
    support_data_dir: Path = Path("data/support")

    @classmethod
    def load(cls, *, require_api_key: bool = True) -> "Config":
        repo = Path(_get("TARGET_REPO_PATH", required=True)).expanduser().resolve()
        return cls(
            deepseek_api_key=_get("DEEPSEEK_API_KEY", required=require_api_key),
            deepseek_model=_get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
            deepseek_base_url=_get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            request_timeout=int(_get("REQUEST_TIMEOUT", "60")),
            embedding_model=_get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
            chroma_path=Path(_get("CHROMA_PATH", "./.chroma")).expanduser().resolve(),
            chroma_collection=_get("CHROMA_COLLECTION", "element_docs"),
            target_repo_path=repo,
            target_default_branch=_get("TARGET_DEFAULT_BRANCH", "develop"),
            metrics_path=Path(_get("METRICS_PATH", "./metrics.jsonl")).expanduser().resolve(),
            support_chroma_collection=_get("SUPPORT_CHROMA_COLLECTION", "support_kb"),
            support_data_dir=Path(_get("SUPPORT_DATA_DIR", "./data/support")).expanduser().resolve(),
        )
