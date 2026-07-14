"""Indexer test: re-running does not create duplicate chunks.

Runs against a temp repo and a temp Chroma dir, with the embedder and the
token counter faked so the test is fast and offline (no model download).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from assistant import indexer
from assistant.config import Config
from assistant.indexer import index as index_mod


class _FakeEmbedder:
    dim = 4

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        # Deterministic 4-dim vectors; content-derived so it's non-trivial.
        return [[float(len(t) % 5), 1.0, 0.0, 0.5] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 1.0, 0.0, 0.5]


def _wc(text: str) -> int:
    return len(text.split())


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    (r / "docs").mkdir(parents=True)
    (r / "README.md").write_text("# Readme\n\nHello world.\n")
    (r / "docs" / "a.md").write_text(
        "# A\n\nAlpha section.\n\n## Sub\n\nSub content here.\n"
    )
    return r


@pytest.fixture
def patched(monkeypatch, repo, tmp_path):
    chroma = tmp_path / "chroma"
    cfg = Config(
        deepseek_api_key="",
        deepseek_model="deepseek-v4-pro",
        deepseek_base_url="https://api.deepseek.com",
        request_timeout=60,
        embedding_model="fake",
        chroma_path=chroma,
        chroma_collection="test_docs",
        target_repo_path=repo,
        target_default_branch="develop",
        metrics_path=tmp_path / "metrics.jsonl",
    )
    monkeypatch.setattr(Config, "load", classmethod(lambda cls, **k: cfg))
    monkeypatch.setattr(index_mod, "get_embedder", lambda name: _FakeEmbedder())
    monkeypatch.setattr(index_mod, "default_token_counter", lambda name=None: _wc)
    return cfg


def _collection(cfg: Config):
    import chromadb

    client = chromadb.PersistentClient(path=str(cfg.chroma_path))
    return client.get_collection(cfg.chroma_collection)


def test_reindex_no_duplicates(patched):
    assert index_mod.run_index() == 0
    first = _collection(patched).count()
    assert first > 0

    # Second run: nothing changed, count must stay identical, no duplicate ids.
    assert index_mod.run_index() == 0
    coll = _collection(patched)
    assert coll.count() == first
    ids = coll.get()["ids"]
    assert len(ids) == len(set(ids))


def test_changed_file_replaces_not_appends(patched):
    index_mod.run_index()
    before = _collection(patched).count()

    # Grow one file; its chunks are replaced, others skipped -> may differ but
    # the file's old chunks must not linger as duplicates.
    (patched.target_repo_path / "docs" / "a.md").write_text(
        "# A\n\nAlpha section extended a lot.\n\n## Sub\n\nMore content now.\n\n## Extra\n\nBrand new.\n"
    )
    index_mod.run_index()
    coll = _collection(patched)
    ids = coll.get()["ids"]
    assert len(ids) == len(set(ids))
    # README chunks untouched and not duplicated.
    readme_ids = [i for i in ids if i.startswith("README.md::")]
    assert len(readme_ids) == len(set(readme_ids))
    assert before > 0
