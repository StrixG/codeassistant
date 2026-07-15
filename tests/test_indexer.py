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


class _CountingEmbedder(_FakeEmbedder):
    """Counts how many passages actually get embedded."""

    def __init__(self) -> None:
        self.embedded = 0

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.embedded += len(texts)
        return super().embed_passages(texts)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    (r / "docs").mkdir(parents=True)
    (r / "README.md").write_text("# Readme\n\nHello world.\n")
    (r / "docs" / "a.md").write_text(
        "# A\n\nAlpha section.\n\n## Sub\n\nSub content here.\n"
    )
    main = r / "vector" / "src" / "main" / "java" / "im" / "vector" / "app"
    main.mkdir(parents=True)
    (main / "Foo.kt").write_text(
        "package im.vector.app\n\nclass Foo {\n    fun bar() = 1\n}\n"
    )
    tests = r / "vector" / "src" / "test" / "java" / "im" / "vector" / "app"
    tests.mkdir(parents=True)
    (tests / "FooTest.kt").write_text(
        "package im.vector.app\n\nclass FooTest {\n    fun testBar() = 1\n}\n"
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


def test_code_and_docs_carry_source_metadata(patched):
    index_mod.run_index()
    metas = _collection(patched).get()["metadatas"]
    by_source: dict[str, set[str]] = {}
    for m in metas:
        by_source.setdefault(m["source"], set()).add(m["file_path"])

    assert by_source["code"] == {"vector/src/main/java/im/vector/app/Foo.kt"}
    assert "README.md" in by_source["docs"]
    assert "docs/a.md" in by_source["docs"]


def test_test_sources_are_not_indexed(patched):
    repo = patched.target_repo_path
    assert index_mod.discover_code_files(repo) == [
        "vector/src/main/java/im/vector/app/Foo.kt"
    ]

    index_mod.run_index()
    paths = {m["file_path"] for m in _collection(patched).get()["metadatas"]}
    assert not any("src/test/" in p for p in paths)


def test_include_code_false_indexes_docs_only(patched):
    index_mod.run_index(include_code=False)
    sources = {m["source"] for m in _collection(patched).get()["metadatas"]}
    assert sources == {"docs"}


def test_run_index_accepts_cfg_without_patching_load(patched, monkeypatch):
    # Config.load must not be needed when a cfg is handed in directly.
    monkeypatch.setattr(
        Config, "load", classmethod(lambda cls, **k: pytest.fail("Config.load called"))
    )
    assert index_mod.run_index(cfg=patched) == 0
    assert _collection(patched).count() > 0


def test_unchanged_files_are_not_reembedded(patched, monkeypatch):
    counter = _CountingEmbedder()
    monkeypatch.setattr(index_mod, "get_embedder", lambda name: counter)

    index_mod.run_index()
    after_first = counter.embedded
    assert after_first > 0

    # Nothing changed: not a single passage should hit the embedder again.
    index_mod.run_index()
    assert counter.embedded == after_first

    # Touching one file re-embeds that file alone.
    kt = patched.target_repo_path / "vector/src/main/java/im/vector/app/Foo.kt"
    kt.write_text("package im.vector.app\n\nclass Foo {\n    fun bar() = 42\n}\n")
    index_mod.run_index()
    assert counter.embedded == after_first + 1
