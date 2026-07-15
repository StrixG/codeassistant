"""The seams the PR-review pipeline leans on: JSON mode and metadata filters.

Fakes stand in for the OpenAI client and the Chroma collection so nothing
touches the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from assistant.config import Config
from assistant.core import rag as rag_mod
from assistant.core.llm import _THINKING_OFF, DeepSeekClient
from assistant.core.rag import RagSearcher, SearchHit


def _cfg(tmp_path: Path) -> Config:
    return Config(
        deepseek_api_key="k",
        deepseek_model="deepseek-v4-pro",
        deepseek_base_url="https://api.deepseek.com",
        request_timeout=60,
        embedding_model="fake",
        chroma_path=tmp_path / "chroma",
        chroma_collection="c",
        target_repo_path=tmp_path,
        target_default_branch="develop",
        metrics_path=tmp_path / "m.jsonl",
    )


class _FakeCompletions:
    def __init__(self) -> None:
        self.seen: dict = {}

    def create(self, **params):
        self.seen = params
        return object()


class _FakeOpenAI:
    def __init__(self, **_) -> None:
        self.chat = type("C", (), {"completions": _FakeCompletions()})()


@pytest.fixture
def client(tmp_path, monkeypatch) -> DeepSeekClient:
    monkeypatch.setattr("assistant.core.llm.OpenAI", _FakeOpenAI)
    return DeepSeekClient(_cfg(tmp_path))


def test_response_format_reaches_the_api(client):
    client.chat([{"role": "user", "content": "hi"}], response_format={"type": "json_object"})
    assert client._client.chat.completions.seen["response_format"] == {"type": "json_object"}


def test_plain_call_sends_no_response_format(client):
    client.chat([{"role": "user", "content": "hi"}])
    seen = client._client.chat.completions.seen

    # Day 31 behaviour must be byte-identical: no new keys, same pinning.
    assert "response_format" not in seen
    assert seen["temperature"] == 0.0
    assert seen["extra_body"] == _THINKING_OFF
    assert seen["tools"] is None
    assert seen["tool_choice"] is None


def test_kwargs_can_override_a_default(client):
    client.chat([{"role": "user", "content": "hi"}], max_tokens=64)
    assert client._client.chat.completions.seen["max_tokens"] == 64


class _FakeCollection:
    def __init__(self, meta: dict | None = None) -> None:
        self.seen: dict = {}
        self._meta = meta or {"file_path": "a.md", "heading_path": "H", "git_sha": "s"}

    def count(self) -> int:
        return 1

    def query(self, **kwargs):
        self.seen = kwargs
        return {
            "documents": [["text"]],
            "metadatas": [[self._meta]],
            "distances": [[0.25]],
        }


@pytest.fixture
def searcher(tmp_path, monkeypatch):
    def _make(meta: dict | None = None) -> tuple[RagSearcher, _FakeCollection]:
        coll = _FakeCollection(meta)
        monkeypatch.setattr(rag_mod, "get_embedder", lambda name: _FakeEmbedder())
        monkeypatch.setattr(
            "chromadb.PersistentClient",
            lambda path: type("Cl", (), {"get_collection": lambda self, n: coll})(),
        )
        return RagSearcher(_cfg(tmp_path)), coll

    return _make


class _FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


def test_search_forwards_where(searcher):
    s, coll = searcher()
    s.search("q", where={"source": "code"})
    assert coll.seen["where"] == {"source": "code"}


def test_search_without_where_passes_none(searcher):
    s, coll = searcher()
    s.search("q")
    assert coll.seen["where"] is None


def test_hit_carries_source_and_defaults_to_docs(searcher):
    s, _ = searcher({"file_path": "a.kt", "heading_path": "C", "git_sha": "x", "source": "code"})
    assert s.search("q")[0].source == "code"

    s2, _ = searcher({"file_path": "a.md", "heading_path": "H", "git_sha": "x"})
    assert s2.search("q")[0].source == "docs"


def test_search_hit_still_constructs_positionally():
    # Old callers build SearchHit without source.
    hit = SearchHit("a.md", "H", "sha", "text", 0.5)
    assert hit.source == "docs"
