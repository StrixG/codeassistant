"""RAG search over the persistent Chroma collection.

Embeds the query locally with the e5 ``query:`` prefix and returns the
nearest document chunks with their source metadata, so the assistant can
cite ``file_path`` and ``heading_path``.
"""

from __future__ import annotations

from dataclasses import dataclass

from assistant.config import Config
from assistant.core.embeddings import get_embedder


@dataclass
class SearchHit:
    file_path: str
    heading_path: str
    git_sha: str
    text: str
    distance: float
    source: str = "docs"  # "docs" | "code"; defaults for pre-code indexes


class RagSearcher:
    def __init__(self, cfg: Config) -> None:
        import chromadb

        self._embedder = get_embedder(cfg.embedding_model)
        client = chromadb.PersistentClient(path=str(cfg.chroma_path))
        # get (not create): searching an unbuilt index is a usage error.
        self._collection = client.get_collection(cfg.chroma_collection)

    def count(self) -> int:
        return self._collection.count()

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        where: dict | None = None,
    ) -> list[SearchHit]:
        """Nearest chunks, optionally scoped by metadata (e.g. source=docs)."""
        if self._collection.count() == 0:
            return []
        qv = self._embedder.embed_query(query)
        res = self._collection.query(
            query_embeddings=[qv],
            n_results=top_k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[SearchHit] = []
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for doc, meta, dist in zip(docs, metas, dists):
            hits.append(
                SearchHit(
                    file_path=meta.get("file_path", "?"),
                    heading_path=meta.get("heading_path", ""),
                    git_sha=meta.get("git_sha", ""),
                    text=doc,
                    distance=float(dist),
                    source=meta.get("source", "docs"),
                )
            )
        return hits
