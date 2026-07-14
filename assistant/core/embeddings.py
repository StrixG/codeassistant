"""Local embeddings via sentence-transformers (multilingual e5).

DeepSeek has no /embeddings endpoint, so embeddings are computed locally.
The e5 family requires task prefixes: ``passage: `` for indexed documents
and ``query: `` for search queries. Getting these wrong quietly degrades
retrieval, so they live in one place.

Shared by the indexer (passages) and RAG search (queries) so both use the
exact same model and prefixes.
"""

from __future__ import annotations

from functools import lru_cache


class Embedder:
    def __init__(self, model_name: str) -> None:
        # Imported lazily: loading the model is slow and only needed when
        # indexing or searching, not on every CLI invocation.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        return self._encode(prefixed)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"query: {text}"])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,  # cosine similarity via inner product
            convert_to_numpy=True,
        )
        return vecs.tolist()


@lru_cache(maxsize=2)
def get_embedder(model_name: str) -> Embedder:
    """Process-wide cached embedder; the model loads once."""
    return Embedder(model_name)
