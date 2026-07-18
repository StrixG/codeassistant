"""Index the support knowledge base (FAQ + product guide) into Chroma.

Reuses the same building blocks as ``assistant.indexer.index``: heading-
aligned markdown chunking (``chunk_markdown``) and the shared local
embedder (``get_embedder``) — so the support KB is embedded with the same
model and the same ``passage:`` / ``query:`` prefix convention as the main
docs index. What differs from ``assistant.indexer.index`` is deliberate:
this corpus is two fixed files (not a repo crawl with git blob shas and
incremental diffing), so it is re-embedded and rewritten wholesale on
every run rather than diffed — simpler, and cheap at this size.

Writes into a *separate* collection (``support_kb`` by default) at the
same Chroma path as the main index, so the two never mix: the main
assistant's ``rag_search`` tool queries ``element_docs``, the support
assistant queries ``support_kb``.

Run standalone:  python -m support_assistant.index_support_kb
"""

from __future__ import annotations

from rich.console import Console

from assistant.config import Config
from assistant.core.embeddings import get_embedder
from assistant.indexer.chunker import chunk_markdown, default_token_counter

console = Console()

# Fixed corpus: (filename, doc_type tag used in metadata + debug output).
_KB_FILES = [
    ("faq.md", "faq"),
    ("product_guide.md", "guide"),
]


def run_support_index(*, cfg: Config | None = None) -> int:
    cfg = cfg or Config.load(require_api_key=False)
    data_dir = cfg.support_data_dir
    if not data_dir.is_dir():
        console.print(f"[red]SUPPORT_DATA_DIR is not a directory: {data_dir}[/red]")
        return 1

    cfg.chroma_path.mkdir(parents=True, exist_ok=True)

    import chromadb

    client = chromadb.PersistentClient(path=str(cfg.chroma_path))
    try:
        client.delete_collection(cfg.support_chroma_collection)
    except Exception:
        pass
    collection = client.get_or_create_collection(cfg.support_chroma_collection)

    counter = default_token_counter(cfg.embedding_model)
    embedder = get_embedder(cfg.embedding_model)

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    chunks_by_file: dict[str, int] = {}

    for filename, doc_type in _KB_FILES:
        path = data_dir / filename
        if not path.is_file():
            console.print(f"[yellow]Skipping missing file: {path}[/yellow]")
            continue
        text = path.read_text(encoding="utf-8")
        chunks = chunk_markdown(text, count_tokens=counter)
        for i, c in enumerate(chunks):
            ids.append(f"{filename}::{i}")
            docs.append(c.text)
            metas.append(
                {
                    "file_path": filename,
                    "heading_path": c.heading_path,
                    "chunk_index": i,
                    "source": doc_type,
                }
            )
        chunks_by_file[filename] = len(chunks)

    if not ids:
        console.print("[red]No support KB chunks produced — nothing indexed.[/red]")
        return 1

    embeddings = embedder.embed_passages(docs)
    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

    for filename, n in chunks_by_file.items():
        console.print(f"  [dim]{filename}: {n} chunks[/dim]")
    console.print(
        f"\n[green]Done.[/green] collection={cfg.support_chroma_collection} "
        f"chunks_added={len(ids)} collection_total={collection.count()}"
    )
    return 0


def main() -> int:
    return run_support_index()


if __name__ == "__main__":
    raise SystemExit(main())
