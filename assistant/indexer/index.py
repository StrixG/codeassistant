"""Index target-repo documentation into ChromaDB.

Walks the docs/README corpus (never source code), chunks each file by
markdown heading, embeds passages locally, and upserts into a persistent
Chroma collection.

Incremental by default: a sidecar state file maps each indexed file to a
content hash. On re-run, unchanged files are skipped; changed and deleted
files have their old chunks removed first, so no duplicates accumulate.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from assistant.config import Config
from assistant.core.embeddings import get_embedder
from assistant.indexer.chunker import chunk_markdown, default_token_counter

console = Console()

# Glob patterns, relative to the repo root. Docs only, never code.
_DOC_GLOBS = ["docs/*.md", "*/README.md"]
_ROOT_FILES = ["README.md", "CONTRIBUTING.md"]
# Defensive skips even if a glob would match them.
_SKIP_DIR_PARTS = {"images"}
_SKIP_SUFFIXES = {".pdf", ".mwb", ".png", ".jpg", ".jpeg", ".gif", ".svg"}


def _state_path(cfg: Config) -> Path:
    return cfg.chroma_path / "index_state.json"


def discover_files(repo: Path) -> list[str]:
    """Return sorted, unique repo-relative paths of documents to index."""
    found: set[str] = set()
    for name in _ROOT_FILES:
        if (repo / name).is_file():
            found.add(name)
    for pattern in _DOC_GLOBS:
        for p in repo.glob(pattern):
            if not p.is_file():
                continue
            if p.suffix.lower() in _SKIP_SUFFIXES:
                continue
            if _SKIP_DIR_PARTS & set(p.relative_to(repo).parts):
                continue
            found.add(str(p.relative_to(repo)))
    return sorted(found)


def _content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _git_blob_shas(repo: Path) -> dict[str, str]:
    """Map repo-relative path -> git blob sha (short). Empty if not a repo."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "-s"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    shas: dict[str, str] = {}
    for line in out.splitlines():
        # format: "<mode> <blobsha> <stage>\t<path>"
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) >= 2 and path:
            shas[path] = parts[1][:12]
    return shas


@dataclass
class IndexStats:
    indexed_files: int = 0
    skipped_files: int = 0
    deleted_files: int = 0
    chunks_added: int = 0


def run_index(force: bool = False) -> int:
    cfg = Config.load(require_api_key=False)
    repo = cfg.target_repo_path
    if not repo.is_dir():
        console.print(f"[red]TARGET_REPO_PATH is not a directory: {repo}[/red]")
        return 1

    cfg.chroma_path.mkdir(parents=True, exist_ok=True)

    import chromadb

    client = chromadb.PersistentClient(path=str(cfg.chroma_path))
    if force:
        try:
            client.delete_collection(cfg.chroma_collection)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        cfg.chroma_collection, metadata={"hnsw:space": "cosine"}
    )

    files = discover_files(repo)
    console.print(f"Discovered [bold]{len(files)}[/bold] documents under {repo}")

    state: dict[str, str] = {} if force else _load_state(cfg)
    blob_shas = _git_blob_shas(repo)
    counter = default_token_counter(cfg.embedding_model)
    embedder = get_embedder(cfg.embedding_model)

    current: dict[str, str] = {}
    to_index: list[str] = []
    stats = IndexStats()

    for rel in files:
        h = _content_hash(repo / rel)
        current[rel] = h
        if not force and state.get(rel) == h:
            stats.skipped_files += 1
        else:
            to_index.append(rel)

    # Files previously indexed but now gone from the corpus.
    deleted = [p for p in state if p not in current]

    # Remove stale chunks for changed + deleted files before re-adding.
    for rel in [*to_index, *deleted]:
        collection.delete(where={"file_path": rel})
    stats.deleted_files = len(deleted)

    for rel in to_index:
        text = (repo / rel).read_text(encoding="utf-8", errors="replace")
        chunks = chunk_markdown(text, count_tokens=counter)
        if not chunks:
            continue
        git_sha = blob_shas.get(rel, "untracked")
        ids = [f"{rel}::{i}" for i in range(len(chunks))]
        docs = [c.text for c in chunks]
        metas = [
            {
                "file_path": rel,
                "heading_path": c.heading_path,
                "git_sha": git_sha,
                "chunk_index": i,
            }
            for i, c in enumerate(chunks)
        ]
        embeddings = embedder.embed_passages(docs)
        collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        stats.indexed_files += 1
        stats.chunks_added += len(chunks)
        console.print(f"  indexed {rel} [dim]({len(chunks)} chunks, sha {git_sha})[/dim]")

    _save_state(cfg, current)

    console.print(
        f"\n[green]Done.[/green] indexed={stats.indexed_files} "
        f"skipped={stats.skipped_files} deleted={stats.deleted_files} "
        f"chunks_added={stats.chunks_added} "
        f"collection_total={collection.count()}"
    )
    return 0


def _load_state(cfg: Config) -> dict[str, str]:
    p = _state_path(cfg)
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_state(cfg: Config, state: dict[str, str]) -> None:
    _state_path(cfg).write_text(json.dumps(state, indent=2, sort_keys=True))
