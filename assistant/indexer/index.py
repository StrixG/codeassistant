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
from assistant.indexer.code_chunker import chunk_kotlin

console = Console()

# Glob patterns, relative to the repo root. Docs only, never code.
_DOC_GLOBS = ["docs/*.md", "*/README.md"]
_ROOT_FILES = ["README.md", "CONTRIBUTING.md"]
# Defensive skips even if a glob would match them.
_SKIP_DIR_PARTS = {"images"}
_SKIP_SUFFIXES = {".pdf", ".mwb", ".png", ".jpg", ".jpeg", ".gif", ".svg"}

# Kotlin sources. `src/*/` already excludes sibling build/ output.
_CODE_GLOBS = [
    "vector/src/*/**/*.kt",
    "vector-app/src/*/**/*.kt",
    "matrix-sdk-android/src/*/**/*.kt",
    "library/*/src/*/**/*.kt",
]
_CODE_SKIP_PARTS = {"build", "generated"}
_TEST_SOURCE_SETS = ("test", "androidTest", "sharedTest")

# Embedding one file at a time wastes most of the batch; 3.5k files would mean
# 3.5k tiny encode calls.
_EMBED_BATCH = 256


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


def _is_test_source(parts: tuple[str, ...]) -> bool:
    """True for <module>/src/<testSourceSet>/... paths.

    Matches test, testFdroid, androidTest, androidTestGplay, sharedTest —
    while leaving main, debug, release, fdroid, gplay and nightly alone.
    """
    for i, p in enumerate(parts):
        if p == "src" and i + 1 < len(parts):
            return parts[i + 1].startswith(_TEST_SOURCE_SETS)
    return False


def discover_code_files(repo: Path) -> list[str]:
    """Sorted, unique repo-relative paths of Kotlin sources to index."""
    found: set[str] = set()
    for pattern in _CODE_GLOBS:
        for p in repo.glob(pattern):
            if not p.is_file():
                continue
            parts = p.relative_to(repo).parts
            if _CODE_SKIP_PARTS & set(parts) or _is_test_source(parts):
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


def run_index(
    force: bool = False,
    *,
    cfg: Config | None = None,
    include_code: bool = True,
) -> int:
    cfg = cfg or Config.load(require_api_key=False)
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

    doc_files = discover_files(repo)
    code_files = discover_code_files(repo) if include_code else []
    source_of = {r: "docs" for r in doc_files} | {r: "code" for r in code_files}
    files = sorted(source_of)
    console.print(
        f"Discovered [bold]{len(doc_files)}[/bold] documents and "
        f"[bold]{len(code_files)}[/bold] Kotlin sources under {repo}"
    )

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

    # Remove stale chunks before re-adding — but only for files we actually
    # indexed before. On a cold run every file is in to_index and none is in
    # state, and firing a delete per file would mean thousands of no-ops.
    stale = [r for r in to_index if r in state]
    for rel in [*stale, *deleted]:
        collection.delete(where={"file_path": rel})
    stats.deleted_files = len(deleted)

    buf_ids: list[str] = []
    buf_docs: list[str] = []
    buf_metas: list[dict] = []

    def flush() -> None:
        if not buf_ids:
            return
        embeddings = embedder.embed_passages(buf_docs)
        collection.add(ids=buf_ids, documents=buf_docs, metadatas=buf_metas, embeddings=embeddings)
        buf_ids.clear()
        buf_docs.clear()
        buf_metas.clear()

    for rel in to_index:
        text = (repo / rel).read_text(encoding="utf-8", errors="replace")
        source = source_of[rel]
        if source == "code":
            chunks = chunk_kotlin(text, file_path=rel, count_tokens=counter)
        else:
            chunks = chunk_markdown(text, count_tokens=counter)
        if not chunks:
            continue
        git_sha = blob_shas.get(rel, "untracked")
        buf_ids.extend(f"{rel}::{i}" for i in range(len(chunks)))
        buf_docs.extend(c.text for c in chunks)
        buf_metas.extend(
            {
                "file_path": rel,
                "heading_path": c.heading_path,
                "git_sha": git_sha,
                "chunk_index": i,
                "source": source,
            }
            for i, c in enumerate(chunks)
        )
        stats.indexed_files += 1
        stats.chunks_added += len(chunks)
        if len(buf_ids) >= _EMBED_BATCH:
            flush()
            console.print(
                f"  [dim]{stats.indexed_files}/{len(to_index)} files, "
                f"{stats.chunks_added} chunks[/dim]"
            )
    flush()

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
