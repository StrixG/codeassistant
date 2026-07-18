"""File tools backing the goal-driven agent, plus their function-calling schemas.

Every tool operates on a single repository root (``TARGET_REPO_PATH``) captured
at construction. The model supplies paths, globs and text; it can never point a
tool at another directory.

Security invariants (the whole point of this layer):

* Every path is resolved and checked to stay inside the root — ``../`` traversal,
  symlinks and absolute paths that escape the root are refused.
* ``.git`` internals are off-limits (read and write); binary files are never
  read or written as text.
* Writes honour a ``dry_run`` overlay: nothing touches disk, mutations are
  staged in memory and a unified diff is produced at the end. Reads are
  overlay-aware, so a chain of edits in dry-run stays self-consistent.

Kept free of any LLM/MCP dependency so it is unit-testable on its own. The
semantic search mode reuses the existing :class:`RagSearcher` over the Chroma
code index; it degrades to a clear message when the code index is empty.
"""

from __future__ import annotations

import difflib
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from assistant.core.tools import Tool, ToolRegistry

if TYPE_CHECKING:
    from assistant.core.rag import RagSearcher

_GIT_GREP_TIMEOUT = 20
_MAX_READ_BYTES = 100_000
_MAX_LIST = 400
_MAX_GREP_LINES = 200
_MAX_SEMANTIC_HITS = 8

# Emitted per tool step so the agent loop / demo video can show progress.
StepLog = Callable[[str], None]


class FileToolError(ValueError):
    """Raised for disallowed or failed file operations; message is user-safe."""


def _is_binary(data: bytes) -> bool:
    """A NUL byte in the head is the classic 'this is not text' signal."""
    return b"\x00" in data[:4096]


class FileTools:
    """The five file tools, bound to one repo root, with a dry-run overlay.

    In dry-run mode ``write_file`` / ``edit_file`` never write to disk: they
    record the original on-disk content the first time a path is touched and
    keep the latest staged content in ``_staged``. ``read_file`` prefers staged
    content, so the agent sees its own pending edits. :meth:`pending_diff`
    renders every staged change as a single unified diff.
    """

    def __init__(
        self,
        root: Path,
        *,
        searcher: "RagSearcher | None" = None,
        dry_run: bool = False,
        max_read_bytes: int = _MAX_READ_BYTES,
    ) -> None:
        self._root = root.resolve()
        self._searcher = searcher
        self._dry_run = dry_run
        self._max_read_bytes = max_read_bytes
        # rel_path -> original content on disk when first touched (None = new file)
        self._orig: dict[str, str | None] = {}
        # rel_path -> latest staged content (dry-run overlay)
        self._staged: dict[str, str] = {}

    # --- path safety -----------------------------------------------------

    def _resolve(self, path: str) -> Path:
        """Resolve ``path`` under the root, refusing escapes and git internals."""
        if not path or not path.strip():
            raise FileToolError("empty path")
        target = (self._root / path).resolve()
        if target != self._root and self._root not in target.parents:
            raise FileToolError(f"path escapes repository root: {path!r}")
        rel_parts = target.relative_to(self._root).parts
        if ".git" in rel_parts:
            raise FileToolError(f"refusing to touch git-internal path: {path!r}")
        return target

    def _rel(self, target: Path) -> str:
        return str(target.relative_to(self._root))

    # --- tool handlers ---------------------------------------------------

    def list_files(self, glob_pattern: str) -> str:
        """List repo files matching a glob (e.g. ``vector-config/**/*.kt``)."""
        matches: list[str] = []
        for p in sorted(self._root.glob(glob_pattern)):
            if not p.is_file():
                continue
            try:
                rel = str(p.relative_to(self._root))
            except ValueError:
                continue  # glob wandered outside the root via a symlink
            if ".git" in p.relative_to(self._root).parts:
                continue
            matches.append(rel)
        if not matches:
            return f"No files match {glob_pattern!r}."
        head = matches[:_MAX_LIST]
        out = "\n".join(head)
        if len(matches) > _MAX_LIST:
            out += f"\n... ({len(matches) - _MAX_LIST} more, refine the glob)"
        return out

    def read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> str:
        """Read a text file relative to the repo root (dry-run overlay aware).

        ``offset`` (1-based line) and ``limit`` (line count) read a slice of a
        large file instead of the whole thing.
        """
        target = self._resolve(path)
        rel = self._rel(target)
        if rel in self._staged:  # show the agent its own pending edits
            text = self._staged[rel]
        else:
            if not target.is_file():
                raise FileToolError(f"not a file: {path!r}")
            data = target.read_bytes()
            if _is_binary(data):
                raise FileToolError(f"refusing to read binary file: {path!r}")
            text = data[: self._max_read_bytes].decode("utf-8", errors="replace")
            if len(data) > self._max_read_bytes:
                text += f"\n... (truncated at {self._max_read_bytes} bytes)"
        if offset is None and limit is None:
            return text
        lines = text.splitlines()
        start = max(0, (offset or 1) - 1)
        end = start + limit if limit is not None else None
        return "\n".join(lines[start:end])

    def search(self, query: str, mode: str = "text") -> str:
        """Search the repo. ``text`` = literal grep, ``semantic`` = code index."""
        if mode == "semantic":
            return self._search_semantic(query)
        if mode == "text":
            return self._search_text(query)
        raise FileToolError(f"unknown search mode {mode!r}; use 'text' or 'semantic'")

    def _search_text(self, query: str) -> str:
        # git grep: fast, tracked-files-only, -I skips binaries. Fixed argv,
        # no shell — the query is a positional pattern, never interpolated.
        try:
            proc = subprocess.run(
                ["git", "-C", str(self._root), "grep", "-n", "-I",
                 "--no-color", "-F", "-e", query],
                capture_output=True,
                text=True,
                timeout=_GIT_GREP_TIMEOUT,
            )
        except FileNotFoundError as e:
            raise FileToolError("git executable not found") from e
        except subprocess.TimeoutExpired as e:
            raise FileToolError("text search timed out") from e
        if proc.returncode == 1:  # git grep: 1 == no matches (not an error)
            return f"No text matches for {query!r}."
        if proc.returncode != 0:
            raise FileToolError(f"git grep failed: {proc.stderr.strip()}")
        lines = proc.stdout.splitlines()
        out = "\n".join(lines[:_MAX_GREP_LINES])
        if len(lines) > _MAX_GREP_LINES:
            out += f"\n... ({len(lines) - _MAX_GREP_LINES} more matches, refine the query)"
        return out

    def _search_semantic(self, query: str) -> str:
        if self._searcher is None:
            return "Semantic search unavailable: no code index loaded."
        hits = self._searcher.search(query, top_k=_MAX_SEMANTIC_HITS, where={"source": "code"})
        if not hits:
            return (
                "No semantic matches in the code index. "
                "Build it with `python -m assistant index`, or use mode='text'."
            )
        blocks = []
        for h in hits:
            head = h.file_path + (f" :: {h.heading_path}" if h.heading_path else "")
            blocks.append(f"[CODE: {head}]\n{h.text}")
        return "\n\n---\n\n".join(blocks)

    def write_file(self, path: str, content: str) -> str:
        """Create or overwrite a text file with ``content``."""
        target = self._resolve(path)
        rel = self._rel(target)
        if target.exists() and not target.is_file():
            raise FileToolError(f"not a regular file: {path!r}")
        if self._dry_run:
            self._stage(rel, target, content)
            return f"[dry-run] staged write to {rel} ({len(content)} bytes)."
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {rel} ({len(content)} bytes)."

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        """Replace ``old_text`` with ``new_text``; ``old_text`` must be unique."""
        target = self._resolve(path)
        rel = self._rel(target)
        current = self.read_file(path)  # overlay-aware
        count = current.count(old_text)
        if count == 0:
            raise FileToolError(f"old_text not found in {rel}")
        if count > 1:
            raise FileToolError(
                f"old_text is ambiguous in {rel}: found {count} times, expected exactly 1. "
                "Include more surrounding context to make it unique."
            )
        updated = current.replace(old_text, new_text)
        if self._dry_run:
            self._stage(rel, target, updated)
            return f"[dry-run] staged edit to {rel}."
        target.write_text(updated, encoding="utf-8")
        return f"Edited {rel}."

    # --- dry-run bookkeeping --------------------------------------------

    def _stage(self, rel: str, target: Path, content: str) -> None:
        if rel not in self._orig:
            # Capture the pristine on-disk content once (None if brand new).
            self._orig[rel] = (
                target.read_text(encoding="utf-8", errors="replace")
                if target.is_file()
                else None
            )
        self._staged[rel] = content

    def has_pending(self) -> bool:
        return bool(self._staged)

    def pending_diff(self) -> str:
        """Unified diff of all staged (dry-run) changes, empty string if none."""
        chunks: list[str] = []
        for rel in sorted(self._staged):
            old = self._orig.get(rel) or ""
            new = self._staged[rel]
            label = "new file" if self._orig.get(rel) is None else "modified"
            diff = difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{rel} ({label})",
                tofile=f"b/{rel}",
            )
            chunks.append("".join(diff).rstrip("\n"))
        return "\n\n".join(c for c in chunks if c)


# --- function-calling schemas -------------------------------------------

_LIST_PARAMS = {
    "type": "object",
    "properties": {
        "glob_pattern": {
            "type": "string",
            "description": "Glob relative to the repo root, e.g. 'vector-config/**/*.kt' or '*.md'.",
        }
    },
    "required": ["glob_pattern"],
}
_READ_PARAMS = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File path relative to the repo root."},
        "offset": {"type": "integer", "description": "Optional 1-based start line for a partial read."},
        "limit": {"type": "integer", "description": "Optional number of lines to read from offset."},
    },
    "required": ["path"],
}
_SEARCH_PARAMS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "What to look for."},
        "mode": {
            "type": "string",
            "enum": ["text", "semantic"],
            "description": (
                "'text' = literal grep across tracked files (exact identifiers, "
                "call sites). 'semantic' = nearest chunks from the code index "
                "(concepts, 'where is X handled')."
            ),
            "default": "text",
        },
    },
    "required": ["query"],
}
_WRITE_PARAMS = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File path relative to the repo root."},
        "content": {"type": "string", "description": "Full new file content."},
    },
    "required": ["path", "content"],
}
_EDIT_PARAMS = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File path relative to the repo root."},
        "old_text": {
            "type": "string",
            "description": "Exact text to replace. Must occur EXACTLY once in the file.",
        },
        "new_text": {"type": "string", "description": "Replacement text."},
    },
    "required": ["path", "old_text", "new_text"],
}


def build_registry(tools: FileTools) -> ToolRegistry:
    """Register the five file tools for DeepSeek function calling."""
    reg = ToolRegistry()
    reg.register(Tool(
        name="list_files",
        description="List repository files matching a glob pattern.",
        parameters=_LIST_PARAMS,
        handler=tools.list_files,
    ))
    reg.register(Tool(
        name="read_file",
        description="Read the text content of a file in the repository.",
        parameters=_READ_PARAMS,
        handler=tools.read_file,
    ))
    reg.register(Tool(
        name="search",
        description=(
            "Search the repository for code or text. Use mode='text' for exact "
            "identifiers and call sites, mode='semantic' for conceptual queries."
        ),
        parameters=_SEARCH_PARAMS,
        handler=tools.search,
    ))
    reg.register(Tool(
        name="write_file",
        description="Create a new file or overwrite an existing one with full content.",
        parameters=_WRITE_PARAMS,
        handler=tools.write_file,
    ))
    reg.register(Tool(
        name="edit_file",
        description=(
            "Replace a unique snippet in an existing file. old_text must match "
            "exactly once; otherwise the edit is rejected."
        ),
        parameters=_EDIT_PARAMS,
        handler=tools.edit_file,
    ))
    return reg
