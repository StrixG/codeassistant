"""Markdown chunking by heading structure.

Splits a markdown document into chunks aligned to heading boundaries
(H1-H3), not fixed character windows. Each chunk carries the chain of
enclosing headings (``heading_path``) and is prefixed with that context so
it stands alone when retrieved.

Token counting is injectable. The default counter uses the same tokenizer
as the embedding model (e5), so chunk budgets reflect what actually gets
embedded. Tests pass a trivial counter for determinism without network.

Chunk limit is 512 tokens, not the spec's soft ceiling of 800:
``multilingual-e5-small`` truncates input at 512 tokens, so a larger chunk
would silently lose its tail at embed time. 512 stays under the 800 limit,
so the "no chunk exceeds the limit" contract holds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

# e5-small hard sequence limit. Chunks above this get truncated at embed
# time, so we treat it as the hard max.
MAX_TOKENS = 512
# Soft floor: sections shorter than this are fine, it just marks "small".
MIN_TOKENS = 60

# Levels that open a new chunk boundary. Deeper headings (####+) stay as
# body content of their enclosing section.
_SPLIT_MAX_LEVEL = 3

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


@dataclass
class Chunk:
    text: str
    heading_path: str  # "A > B > C" of enclosing H1-H3 headings
    token_count: int
    # Filled in by the indexer, not the chunker:
    file_path: str = ""
    git_sha: str = ""


@dataclass
class _Section:
    heading_path: list[str]
    body: list[str] = field(default_factory=list)


CountTokens = Callable[[str], int]


@lru_cache(maxsize=1)
def default_token_counter(model_name: str = "intfloat/multilingual-e5-small") -> CountTokens:
    """Token counter backed by the embedding model's own tokenizer.

    Cached: the tokenizer is loaded once per process. Downloads only the
    tokenizer files (small) on first use.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)

    def count(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    return count


def _parse_sections(md: str) -> list[_Section]:
    """Walk the document, splitting into sections at H1-H3 headings.

    Headings inside fenced code blocks are ignored.
    """
    sections: list[_Section] = []
    # heading_stack[i] is the active heading text at level i+1, or None.
    heading_stack: list[str | None] = [None] * 6
    current = _Section(heading_path=[])
    sections.append(current)
    in_fence = False

    for line in md.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            current.body.append(line)
            continue
        if in_fence:
            current.body.append(line)
            continue

        m = _HEADING_RE.match(line)
        if not m:
            current.body.append(line)
            continue

        level = len(m.group(1))
        title = m.group(2).strip()

        # Update the heading stack: set this level, clear deeper levels.
        heading_stack[level - 1] = title
        for deeper in range(level, 6):
            heading_stack[deeper] = None

        if level <= _SPLIT_MAX_LEVEL:
            # Open a new section boundary.
            path = [h for h in heading_stack[:_SPLIT_MAX_LEVEL] if h]
            current = _Section(heading_path=path)
            sections.append(current)
        else:
            # Deeper heading stays as body content of the current section.
            current.body.append(line)

    # Drop sections with no real content and no heading (e.g. leading empty).
    return [s for s in sections if s.heading_path or "".join(s.body).strip()]


def _split_paragraph(para: str, count: CountTokens, budget: int) -> list[str]:
    """Hard-split a single oversized paragraph by sentences, then words."""
    pieces: list[str] = []
    units = re.split(r"(?<=[.!?])\s+", para)
    if len(units) == 1:
        units = para.split()
    cur: list[str] = []
    for u in units:
        cand = (" ".join(cur + [u])).strip()
        if cur and count(cand) > budget:
            pieces.append(" ".join(cur).strip())
            cur = [u]
        else:
            cur.append(u)
    if cur:
        pieces.append(" ".join(cur).strip())
    return [p for p in pieces if p]


def _chunk_section(sec: _Section, count: CountTokens, max_tokens: int) -> list[Chunk]:
    heading_path = " > ".join(sec.heading_path)
    prefix = f"{heading_path}\n\n" if heading_path else ""
    body = "\n".join(sec.body).strip()

    full = f"{prefix}{body}".strip()
    if not full:
        return []
    if count(full) <= max_tokens:
        return [Chunk(text=full, heading_path=heading_path, token_count=count(full))]

    # Section too big: pack paragraphs greedily, heading prefixed on each.
    # Budget for body leaves room for the repeated heading prefix.
    prefix_tokens = count(prefix) if prefix else 0
    body_budget = max(1, max_tokens - prefix_tokens)

    paragraphs = re.split(r"\n\s*\n", body)
    packed: list[str] = []
    cur: list[str] = []

    def flush() -> None:
        if cur:
            packed.append("\n\n".join(cur).strip())
            cur.clear()

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if count(para) > body_budget:
            # Paragraph alone exceeds budget: flush, then hard-split it.
            flush()
            packed.extend(_split_paragraph(para, count, body_budget))
            continue
        cand = "\n\n".join(cur + [para])
        if cur and count(cand) > body_budget:
            flush()
            cur.append(para)
        else:
            cur.append(para)
    flush()

    chunks: list[Chunk] = []
    for piece in packed:
        text = f"{prefix}{piece}".strip()
        chunks.append(Chunk(text=text, heading_path=heading_path, token_count=count(text)))
    return chunks


def chunk_markdown(
    md: str,
    *,
    count_tokens: CountTokens | None = None,
    max_tokens: int = MAX_TOKENS,
) -> list[Chunk]:
    """Split markdown into heading-aligned chunks, none exceeding ``max_tokens``."""
    count = count_tokens or default_token_counter()
    chunks: list[Chunk] = []
    for sec in _parse_sections(md):
        chunks.extend(_chunk_section(sec, count, max_tokens))
    return chunks
