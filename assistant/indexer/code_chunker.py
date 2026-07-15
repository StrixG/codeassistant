"""Kotlin chunking by symbol boundaries.

Splits a Kotlin file at top-level declarations (``class`` / ``object`` /
``interface`` / ``fun`` / ``val`` ...) rather than at a fixed line count, so a
retrieved chunk is a whole symbol instead of an arbitrary window. Each chunk
carries its file path, package and symbol path as a prefix, mirroring what
``chunk_markdown`` does with heading chains.

Structure comes from brace depth, never indentation: Element's style indents
continuations by 8 spaces, so any indent-based heuristic misreads it. Depth is
counted on lines stripped of comments and string literals — ``matrix-sdk-android``
is full of raw strings holding JSON, and their braces would otherwise desync
the scanner for the rest of the file.

Regex-based, not a real parser: exotic syntax may land a boundary in the wrong
place. The cost of that is a slightly worse retrieval hit, not a crash.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from assistant.indexer.chunker import (
    MAX_TOKENS,
    Chunk,
    CountTokens,
    default_token_counter,
)

_LICENSE_RE = re.compile(r"\A\s*/\*.*?\*/\s*", re.DOTALL)
_LICENSE_MARKERS = ("Copyright", "SPDX-License-Identifier", "Licensed under")

_IMPORT_RE = re.compile(r"^\s*import\s+")
_PACKAGE_RE = re.compile(r"^\s*package\s+(\S+)")

_MODIFIERS = (
    "public|private|internal|protected|abstract|final|open|sealed|data|inner|"
    "enum|annotation|value|inline|expect|actual|external|override|suspend|"
    "operator|infix|tailrec|lateinit|const|companion|vararg|noinline|crossinline"
)
_DECL_RE = re.compile(
    r"^[ \t]*"
    r"(?:@[\w.]+(?:\([^\n]*\))?[ \t]+)*"
    rf"(?:(?:{_MODIFIERS})[ \t]+)*"
    r"(?P<kind>class|interface|object|fun|val|var|typealias)\b"
    r"(?:[ \t]+(?P<name>`[^`]+`|[A-Za-z_][A-Za-z0-9_]*))?"
)
# Lines that belong to the declaration above them rather than the symbol before.
_ATTACHES_FORWARD_RE = re.compile(r"^\s*(?:@|/\*\*|\*|//)")


@dataclass
class _Scan:
    """Mutable lexer state carried across lines."""

    in_block_comment: bool = False
    in_raw_string: bool = False


@dataclass
class _Symbol:
    name: str
    kind: str
    start: int  # line index, inclusive
    end: int  # line index, exclusive


def strip_license_header(src: str) -> str:
    """Drop the leading ``/* */`` block, but only if it reads like a licence."""
    m = _LICENSE_RE.match(src)
    if m and any(k in m.group(0) for k in _LICENSE_MARKERS):
        return src[m.end() :]
    return src


def _strip_line(line: str, st: _Scan) -> str:
    """Line with comments and literals removed; ``st`` carries over lines."""
    out: list[str] = []
    i, n = 0, len(line)
    while i < n:
        if st.in_block_comment:
            j = line.find("*/", i)
            if j < 0:
                return "".join(out)
            st.in_block_comment = False
            i = j + 2
            continue
        if st.in_raw_string:
            j = line.find('"""', i)
            if j < 0:
                return "".join(out)
            st.in_raw_string = False
            i = j + 3
            continue
        if line.startswith('"""', i):
            st.in_raw_string = True
            i += 3
            continue
        if line.startswith("/*", i):
            st.in_block_comment = True
            i += 2
            continue
        if line.startswith("//", i):
            break
        c = line[i]
        if c in '"\'':
            i += 1
            while i < n and line[i] != c:
                i += 2 if line[i] == "\\" else 1
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _scan_depths(lines: list[str]) -> tuple[list[int], list[int]]:
    """Brace and paren depth *before* each line.

    Paren depth matters as much as brace depth: inside a multi-line primary
    constructor, ``private val session: Session,`` sits at brace depth 0 and
    would otherwise read as a top-level declaration.
    """
    st = _Scan()
    braces: list[int] = []
    parens: list[int] = []
    b = p = 0
    for line in lines:
        braces.append(b)
        parens.append(p)
        clean = _strip_line(line, st)
        b = max(0, b + clean.count("{") - clean.count("}"))
        p = max(0, p + clean.count("(") - clean.count(")"))
    return braces, parens


def _extend_start_backwards(lines: list[str], i: int, floor: int) -> int:
    """Pull annotations and doc comments above a declaration into it."""
    j = i
    while j > floor and _ATTACHES_FORWARD_RE.match(lines[j - 1]):
        j -= 1
    return j


def _symbols_at(
    lines: list[str],
    braces: list[int],
    parens: list[int],
    level: int,
    lo: int,
    hi: int,
) -> list[_Symbol]:
    """Declarations sitting exactly at ``level`` within ``[lo, hi)``.

    A symbol runs until the next declaration at the same level. Since a
    declaration only counts when the scanner is at that exact depth, the next
    one cannot be nested inside the previous body — so the end needs no search.
    """
    starts: list[tuple[int, str, str]] = []
    for i in range(lo, hi):
        if braces[i] != level or parens[i] != 0:
            continue
        m = _DECL_RE.match(lines[i])
        if m:
            starts.append((i, m.group("kind"), m.group("name") or "<anon>"))

    syms: list[_Symbol] = []
    for k, (i, kind, name) in enumerate(starts):
        prev_end = syms[-1].end if syms else lo
        start = _extend_start_backwards(lines, i, prev_end)
        end = starts[k + 1][0] if k + 1 < len(starts) else hi
        if syms:
            syms[-1].end = start
        syms.append(_Symbol(name=name.strip("`"), kind=kind, start=start, end=end))
    return syms


def _mk(ctx: str, heading: str, body: str, count: CountTokens) -> Chunk:
    text = f"{ctx}{heading}\n\n{body}".strip() if heading else f"{ctx}{body}".strip()
    return Chunk(text=text, heading_path=heading, token_count=count(text))


def _hard_split(
    body: list[str], ctx: str, heading: str, count: CountTokens, max_tokens: int
) -> list[Chunk]:
    """Pack whole lines. A symbol is never cut mid-line."""
    out: list[Chunk] = []
    cur: list[str] = []
    part = 1
    for ln in body:
        cand = "\n".join([*cur, ln])
        if cur and count(f"{ctx}{heading}\n\n{cand}") > max_tokens:
            out.append(_mk(ctx, f"{heading} (часть {part})", "\n".join(cur), count))
            part += 1
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        tail = f"{heading} (часть {part})" if part > 1 else heading
        out.append(_mk(ctx, tail, "\n".join(cur), count))
    return out


def _chunk_symbol(
    lines: list[str],
    braces: list[int],
    parens: list[int],
    sym: _Symbol,
    ctx: str,
    path: list[str],
    count: CountTokens,
    max_tokens: int,
) -> list[Chunk]:
    heading = " > ".join([*path, sym.name])
    body = "\n".join(lines[sym.start : sym.end]).rstrip()
    if not body.strip():
        return []

    if count(f"{ctx}{heading}\n\n{body}") <= max_tokens:
        return [_mk(ctx, heading, body, count)]

    # Too big: cut at its own members, keeping the signature as context.
    inner = braces[sym.start] + 1
    members = _symbols_at(lines, braces, parens, inner, sym.start, sym.end)
    if members:
        header = "\n".join(lines[sym.start : members[0].start]).rstrip()
        member_ctx = f"{ctx}{heading}\n{header}\n\n" if header else ctx
        out: list[Chunk] = []
        for m in members:
            out.extend(
                _chunk_symbol(
                    lines, braces, parens, m, member_ctx, [*path, sym.name], count, max_tokens
                )
            )
        return out

    return _hard_split(lines[sym.start : sym.end], ctx, heading, count, max_tokens)


def chunk_kotlin(
    src: str,
    *,
    file_path: str = "",
    count_tokens: CountTokens | None = None,
    max_tokens: int = MAX_TOKENS,
) -> list[Chunk]:
    """Split Kotlin source into chunks aligned to symbol boundaries.

    Chunk edges always fall on line boundaries, and a symbol that fits within
    ``max_tokens`` is never split.
    """
    count = count_tokens or default_token_counter()
    lines_in = strip_license_header(src).splitlines()

    package = ""
    lines: list[str] = []
    for ln in lines_in:
        m = _PACKAGE_RE.match(ln)
        if m and not package:
            package = m.group(1)
            continue
        if _IMPORT_RE.match(ln):
            # ~60 imports per file would eat the whole chunk budget.
            continue
        lines.append(ln)

    if not any(ln.strip() for ln in lines):
        return []

    ctx_bits = [b for b in (file_path, f"package {package}" if package else "") if b]
    ctx = ("\n".join(ctx_bits) + "\n") if ctx_bits else ""

    braces, parens = _scan_depths(lines)
    top = _symbols_at(lines, braces, parens, 0, 0, len(lines))

    chunks: list[Chunk] = []
    preamble = "\n".join(lines[: top[0].start] if top else lines).strip()
    if preamble:
        chunks.append(_mk(ctx, "", preamble, count))
    for sym in top:
        chunks.extend(_chunk_symbol(lines, braces, parens, sym, ctx, [], count, max_tokens))

    return [c for c in chunks if c.text.strip()]
