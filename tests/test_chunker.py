"""Chunker tests: heading_path is preserved; no chunk exceeds the limit.

Uses a trivial whitespace token counter so the test is deterministic and
needs no model download.
"""

from __future__ import annotations

from assistant.indexer.chunker import chunk_markdown

# 1 token per whitespace-separated word. Deterministic, offline.
def wc(text: str) -> int:
    return len(text.split())


DOC = """# Title

Intro paragraph under the title.

## Notifications

Some text about notifications.

### Firebase Push

Details about firebase push here.

#### Deep subsection

This H4 stays as body of the Firebase Push section.
"""


def test_heading_path_preserved():
    chunks = chunk_markdown(DOC, count_tokens=wc, max_tokens=1000)
    paths = [c.heading_path for c in chunks]
    assert "Title" in paths
    assert "Title > Notifications" in paths
    assert "Title > Notifications > Firebase Push" in paths


def test_heading_prefixed_in_chunk_text():
    chunks = chunk_markdown(DOC, count_tokens=wc, max_tokens=1000)
    fb = next(c for c in chunks if c.heading_path == "Title > Notifications > Firebase Push")
    # The heading chain is prefixed so the chunk stands alone.
    assert fb.text.startswith("Title > Notifications > Firebase Push")
    # H4 content folded into this section, not a separate chunk.
    assert "Deep subsection" in fb.text


def test_no_chunk_exceeds_limit():
    # A single huge section must be split by paragraphs under the limit.
    paras = "\n\n".join(f"word " * 40 for _ in range(30))
    doc = f"## Big Section\n\n{paras}"
    limit = 50
    chunks = chunk_markdown(doc, count_tokens=wc, max_tokens=limit)
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= limit, f"chunk over limit: {c.token_count}"
    # Heading retained on every piece of the split section.
    assert all(c.heading_path == "Big Section" for c in chunks)


def test_oversized_single_paragraph_hard_split():
    # One paragraph bigger than the limit, no blank lines to split on.
    doc = "## S\n\n" + ("token " * 200)
    limit = 30
    chunks = chunk_markdown(doc, count_tokens=wc, max_tokens=limit)
    assert len(chunks) > 1
    assert all(c.token_count <= limit for c in chunks)


def test_headings_in_code_fence_ignored():
    doc = """## Real Heading

Text.

```bash
# this is a shell comment, not a heading
## neither is this
```

More text.
"""
    chunks = chunk_markdown(doc, count_tokens=wc, max_tokens=1000)
    paths = [c.heading_path for c in chunks]
    assert paths == ["Real Heading"]
    assert "# this is a shell comment" in chunks[0].text
