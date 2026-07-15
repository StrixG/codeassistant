"""Kotlin chunker tests.

Token counting is a word count here: deterministic, offline, and enough to
drive the size-based branches.
"""

from assistant.indexer.code_chunker import chunk_kotlin, strip_license_header

LICENSE = """/*
 * Copyright 2024 New Vector Ltd.
 *
 * SPDX-License-Identifier: AGPL-3.0-only
 * Please see LICENSE files in the repository root for full details.
 */
"""


def wc(text: str) -> int:
    return len(text.split())


def test_license_header_stripped_but_plain_comment_kept():
    src = LICENSE + "package im.vector.app\n\nclass A"
    assert "Copyright" not in strip_license_header(src)
    assert "class A" in strip_license_header(src)

    plain = "/* just a note */\nclass A"
    assert strip_license_header(plain) == plain


def test_top_level_symbols_become_separate_chunks():
    src = LICENSE + """package im.vector.app

import java.util.Locale

class A {
    fun inside() = 1
}

object B {
    val x = 2
}

fun topLevel(): Int {
    return 3
}
"""
    chunks = chunk_kotlin(src, file_path="a/A.kt", count_tokens=wc)

    assert [c.heading_path for c in chunks] == ["A", "B", "topLevel"]
    # Imports are dropped; package survives as context.
    assert all("import java.util.Locale" not in c.text for c in chunks)
    assert all("package im.vector.app" in c.text for c in chunks)


def test_nested_declarations_stay_inside_their_parent():
    src = """package p

class Outer {

    @AssistedFactory
    interface Factory : MavericksAssistedViewModelFactory<Outer, State> {
        override fun create(initialState: State): Outer
    }

    companion object : MavericksViewModelFactory<Outer, State> by hiltMavericksViewModelFactory()

    fun handle(action: Action) {
        post(action)
    }
}
"""
    chunks = chunk_kotlin(src, file_path="p/Outer.kt", count_tokens=wc)

    assert len(chunks) == 1
    assert chunks[0].heading_path == "Outer"
    for member in ("interface Factory", "companion object", "fun handle"):
        assert member in chunks[0].text


def test_multiline_constructor_params_are_not_symbols():
    """Regression: `private val a: A,` sits at brace depth 0 inside the
    constructor's parens, and must not read as a top-level declaration."""
    src = """package p

class QrCodeScannerViewModel @AssistedInject constructor(
        @Assisted initialState: VectorDummyViewState,
        private val session: Session,
        private val clock: Clock,
) : VectorViewModel<A, B, C>(initialState) {

    fun handle(action: Action) = Unit
}
"""
    chunks = chunk_kotlin(src, file_path="p/Vm.kt", count_tokens=wc)

    assert [c.heading_path for c in chunks] == ["QrCodeScannerViewModel"]
    assert "private val session" in chunks[0].text


def test_raw_string_braces_do_not_desync_depth():
    """Braces inside a `\"\"\"` literal must not count toward nesting."""
    src = '''package p

fun query(): String {
    return """{"a": {"b": 1}}"""
}

fun after(): Int {
    return 1
}
'''
    chunks = chunk_kotlin(src, file_path="p/Q.kt", count_tokens=wc)

    assert [c.heading_path for c in chunks] == ["query", "after"]


def test_braces_in_line_comments_and_strings_ignored():
    src = '''package p

fun a(): String {
    // closing brace } in a comment
    val s = "an unbalanced { in a string"
    return s
}

fun b(): Int = 2
'''
    chunks = chunk_kotlin(src, file_path="p/C.kt", count_tokens=wc)

    assert [c.heading_path for c in chunks] == ["a", "b"]


def test_expression_body_functions_split():
    src = """package p

fun a() = 1

fun b() = 2
"""
    chunks = chunk_kotlin(src, file_path="p/E.kt", count_tokens=wc)

    assert [c.heading_path for c in chunks] == ["a", "b"]


def test_every_code_line_survives_intact():
    src = "package p\n\n" + "\n".join(
        f"fun f{i}() {{\n    return {i}\n}}\n" for i in range(20)
    )
    # 12 words fits "p/M.kt package p f0 fun f0() { return 0 }" — one chunk
    # per function, so any split would be the chunker tearing a symbol.
    chunks = chunk_kotlin(src, file_path="p/M.kt", count_tokens=wc, max_tokens=12)
    original = {ln for ln in src.splitlines() if ln.strip()}
    context = {"p/M.kt", "package p"}

    assert len(chunks) == 20
    for c in chunks:
        body = [ln for ln in c.text.splitlines() if ln.strip()]
        # Anything that isn't the prefix we added must be an untouched
        # original line — no chunk edge lands inside a line.
        code = [ln for ln in body if ln not in context and ln != c.heading_path]
        assert code, "chunk carries no code"
        for line in code:
            assert line in original


def test_oversized_class_splits_at_member_boundaries():
    members = "\n\n".join(
        f"    fun member{i}() {{\n        doSomething({i})\n        doMore({i})\n    }}"
        for i in range(6)
    )
    src = f"package p\n\nclass Big {{\n\n{members}\n}}\n"

    chunks = chunk_kotlin(src, file_path="p/Big.kt", count_tokens=wc, max_tokens=25)

    assert len(chunks) > 1
    assert all(c.heading_path.startswith("Big > member") for c in chunks)
    # The class signature rides along as context on every piece.
    assert all("class Big" in c.text for c in chunks)


def test_oversized_member_hard_split_keeps_lines_whole():
    body = "\n".join(f"        step{i}()" for i in range(40))
    src = f"package p\n\nclass Big {{\n    fun huge() {{\n{body}\n    }}\n}}\n"

    chunks = chunk_kotlin(src, file_path="p/H.kt", count_tokens=wc, max_tokens=12)
    original = set(line for line in src.splitlines() if line.strip())

    assert len(chunks) > 1
    assert any("часть" in c.heading_path for c in chunks)
    for c in chunks:
        # Skip the context prefix lines; the code lines must be intact.
        for line in c.text.splitlines():
            if line.strip().startswith("step"):
                assert line in original


def test_empty_and_import_only_file_yields_nothing():
    assert chunk_kotlin("", count_tokens=wc) == []
    assert chunk_kotlin("package p\n\nimport a.b.C\n", count_tokens=wc) == []
