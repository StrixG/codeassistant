"""File-tool safety and behaviour: path validation, edit uniqueness, dry-run."""

from __future__ import annotations

from pathlib import Path

import pytest

from file_agent.tools import FileTools, FileToolError


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    (r / "docs").mkdir(parents=True)
    (r / ".git").mkdir()
    (r / "docs" / "guide.md").write_text("# Guide\nhello world\n")
    (r / ".git" / "config").write_text("[core]\n")
    (r / "logo.png").write_bytes(b"\x89PNG\x00\x00binary")
    # A secret sibling to the repo, reachable only via traversal.
    (tmp_path / "secret.txt").write_text("TOP SECRET\n")
    return r


# --- path validation -----------------------------------------------------

def test_read_inside_ok(repo):
    assert "hello world" in FileTools(repo).read_file("docs/guide.md")


def test_read_offset_limit_slices_lines(repo):
    (repo / "docs" / "many.md").write_text("l1\nl2\nl3\nl4\nl5\n")
    out = FileTools(repo).read_file("docs/many.md", offset=2, limit=2)
    assert out == "l2\nl3"


def test_read_rejects_dotdot_traversal(repo):
    with pytest.raises(FileToolError):
        FileTools(repo).read_file("../secret.txt")


def test_read_rejects_absolute_path(repo):
    with pytest.raises(FileToolError):
        FileTools(repo).read_file("/etc/passwd")


def test_write_rejects_traversal(repo):
    with pytest.raises(FileToolError):
        FileTools(repo).write_file("../escape.txt", "nope")
    assert not (repo.parent / "escape.txt").exists()


def test_git_internals_refused(repo):
    tools = FileTools(repo)
    with pytest.raises(FileToolError):
        tools.read_file(".git/config")
    with pytest.raises(FileToolError):
        tools.write_file(".git/hooks/pre-commit", "x")


def test_binary_read_refused(repo):
    with pytest.raises(FileToolError):
        FileTools(repo).read_file("logo.png")


# --- edit_file uniqueness ------------------------------------------------

def test_edit_unique_ok(repo):
    tools = FileTools(repo)
    tools.edit_file("docs/guide.md", "hello world", "goodbye world")
    assert "goodbye world" in (repo / "docs" / "guide.md").read_text()


def test_edit_ambiguous_old_text_rejected(repo):
    (repo / "docs" / "dup.md").write_text("x\nx\n")
    tools = FileTools(repo)
    with pytest.raises(FileToolError) as e:
        tools.edit_file("docs/dup.md", "x", "y")
    assert "ambiguous" in str(e.value)
    assert (repo / "docs" / "dup.md").read_text() == "x\nx\n"  # untouched


def test_edit_missing_old_text_rejected(repo):
    with pytest.raises(FileToolError):
        FileTools(repo).edit_file("docs/guide.md", "not present", "y")


# --- dry-run -------------------------------------------------------------

def test_dry_run_write_does_not_touch_disk(repo):
    tools = FileTools(repo, dry_run=True)
    tools.write_file("NEW.md", "brand new\n")
    tools.edit_file("docs/guide.md", "hello world", "changed")
    assert not (repo / "NEW.md").exists()
    assert "hello world" in (repo / "docs" / "guide.md").read_text()  # original intact

    diff = tools.pending_diff()
    assert "NEW.md" in diff and "brand new" in diff
    assert "+changed" in diff and "-# Guide\n" not in diff  # only the edited line changed


def test_dry_run_read_sees_staged_edit(repo):
    """A chain of dry-run edits stays self-consistent via the overlay."""
    tools = FileTools(repo, dry_run=True)
    tools.edit_file("docs/guide.md", "hello world", "step one")
    assert "step one" in tools.read_file("docs/guide.md")
    tools.edit_file("docs/guide.md", "step one", "step two")  # edits the staged content
    assert "step two" in tools.read_file("docs/guide.md")
    assert "hello world" in (repo / "docs" / "guide.md").read_text()  # disk still pristine


def test_live_write_persists(repo):
    FileTools(repo).write_file("report.md", "done\n")
    assert (repo / "report.md").read_text() == "done\n"


# --- search --------------------------------------------------------------

def test_search_unknown_mode_rejected(repo):
    with pytest.raises(FileToolError):
        FileTools(repo).search("x", mode="fuzzy")


def test_semantic_without_index_is_graceful(repo):
    out = FileTools(repo, searcher=None).search("anything", mode="semantic")
    assert "unavailable" in out.lower()
