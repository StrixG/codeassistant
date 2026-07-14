"""Security test for MCP read_file: paths outside the repo are refused."""

from __future__ import annotations

from pathlib import Path

import pytest

from assistant.mcp_server import repo_tools
from assistant.mcp_server.repo_tools import RepoToolError


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    (r / "docs").mkdir(parents=True)
    (r / "docs" / "ok.md").write_text("inside content\n")
    # A secret sibling to the repo, reachable only via traversal.
    (tmp_path / "secret.txt").write_text("TOP SECRET\n")
    return r


def test_read_file_inside_ok(repo):
    assert repo_tools.read_file(repo, "docs/ok.md") == "inside content\n"


def test_read_file_rejects_dotdot_traversal(repo):
    with pytest.raises(RepoToolError):
        repo_tools.read_file(repo, "../secret.txt")


def test_read_file_rejects_deep_traversal(repo):
    with pytest.raises(RepoToolError):
        repo_tools.read_file(repo, "../../../../etc/passwd")


def test_read_file_rejects_absolute_path(repo):
    with pytest.raises(RepoToolError):
        repo_tools.read_file(repo, "/etc/passwd")


def test_read_file_rejects_nonexistent(repo):
    with pytest.raises(RepoToolError):
        repo_tools.read_file(repo, "docs/missing.md")
