"""Pure git/file helpers backing the MCP tools.

Security invariants (the whole point of this layer):

* The repo path is passed in by the caller from config — never taken from
  an LLM argument.
* Every git call is a fixed subcommand run via ``subprocess.run`` with an
  argument list. No ``shell=True``, no string interpolation of untrusted
  input into a command.
* ``read_file`` resolves the requested path and refuses anything that
  escapes the repository root (``../`` traversal, absolute paths).

Kept free of MCP so it is unit-testable on its own.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_GIT_TIMEOUT = 15


class RepoToolError(ValueError):
    """Raised for disallowed or failed repo operations."""


def _run_git(repo: Path, args: list[str]) -> str:
    """Run ``git -C <repo> <args...>`` safely and return stdout.

    ``args`` is a fixed subcommand chosen by the caller, never free-form
    input from the model.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=True,
        )
    except FileNotFoundError as e:
        raise RepoToolError("git executable not found") from e
    except subprocess.TimeoutExpired as e:
        raise RepoToolError("git command timed out") from e
    except subprocess.CalledProcessError as e:
        raise RepoToolError(f"git failed: {e.stderr.strip() or e}") from e
    return proc.stdout


def git_current_branch(repo: Path) -> str:
    return _run_git(repo, ["rev-parse", "--abbrev-ref", "HEAD"]).strip()


def git_list_files(repo: Path, prefix: str = "") -> str:
    args = ["ls-files"]
    if prefix:
        # prefix is passed as a pathspec argument, not shell-interpolated.
        args += ["--", prefix]
    return _run_git(repo, args).strip()


def git_diff(repo: Path) -> str:
    out = _run_git(repo, ["diff", "HEAD"]).strip()
    return out or "(no unstaged/staged changes vs HEAD)"


def read_file(repo: Path, path: str, *, max_bytes: int = 200_000) -> str:
    """Read a file by repo-relative path, refusing traversal outside the repo."""
    repo = repo.resolve()
    target = (repo / path).resolve()
    if target != repo and repo not in target.parents:
        raise RepoToolError(f"path escapes repository: {path!r}")
    if not target.is_file():
        raise RepoToolError(f"not a file: {path!r}")
    data = target.read_bytes()[:max_bytes]
    return data.decode("utf-8", errors="replace")
