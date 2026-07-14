"""MCP stdio server exposing read-only git tools over the target repo.

Runs as a separate process. The repo path is loaded from config at
startup and captured in the tool closures — the LLM can pass a file path
to ``read_file`` (validated against traversal) but can never redirect the
tools at a different repository.

Run standalone:  python -m assistant.mcp_server.server
The CLI spawns this over stdio via the MCP client.
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

# Quiet the per-request INFO chatter so it doesn't pollute the CLI's stderr.
logging.getLogger("mcp").setLevel(logging.WARNING)

from assistant.config import Config
from assistant.mcp_server import repo_tools

_cfg = Config.load(require_api_key=False)
_REPO = _cfg.target_repo_path

mcp = FastMCP("element-git")


@mcp.tool()
def git_current_branch() -> str:
    """Return the current git branch of the target repository."""
    return repo_tools.git_current_branch(_REPO)


@mcp.tool()
def git_list_files(prefix: str = "") -> str:
    """List tracked files. Optional path prefix filters results (e.g. 'docs/')."""
    return repo_tools.git_list_files(_REPO, prefix)


@mcp.tool()
def git_diff() -> str:
    """Return `git diff HEAD` of the target repository."""
    return repo_tools.git_diff(_REPO)


@mcp.tool()
def read_file(path: str) -> str:
    """Read a file by path relative to the repo root. Traversal outside is refused."""
    return repo_tools.read_file(_REPO, path)


def main() -> None:
    mcp.run()  # stdio transport by default


if __name__ == "__main__":
    main()
