"""CLI: ``python -m file_agent "цель текстом" [--dry-run]``.

The goal is stated at the level of intent; the agent picks the tools. With
``--dry-run`` no file is written — staged changes are printed as a unified diff.
"""

from __future__ import annotations

import argparse
import sys

from assistant.core.llm import LlmError
from file_agent.agent import build_and_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="file_agent",
        description="Goal-driven file agent over TARGET_REPO_PATH.",
    )
    parser.add_argument("goal", help="The goal, in plain language (e.g. 'обнови документацию').")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write to disk; print a unified diff of staged changes instead.",
    )
    args = parser.parse_args(argv)

    try:
        result = build_and_run(args.goal, dry_run=args.dry_run)
    except LlmError as e:
        print(f"DeepSeek unavailable: {e}", file=sys.stderr)
        return 1
    return 2 if result.hit_limit else 0


if __name__ == "__main__":
    sys.exit(main())
