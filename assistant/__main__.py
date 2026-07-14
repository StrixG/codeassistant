"""CLI entrypoint: index | chat | eval.

Subcommands are wired up in later steps. This skeleton parses args and
dispatches; unimplemented commands raise a clear NotImplementedError.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="assistant", description="Element Android docs assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Index target repo docs into Chroma")
    p_index.add_argument("--force", action="store_true", help="Reindex all files, ignore hashes")

    sub.add_parser("chat", help="Start interactive REPL")
    sub.add_parser("eval", help="Run eval questions from eval/questions.yaml")

    args = parser.parse_args(argv)

    if args.command == "index":
        from assistant.indexer.index import run_index

        return run_index(force=args.force)
    if args.command == "chat":
        from assistant.cli.chat import run_chat

        return run_chat()
    if args.command == "eval":
        from assistant.eval_runner import run_eval

        return run_eval()

    parser.error(f"Unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
