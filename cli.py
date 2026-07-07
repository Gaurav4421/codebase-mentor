"""
Terminal entrypoint for Codebase Mentor. Thin wiring around pipeline.CodebaseMentor,
same as streamlit_app.py -- exists so the tool is usable (and scriptable/demo-able)
without spinning up a Streamlit server.

Usage:
    python cli.py index /path/to/repo          # one-shot: index + interactive chat
    python cli.py index /path/to/repo --ask "What does the parser module do?"
"""
import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from llm.models import GeminiClient
from pipeline import CodebaseMentor

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING"))


def _print_banner(repo_path: str, num_chunks: int) -> None:
    print(f"\nIndexed {num_chunks} chunks from {repo_path}")
    print("Ask questions about the codebase. Type 'exit' or 'quit' to leave.\n")


def run_interactive(mentor: CodebaseMentor) -> None:
    memory = mentor.new_conversation()
    while True:
        try:
            query = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        answer, intent, _ = mentor.ask(query, memory)
        memory.add_user_turn(query)
        memory.add_assistant_turn(answer)
        memory.trim()

        print(f"\nmentor [{intent}]> {answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Codebase Mentor CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index a repository and start chatting.")
    index_parser.add_argument("repo_path", help="Path to the local repository to index.")
    index_parser.add_argument(
        "--ask", default=None,
        help="Ask a single question non-interactively and exit (useful for scripts/CI).",
    )

    args = parser.parse_args()

    try:
        mentor = CodebaseMentor(llm=GeminiClient())
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.command == "index":
        try:
            repo_index = mentor.index_repository(args.repo_path)
        except (ValueError, FileNotFoundError) as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)

        if args.ask:
            answer, intent, _ = mentor.ask(args.ask, mentor.new_conversation())
            print(f"[{intent}] {answer}")
            return

        _print_banner(args.repo_path, len(repo_index.chunks))
        run_interactive(mentor)


if __name__ == "__main__":
    main()
