"""Command-line entry point.

    python -m tripmate "Do I need a visa for Japan?"
    python -m tripmate --verbose "What should I pack for Reykjavik in December?"
    python -m tripmate --interactive

The answer goes to stdout; structured logs go to stderr. That keeps
`python -m tripmate "..." > answer.txt` doing the obvious thing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tripmate.agent.orchestrator import TripMateAgent
from tripmate.config import ConfigError, load_config
from tripmate.llm.base import ProviderError
from tripmate.logging_setup import force_utf8_output, setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tripmate",
        description="TripMate - an AI travel assistant for Tokyo, Reykjavik, Bangkok, and Barcelona.",
    )
    parser.add_argument("query", nargs="*", help="Your travel question.")
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Start a REPL session."
    )
    parser.add_argument(
        "--provider",
        choices=("groq", "anthropic"),
        help="Override TRIPMATE_PROVIDER for this run.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print the reasoning trace after the answer."
    )
    parser.add_argument(
        "--trace-json",
        metavar="PATH",
        help="Write the full trace to PATH as JSON (useful for example logs).",
    )
    parser.add_argument(
        "--log-file", metavar="PATH", help="Also append structured JSON logs to PATH."
    )
    parser.add_argument(
        "--quiet-logs",
        action="store_true",
        help="Suppress structured logs on stderr (sets level to ERROR).",
    )
    return parser


def _answer_once(agent: TripMateAgent, query: str, args: argparse.Namespace) -> None:
    response = agent.run(query)

    print()
    print(response.answer)
    print()

    if args.verbose:
        print(response.trace.render(), file=sys.stderr)

    if args.trace_json:
        path = Path(args.trace_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.trace.to_json(), encoding="utf-8")
        print(f"[trace written to {path}]", file=sys.stderr)


def _repl(agent: TripMateAgent, args: argparse.Namespace) -> None:
    print("TripMate - ask about Tokyo, Reykjavik, Bangkok, or Barcelona.")
    print("Type 'exit' or Ctrl-D to quit.\n")
    while True:
        try:
            query = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if query.lower() in {"exit", "quit"}:
            print("Bye.")
            return
        if not query:
            continue
        _answer_once(agent, query, args)


def main(argv: list[str] | None = None) -> int:
    force_utf8_output()
    args = _build_parser().parse_args(argv)

    if args.provider:
        os.environ["TRIPMATE_PROVIDER"] = args.provider

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    level = "ERROR" if args.quiet_logs else config.log_level
    setup_logging(level=level, log_file=Path(args.log_file) if args.log_file else config.log_file)

    query = " ".join(args.query).strip()
    if not args.interactive and not query:
        _build_parser().print_help()
        return 1

    try:
        # Building the agent ingests the knowledge base and checks the API key,
        # so both failure modes surface here with a clear message.
        agent = TripMateAgent(config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except ProviderError as exc:
        print(f"LLM provider error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 -- startup must fail readably
        print(f"Failed to start TripMate: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.interactive:
        _repl(agent, args)
    else:
        _answer_once(agent, query, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
