"""Regenerate the example trace logs in `examples/`.

    python scripts/generate_traces.py

Runs each demo query against the real agent and writes, per query:

    examples/<slug>.json   full structured trace
    examples/<slug>.txt    the rendered human-readable trace + final answer

Requires ANTHROPIC_API_KEY. These are the traces referenced by the README, so
re-run this after changing prompts or tool descriptions.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tripmate.agent.orchestrator import TripMateAgent  # noqa: E402
from tripmate.config import ConfigError, load_config  # noqa: E402
from tripmate.llm.base import ProviderError  # noqa: E402
from tripmate.logging_setup import force_utf8_output, setup_logging  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "examples"

# One query per capability the assessment asks to see demonstrated.
DEMO_QUERIES: list[tuple[str, str]] = [
    ("single-tool-rag", "Do I need a visa to visit Japan, and what are the local customs I should know?"),
    ("single-tool-weather", "How hot does Bangkok get in April?"),
    ("multi-tool-packing", "What should I pack for Reykjavik in December?"),
    ("multi-tool-best-time", "Is July a good time to visit Barcelona?"),
    ("out-of-scope-booking", "Can you book me a flight to Tokyo next Tuesday?"),
    ("unknown-destination", "What are the visa requirements for Cairo?"),
    ("ambiguous-query", "What should I pack?"),
]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def main() -> int:
    force_utf8_output()
    setup_logging("INFO", log_file=OUTPUT_DIR / "tripmate.log")

    try:
        config = load_config()
        agent = TripMateAgent(config)
    except (ConfigError, ProviderError) as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, query in DEMO_QUERIES:
        print(f"\n=== {name}: {query}")
        response = agent.run(query)

        (OUTPUT_DIR / f"{name}.json").write_text(response.trace.to_json(), encoding="utf-8")
        (OUTPUT_DIR / f"{name}.txt").write_text(
            f"QUERY: {query}\n\n"
            f"{response.trace.render()}\n\n"
            f"FINAL ANSWER\n{'-' * 68}\n{response.answer}\n",
            encoding="utf-8",
        )

        print(f"tools: {response.tools_used or 'none'}")
        print(response.answer)

    print(f"\nWrote {len(DEMO_QUERIES) * 2} files to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
