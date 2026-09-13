"""Reasoning trace.

The orchestrator appends to a `Trace`; two renderers consume it. Same data,
two audiences: `to_dict()` feeds the structured JSON log, `render()` produces
the human-readable `--verbose` view. Nothing in the loop calls `print`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

StepKind = Literal["user_query", "llm_turn", "tool_call", "final_answer", "error"]


@dataclass
class TraceStep:
    kind: StepKind
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    query: str
    model: str
    steps: list[TraceStep] = field(default_factory=list)

    def add(self, kind: StepKind, **detail: Any) -> None:
        self.steps.append(TraceStep(kind=kind, detail=detail))

    # --- derived views ---------------------------------------------------

    @property
    def tools_used(self) -> list[str]:
        seen: list[str] = []
        for step in self.steps:
            if step.kind == "tool_call":
                name = step.detail.get("tool")
                if name and name not in seen:
                    seen.append(name)
        return seen

    @property
    def llm_turns(self) -> int:
        return sum(1 for s in self.steps if s.kind == "llm_turn")

    @property
    def total_tokens(self) -> dict[str, int]:
        totals = {"input": 0, "output": 0}
        for step in self.steps:
            usage = step.detail.get("usage")
            if usage:
                totals["input"] += usage.get("input_tokens", 0)
                totals["output"] += usage.get("output_tokens", 0)
        return totals

    # --- renderers -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "model": self.model,
            "llm_turns": self.llm_turns,
            "tools_used": self.tools_used,
            "usage": self.total_tokens,
            "steps": [{"kind": s.kind, **s.detail} for s in self.steps],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)

    def render(self) -> str:
        """Human-readable trace for the CLI's --verbose mode."""
        lines: list[str] = [
            "=" * 68,
            f"REASONING TRACE  ({self.model})",
            "=" * 68,
            f'  QUERY: "{self.query}"',
        ]

        for step in self.steps:
            d = step.detail

            if step.kind == "llm_turn":
                lines.append("")
                lines.append(f"  [{d.get('iteration')}] LLM TURN -> stop_reason={d.get('stop_reason')}")
                if d.get("reasoning"):
                    lines.append(f"      thinking aloud: {_truncate(d['reasoning'], 300)}")
                planned = d.get("planned_calls") or []
                if planned:
                    lines.append(f"      decided to call: {', '.join(planned)}")
                elif d.get("stop_reason") == "end_turn":
                    lines.append("      decided no (further) tool is needed")

            elif step.kind == "tool_call":
                status = "ERROR" if d.get("is_error") else "ok"
                lines.append("")
                lines.append(f"      TOOL {d.get('tool')}  [{status}, {d.get('duration_ms')}ms]")
                lines.append(f"        args   : {json.dumps(d.get('arguments', {}), ensure_ascii=False)}")
                lines.append(f"        result : {_truncate(d.get('result', ''), 400)}")

            elif step.kind == "error":
                lines.append("")
                lines.append(f"  ERROR [{d.get('stage')}]: {d.get('message')}")

            elif step.kind == "final_answer":
                usage = self.total_tokens
                lines.append("")
                lines.append("-" * 68)
                lines.append(
                    f"  SUMMARY: {self.llm_turns} LLM turn(s), "
                    f"tools used: {', '.join(self.tools_used) or 'none'}, "
                    f"tokens in/out: {usage['input']}/{usage['output']}"
                )

        lines.append("=" * 68)
        return "\n".join(lines)


def _truncate(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."
