"""The agent core.

A hand-written tool-use loop. Both SDKs ship a helper that would write this
loop for us; we do it by hand deliberately, because per-turn visibility into
*why* a tool was chosen is the point of this system, and because the loop is
the part worth reading.

The loop knows nothing about travel and nothing about any LLM vendor. It sees
a `LLMProvider`, a list of `ToolSpec`s, and a `dispatch` function, and would
run unchanged against a different provider or an entirely different tool set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tripmate.agent.prompts import SYSTEM_PROMPT
from tripmate.agent.trace import Trace
from tripmate.config import Config
from tripmate.llm.base import (
    AssistantTurn,
    LLMProvider,
    ToolResult,
    ToolResultsTurn,
    Turn,
    UserTurn,
)
from tripmate.logging_setup import get_logger
from tripmate.tools.registry import ToolRegistry, build_registry
from tripmate.utils import sanitize_query

log = get_logger(__name__)

MAX_QUERY_CHARS = 2000

# Shown when the provider itself fails. Never a fabricated answer.
_PROVIDER_FAILURE = (
    "I could not reach the language model just now, so I cannot answer that "
    "reliably. Please try again in a moment."
)


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    trace: Trace
    # The conversation to carry into the next `run()`. A caller that wants
    # one-shot behaviour simply ignores it, which is why `run()` still
    # defaults to a fresh conversation.
    history: list[Turn] = field(default_factory=list)

    @property
    def tools_used(self) -> list[str]:
        return self.trace.tools_used


class TripMateAgent:
    """Orchestrates: user query -> tool selection -> execution -> synthesis."""

    def __init__(
        self,
        config: Config,
        *,
        registry: ToolRegistry | None = None,
        provider: LLMProvider | None = None,
    ) -> None:
        self.config = config
        self.registry = registry or build_registry(config)
        if provider is None:
            from tripmate.llm.factory import build_provider

            provider = build_provider(config)
        self.provider = provider

    @staticmethod
    def _validate_query(query: object) -> str | None:
        """Return an error message, or None if the query is usable."""
        if query is None or not isinstance(query, str) or not query.strip():
            return "I did not receive a question. Please tell me what you would like to know."
        if len(query) > MAX_QUERY_CHARS:
            return (
                f"That question is very long ({len(query)} characters). "
                f"Please shorten it to under {MAX_QUERY_CHARS} characters."
            )
        return None

    def _trim(self, history: list[Turn]) -> list[Turn]:
        """Drop the oldest exchanges, cutting only at user-turn boundaries.

        An assistant turn carrying tool calls and the `ToolResultsTurn` that
        answers it are one indivisible unit: both wire formats reject a tool
        call with no matching result. Slicing at a `UserTurn` can never split
        that pair, so it is the only safe cut point.
        """
        limit = max(1, self.config.max_history_turns)
        starts = [i for i, turn in enumerate(history) if isinstance(turn, UserTurn)]
        if len(starts) <= limit:
            return list(history)
        return list(history[starts[-limit] :])

    def run(self, query: str, *, history: list[Turn] | None = None) -> AgentResponse:
        """Answer one question, optionally continuing a prior conversation.

        `history` is what a previous call returned as `AgentResponse.history`.
        Pass it to get follow-ups ("and what about Bangkok?"); omit it for a
        one-shot answer.
        """
        prior: list[Turn] = self._trim(list(history or []))

        query = sanitize_query(query)
        trace = Trace(
            query=query,
            model=f"{self.provider.name} · {self.provider.model}",
        )

        rejection = self._validate_query(query)
        if rejection:
            log.info("agent.rejected_input", extra={"reason": rejection})
            trace.add("error", stage="input_validation", message=rejection)
            trace.add("final_answer", answer=rejection)
            return AgentResponse(answer=rejection, trace=trace, history=prior)

        query = query.strip()
        trace.add("user_query", query=query)
        log.info(
            "agent.start",
            extra={
                "query": query,
                "provider": self.provider.name,
                "model": self.provider.model,
                "prior_turns": len(prior),
            },
        )

        history_turns: list[Turn] = [*prior, UserTurn(query)]
        tools = self.registry.specs()

        for iteration in range(1, self.config.max_iterations + 1):
            try:
                response = self.provider.complete(
                    system=SYSTEM_PROMPT, history=history_turns, tools=tools
                )
            except Exception as exc:  # noqa: BLE001 -- classified by the provider
                message = self.provider.describe_error(exc)
                log.error(
                    "agent.llm_call_failed",
                    extra={"iteration": iteration, "error": message},
                    exc_info=True,
                )
                trace.add("error", stage="llm_call", message=message)
                trace.add("final_answer", answer=_PROVIDER_FAILURE)
                return AgentResponse(answer=_PROVIDER_FAILURE, trace=trace, history=prior)

            trace.add(
                "llm_turn",
                iteration=iteration,
                stop_reason=response.stop_reason,
                reasoning=response.thinking,
                planned_calls=[c.name for c in response.tool_calls],
                usage=response.usage,
            )
            log.info(
                "agent.llm_turn",
                extra={
                    "iteration": iteration,
                    "stop_reason": response.stop_reason,
                    "planned_calls": [
                        {"tool": c.name, "arguments": c.arguments} for c in response.tool_calls
                    ],
                },
            )

            if response.stop_reason == "refusal":
                answer = (
                    "I am not able to help with that request. If it was travel-related, "
                    "try rephrasing it and I will do my best."
                )
                trace.add("final_answer", answer=answer)
                return AgentResponse(answer=answer, trace=trace, history=prior)

            if not response.tool_calls:
                # No tool needed, or the model is done: this is the answer.
                answer = response.text or (
                    "I was not able to produce an answer for that. Could you rephrase it?"
                )
                if response.stop_reason == "max_tokens":
                    answer += "\n\n(This answer was cut short by a length limit.)"
                trace.add("final_answer", answer=answer, stop_reason=response.stop_reason)
                log.info(
                    "agent.done",
                    extra={"iterations": iteration, "tools_used": trace.tools_used},
                )
                # Only a turn that closed with real assistant text is safe to
                # replay: an empty assistant message is rejected by the wire
                # formats, and we would be feeding back our own placeholder.
                if response.text:
                    history_turns.append(AssistantTurn(response))
                    return AgentResponse(answer=answer, trace=trace, history=history_turns)
                return AgentResponse(answer=answer, trace=trace, history=prior)

            history_turns.append(AssistantTurn(response))

            results: list[ToolResult] = []
            for call in response.tool_calls:
                outcome = self.registry.dispatch(call.name, call.arguments)
                trace.add(
                    "tool_call",
                    tool=call.name,
                    arguments=call.arguments,
                    result=outcome.content,
                    is_error=outcome.is_error,
                    duration_ms=outcome.duration_ms,
                )
                results.append(
                    ToolResult(
                        id=call.id,
                        name=call.name,
                        content=outcome.content,
                        is_error=outcome.is_error,
                    )
                )

            # One turn carrying every result. How that reaches the wire is the
            # provider's business -- Anthropic batches them into a single user
            # message, OpenAI-shaped APIs emit one `tool` message each.
            history_turns.append(ToolResultsTurn(results))

        # Loop guard tripped: the model kept requesting tools without settling.
        answer = (
            "I gathered information but could not settle on a final answer. "
            "Could you try asking about one destination or topic at a time?"
        )
        log.warning("agent.max_iterations", extra={"max_iterations": self.config.max_iterations})
        trace.add(
            "error", stage="loop_guard", message=f"Hit max_iterations={self.config.max_iterations}"
        )
        trace.add("final_answer", answer=answer)
        # The loop ended on an unanswered tool call, so this exchange cannot be
        # replayed. Hand back the conversation as it stood before it.
        return AgentResponse(answer=answer, trace=trace, history=prior)
