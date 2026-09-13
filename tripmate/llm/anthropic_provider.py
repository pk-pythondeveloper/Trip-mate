"""Anthropic provider (Messages API).

Kept alongside Groq to prove the abstraction: same orchestrator, same tools,
same tests, different vendor. Select it with TRIPMATE_PROVIDER=anthropic.
"""

from __future__ import annotations

from typing import Any

from tripmate.llm.base import (
    AssistantTurn,
    LLMResponse,
    StopReason,
    ToolCall,
    ToolResultsTurn,
    ToolSpec,
    Turn,
    UserTurn,
)

DEFAULT_MODEL = "claude-opus-5"

_STOP_REASONS: dict[str, StopReason] = {
    "tool_use": "tool_use",
    "end_turn": "end_turn",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
    "stop_sequence": "end_turn",
}


class AnthropicProvider:
    def __init__(self, api_key: str, model: str, timeout: float, max_tokens: int) -> None:
        import anthropic  # imported lazily so the dep stays optional

        self.name = "anthropic"
        self.model = model
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)

    # --- request shaping --------------------------------------------------

    @staticmethod
    def _tools_to_wire(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]

    @staticmethod
    def _history_to_wire(history: list[Turn]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []

        for turn in history:
            if isinstance(turn, UserTurn):
                messages.append({"role": "user", "content": turn.text})

            elif isinstance(turn, AssistantTurn):
                # Replay the native content blocks verbatim: thinking blocks
                # and tool_use ids must survive unchanged.
                messages.append({"role": "assistant", "content": turn.response.raw})

            elif isinstance(turn, ToolResultsTurn):
                # ALL results go back in ONE user message. Splitting them
                # across several messages silently teaches the model to stop
                # issuing parallel tool calls, which would turn the two-tool
                # packing flow into two round trips.
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": result.id,
                                "content": result.content,
                                "is_error": result.is_error,
                            }
                            for result in turn.results
                        ],
                    }
                )

        return messages

    # --- the call ---------------------------------------------------------

    def complete(
        self, *, system: str, history: list[Turn], tools: list[ToolSpec]
    ) -> LLMResponse:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            system=system,
            tools=self._tools_to_wire(tools),
            # Adaptive thinking makes the reasoning in the trace the model's
            # own, rather than something reconstructed after the fact.
            thinking={"type": "adaptive", "display": "summarized"},
            messages=self._history_to_wire(history),
        )
        return self._parse(response)

    @staticmethod
    def _parse(response: Any) -> LLMResponse:
        text: list[str] = []
        thinking: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            kind = getattr(block, "type", None)
            if kind == "text":
                text.append(block.text)
            elif kind == "thinking":
                thinking.append(getattr(block, "thinking", "") or "")
            elif kind == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))

        usage = getattr(response, "usage", None)
        return LLMResponse(
            text="\n".join(t for t in text if t).strip(),
            thinking=" ".join(t for t in thinking if t).strip(),
            tool_calls=tool_calls,
            stop_reason=_STOP_REASONS.get(response.stop_reason or "", "other"),
            usage={
                "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                "output_tokens": getattr(usage, "output_tokens", 0) or 0,
            },
            raw=response.content,
        )

    # --- errors -----------------------------------------------------------

    def describe_error(self, exc: Exception) -> str:
        import anthropic

        if isinstance(exc, anthropic.AuthenticationError):
            return "Authentication failed: check ANTHROPIC_API_KEY."
        if isinstance(exc, anthropic.NotFoundError):
            return f"Model '{self.model}' not found: {exc}"
        if isinstance(exc, anthropic.RateLimitError):
            return "Rate limited by Anthropic (retryable)."
        if isinstance(exc, anthropic.APITimeoutError):
            return f"Anthropic request timed out (retryable): {exc}"
        if isinstance(exc, anthropic.APIStatusError):
            return f"Anthropic returned HTTP {exc.status_code}: {exc}"
        if isinstance(exc, anthropic.APIConnectionError):
            return f"Could not connect to Anthropic (retryable): {exc}"
        return f"Unexpected {type(exc).__name__}: {exc}"
