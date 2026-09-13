"""Groq provider (OpenAI-compatible chat completions).

Default provider: the free tier needs no credit card and offers models with
native tool calling, which is what the multi-tool packing flow depends on.

Groq rotates its catalogue often -- `llama-3.3-70b-versatile` had already been
retired by the time this was written -- so treat DEFAULT_MODEL as a starting
point and override it with TRIPMATE_MODEL if you get a 404 model_not_found.
"""

from __future__ import annotations

import json
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
from tripmate.logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-120b"

_STOP_REASONS: dict[str, StopReason] = {
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "refusal",
}


class GroqProvider:
    def __init__(self, api_key: str, model: str, timeout: float) -> None:
        from groq import Groq  # imported lazily so the dep stays optional

        self.name = "groq"
        self.model = model
        self._client = Groq(api_key=api_key, timeout=timeout)

    # --- request shaping --------------------------------------------------

    @staticmethod
    def _tools_to_wire(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

    @staticmethod
    def _history_to_wire(system: str, history: list[Turn]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

        for turn in history:
            if isinstance(turn, UserTurn):
                messages.append({"role": "user", "content": turn.text})

            elif isinstance(turn, AssistantTurn):
                message: dict[str, Any] = {
                    "role": "assistant",
                    "content": turn.response.text or None,
                }
                if turn.response.tool_calls:
                    message["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in turn.response.tool_calls
                    ]
                messages.append(message)

            elif isinstance(turn, ToolResultsTurn):
                # OpenAI-shaped APIs want one `tool` message per result, each
                # keyed to its call id -- unlike Anthropic, which batches them
                # into a single user message.
                for result in turn.results:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": result.id,
                            "name": result.name,
                            "content": result.content,
                        }
                    )

        return messages

    # --- the call ---------------------------------------------------------

    def complete(
        self, *, system: str, history: list[Turn], tools: list[ToolSpec]
    ) -> LLMResponse:
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=self._history_to_wire(system, history),
            tools=self._tools_to_wire(tools),
            tool_choice="auto",
            parallel_tool_calls=True,  # required for the two-tool packing flow
            max_tokens=4096,
            temperature=0.2,
        )
        return self._parse(completion)

    @staticmethod
    def _parse(completion: Any) -> LLMResponse:
        choice = completion.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        for raw_call in message.tool_calls or []:
            try:
                arguments = json.loads(raw_call.function.arguments or "{}")
            except json.JSONDecodeError:
                # A model can emit malformed JSON. Surface it as an empty call
                # so the registry rejects it with a message the model can act
                # on, rather than crashing the loop.
                log.warning(
                    "groq.tool_arguments_unparseable",
                    extra={"tool": raw_call.function.name, "raw": raw_call.function.arguments},
                )
                arguments = {}
            tool_calls.append(
                ToolCall(id=raw_call.id, name=raw_call.function.name, arguments=arguments)
            )

        usage = getattr(completion, "usage", None)
        return LLMResponse(
            text=message.content or "",
            # Some Groq models expose a reasoning trace; most do not.
            thinking=getattr(message, "reasoning", None) or "",
            tool_calls=tool_calls,
            stop_reason=_STOP_REASONS.get(choice.finish_reason or "", "other"),
            usage={
                "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
            },
            raw=message,
        )

    # --- errors -----------------------------------------------------------

    def describe_error(self, exc: Exception) -> str:
        import groq

        if isinstance(exc, groq.AuthenticationError):
            return "Authentication failed: check GROQ_API_KEY."
        if isinstance(exc, groq.NotFoundError):
            return f"Model '{self.model}' not found on Groq: {exc}"
        if isinstance(exc, groq.RateLimitError):
            return "Rate limited by Groq's free tier (retryable)."
        if isinstance(exc, groq.APITimeoutError):
            return f"Groq request timed out (retryable): {exc}"
        if isinstance(exc, groq.APIStatusError):
            return f"Groq returned HTTP {exc.status_code}: {exc}"
        if isinstance(exc, groq.APIConnectionError):
            return f"Could not connect to Groq (retryable): {exc}"
        return f"Unexpected {type(exc).__name__}: {exc}"
