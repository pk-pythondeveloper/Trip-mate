"""Provider-neutral LLM types.

The orchestrator is written against these types alone, so it never learns
which vendor is behind it. Each provider translates this neutral conversation
into its own wire format and normalises the reply back.

This matters because the two wire formats genuinely differ. Anthropic carries
tool calls as `tool_use` content blocks and returns their results as
`tool_result` blocks inside one *user* message; OpenAI-compatible APIs (Groq)
carry them as `tool_calls` on the assistant message and return each result as
its own `role: "tool"` message. Keeping that difference inside the providers is
what makes swapping vendors a one-file change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

StopReason = Literal["tool_use", "end_turn", "max_tokens", "refusal", "other"]


@dataclass(frozen=True)
class ToolCall:
    """A tool the model asked to run."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """The outcome of running one tool, on its way back to the model."""

    id: str  # must match the originating ToolCall.id
    name: str
    content: str
    is_error: bool


@dataclass(frozen=True)
class LLMResponse:
    """One normalised assistant turn."""

    text: str
    thinking: str
    tool_calls: list[ToolCall]
    stop_reason: StopReason
    usage: dict[str, int] = field(default_factory=dict)
    # The provider-native assistant content, kept verbatim so it can be
    # replayed exactly. Anthropic requires this: thinking blocks and tool_use
    # ids must survive unchanged into the next request.
    raw: Any = None


# --- neutral conversation history ---------------------------------------


@dataclass(frozen=True)
class UserTurn:
    text: str


@dataclass(frozen=True)
class AssistantTurn:
    response: LLMResponse


@dataclass(frozen=True)
class ToolResultsTurn:
    results: list[ToolResult]


Turn = UserTurn | AssistantTurn | ToolResultsTurn


@dataclass(frozen=True)
class ToolSpec:
    """A tool as the registry describes it, before provider-specific shaping."""

    name: str
    description: str
    input_schema: dict[str, Any]


class LLMProvider(Protocol):
    """What the orchestrator requires of any LLM vendor."""

    name: str
    model: str

    def complete(
        self, *, system: str, history: list[Turn], tools: list[ToolSpec]
    ) -> LLMResponse: ...

    def describe_error(self, exc: Exception) -> str:
        """Classify a provider exception into a readable, actionable message."""
        ...


class ProviderError(RuntimeError):
    """Raised when a provider cannot be constructed (missing key, bad name)."""
