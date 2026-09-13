"""The tool registry -- the boundary between orchestration and business logic.

The orchestrator sees exactly two things through this module: `schemas()`, the
list it hands the LLM, and `dispatch()`, which runs one tool by name. It never
learns what a tool does. Adding a third tool therefore requires no change to
the agent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from tripmate.config import Config
from tripmate.forecast.service import WeatherService
from tripmate.llm.base import ToolSpec
from tripmate.logging_setup import get_logger
from tripmate.rag.retriever import DestinationRetriever
from tripmate.tools import guide_tool, weather_tool
from tripmate.tools.errors import ToolError, ToolInputError, ToolNotFoundError

log = get_logger(__name__)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., str]

    def to_schema(self) -> dict[str, Any]:
        """Provider-neutral schema dict, handy for tests and documentation."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def to_spec(self) -> ToolSpec:
        """The form the LLM layer consumes; each provider shapes it to its wire format."""
        return ToolSpec(
            name=self.name, description=self.description, input_schema=self.input_schema
        )


@dataclass(frozen=True)
class ToolOutcome:
    """Provider-neutral result of one tool call."""

    name: str
    content: str
    is_error: bool
    duration_ms: float


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.to_schema() for tool in self._tools.values()]

    def specs(self) -> list[ToolSpec]:
        """What the orchestrator hands to the LLM provider."""
        return [tool.to_spec() for tool in self._tools.values()]

    def _validate(self, tool: Tool, arguments: dict[str, Any]) -> None:
        """Validate arguments against tool schema before execution."""
        if not isinstance(arguments, dict):
            raise ToolInputError(f"Arguments for '{tool.name}' must be an object.")

        properties = tool.input_schema.get("properties", {})
        missing = [k for k in tool.input_schema.get("required", []) if k not in arguments]
        if missing:
            raise ToolInputError(
                f"Missing required argument(s) for '{tool.name}': {', '.join(missing)}."
            )
        unexpected = [k for k in arguments if k not in properties]
        if unexpected:
            raise ToolInputError(
                f"Unexpected argument(s) for '{tool.name}': {', '.join(unexpected)}. "
                f"Accepted: {', '.join(properties)}."
            )

    def dispatch(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        """Run one tool. Never raises -- failures come back as error outcomes."""
        started = time.perf_counter()

        def elapsed() -> float:
            return round((time.perf_counter() - started) * 1000, 2)

        try:
            tool = self._tools.get(name)
            if tool is None:
                raise ToolNotFoundError(
                    f"No tool named '{name}'. Available tools: {', '.join(self._tools)}."
                )
            self._validate(tool, arguments)
            content = tool.handler(**arguments)
            outcome = ToolOutcome(name=name, content=content, is_error=False, duration_ms=elapsed())
            log.info(
                "tool.call",
                extra={
                    "tool": name,
                    "arguments": arguments,
                    "status": "ok",
                    "duration_ms": outcome.duration_ms,
                    "result_preview": content[:400],
                },
            )
            return outcome

        except ToolError as exc:
            message = f"{exc.code}: {exc}"
            log.warning(
                "tool.call",
                extra={"tool": name, "arguments": arguments, "status": "error", "error": message},
            )
            return ToolOutcome(name=name, content=message, is_error=True, duration_ms=elapsed())

        except Exception as exc:  # noqa: BLE001 -- deliberate loop-safety net
            # Anything unforeseen (a provider timeout, a bad file, a bug) is
            # reported to the model rather than crashing the conversation.
            message = f"execution_failed: {type(exc).__name__}: {exc}"
            log.error(
                "tool.call",
                extra={"tool": name, "arguments": arguments, "status": "exception"},
                exc_info=True,
            )
            return ToolOutcome(name=name, content=message, is_error=True, duration_ms=elapsed())


def build_registry(
    config: Config,
    *,
    retriever: DestinationRetriever | None = None,
    weather_service: WeatherService | None = None,
) -> ToolRegistry:
    """Wire the tools to their business logic.

    Dependencies are injectable so tests can substitute fakes without touching
    the filesystem or downloading a model.
    """
    retriever = retriever or DestinationRetriever(
        data_dir=config.data_dir,
        embedding_model=config.embedding_model,
        min_similarity=config.min_similarity,
    )
    weather_service = weather_service or WeatherService()

    return ToolRegistry(
        [
            Tool(
                name=guide_tool.NAME,
                description=guide_tool.DESCRIPTION,
                input_schema=guide_tool.INPUT_SCHEMA,
                handler=guide_tool.make_handler(retriever, config.top_k),
            ),
            Tool(
                name=weather_tool.NAME,
                description=weather_tool.DESCRIPTION,
                input_schema=weather_tool.INPUT_SCHEMA,
                handler=weather_tool.make_handler(weather_service),
            ),
        ]
    )
