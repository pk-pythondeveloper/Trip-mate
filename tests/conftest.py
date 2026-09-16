"""Shared fixtures.

The key piece is `ScriptedProvider`, a stand-in for a real LLM vendor that
replays a fixed list of responses and records every call it received. Because
the orchestrator is written against the provider-neutral `LLMProvider`
interface, these tests are vendor-agnostic: they exercise the same loop that
runs against Groq or Anthropic, with no API key, no network, and no flakiness.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from tripmate.config import Config
from tripmate.forecast.service import WeatherService
from tripmate.llm.base import LLMResponse, StopReason, ToolCall, ToolSpec, Turn
from tripmate.logging_setup import setup_logging
from tripmate.rag.retriever import DestinationRetriever
from tripmate.tools.registry import build_registry

setup_logging("CRITICAL")  # keep test output clean


# --- response builders --------------------------------------------------


def tool_call(name: str, arguments: dict[str, Any], call_id: str = "tc_1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def llm_response(
    text: str = "",
    *,
    thinking: str = "",
    tool_calls: list[ToolCall] | None = None,
    stop_reason: StopReason | None = None,
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> LLMResponse:
    calls = tool_calls or []
    return LLMResponse(
        text=text,
        thinking=thinking,
        tool_calls=calls,
        stop_reason=stop_reason or ("tool_use" if calls else "end_turn"),
        usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
        raw=None,
    )


class ScriptedProvider:
    """Replays queued responses; records the calls it received."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.name = "scripted"
        self.model = "test-model"

    def complete(
        self, *, system: str, history: list[Turn], tools: list[ToolSpec]
    ) -> LLMResponse:
        # Snapshot the history: the orchestrator mutates the same list, so
        # storing a reference would let later turns rewrite earlier records.
        self.calls.append({"system": system, "history": copy.copy(history), "tools": tools})
        if not self._responses:
            raise AssertionError("ScriptedProvider ran out of scripted responses")
        return self._responses.pop(0)

    def describe_error(self, exc: Exception) -> str:
        return f"scripted: {type(exc).__name__}: {exc}"


class FailingProvider:
    """Raises a supplied exception on every call, to test provider failure."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.name = "failing"
        self.model = "test-model"
        self.calls: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> LLMResponse:
        raise self._exc

    def describe_error(self, exc: Exception) -> str:
        return f"Could not connect to the provider: {exc}"


# --- fixtures -----------------------------------------------------------


@pytest.fixture(scope="session")
def config() -> Config:
    # A dummy key: no test in the default suite may reach the network.
    return Config(provider="groq", api_key="test-key-not-used", model="test-model")


@pytest.fixture(scope="session")
def retriever(config: Config) -> DestinationRetriever:
    # prefer_semantic=False pins the deterministic lexical embedder, so the
    # suite is fast, offline, and gives identical scores on every machine.
    return DestinationRetriever(
        data_dir=config.data_dir,
        embedding_model=config.embedding_model,
        prefer_semantic=False,
    )


@pytest.fixture(scope="session")
def weather_service() -> WeatherService:
    return WeatherService()


@pytest.fixture
def registry(config, retriever, weather_service):
    return build_registry(config, retriever=retriever, weather_service=weather_service)
