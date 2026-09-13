"""Provider wire-format translation.

The orchestrator speaks one neutral conversation format; each provider must
render it into its vendor's shape. These are pure translation tests -- no
network, no client construction.
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import llm_response, tool_call
from tripmate.config import Config
from tripmate.llm.anthropic_provider import AnthropicProvider
from tripmate.llm.base import (
    AssistantTurn,
    ProviderError,
    ToolResult,
    ToolResultsTurn,
    ToolSpec,
    UserTurn,
)
from tripmate.llm.factory import build_provider
from tripmate.llm.groq_provider import GroqProvider

TOOLS = [
    ToolSpec(
        name="get_weather_forecast",
        description="Get typical weather.",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    )
]


def multi_tool_history():
    """user -> assistant(2 parallel calls) -> both results."""
    assistant = llm_response(
        thinking="Needs both.",
        tool_calls=[
            tool_call("search_destination_guide", {"query": "Reykjavik packing"}, "tc_1"),
            tool_call("get_weather_forecast", {"city": "Reykjavik", "date_or_month": "December"}, "tc_2"),
        ],
    )
    return [
        UserTurn("What should I pack for Reykjavik in December?"),
        AssistantTurn(assistant),
        ToolResultsTurn(
            [
                ToolResult(id="tc_1", name="search_destination_guide", content='{"results": []}', is_error=False),
                ToolResult(id="tc_2", name="get_weather_forecast", content='{"available": true}', is_error=False),
            ]
        ),
    ]


# --- Groq / OpenAI-compatible shape --------------------------------------


def test_groq_tools_use_the_function_wrapper():
    wire = GroqProvider._tools_to_wire(TOOLS)
    assert wire[0]["type"] == "function"
    assert wire[0]["function"]["name"] == "get_weather_forecast"
    # OpenAI calls it `parameters`, not `input_schema`.
    assert wire[0]["function"]["parameters"] == TOOLS[0].input_schema


def test_groq_emits_one_tool_message_per_result():
    wire = GroqProvider._history_to_wire("SYSTEM", multi_tool_history())
    roles = [m["role"] for m in wire]
    assert roles == ["system", "user", "assistant", "tool", "tool"]

    assistant = wire[2]
    assert [c["id"] for c in assistant["tool_calls"]] == ["tc_1", "tc_2"]
    # Arguments are a JSON *string* on the wire, not an object.
    assert json.loads(assistant["tool_calls"][1]["function"]["arguments"])["city"] == "Reykjavik"

    assert [m["tool_call_id"] for m in wire[3:]] == ["tc_1", "tc_2"]


def test_groq_puts_the_system_prompt_in_the_messages_array():
    wire = GroqProvider._history_to_wire("SYSTEM", [UserTurn("hi")])
    assert wire[0] == {"role": "system", "content": "SYSTEM"}


@pytest.mark.parametrize(
    "finish_reason,expected",
    [("tool_calls", "tool_use"), ("stop", "end_turn"), ("length", "max_tokens"), ("weird", "other")],
)
def test_groq_finish_reasons_are_normalised(finish_reason, expected):
    from types import SimpleNamespace

    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content="hi", tool_calls=None, reasoning=None),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    parsed = GroqProvider._parse(completion)
    assert parsed.stop_reason == expected
    assert parsed.usage == {"input_tokens": 10, "output_tokens": 5}


def test_groq_survives_malformed_tool_arguments():
    """A model can emit invalid JSON; that must not crash the loop."""
    from types import SimpleNamespace

    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    reasoning=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="tc_1",
                            function=SimpleNamespace(name="get_weather_forecast", arguments="{not json"),
                        )
                    ],
                ),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )
    parsed = GroqProvider._parse(completion)
    # Empty args reach the registry, which rejects them with a usable message.
    assert parsed.tool_calls[0].arguments == {}


# --- Anthropic shape ------------------------------------------------------


def test_anthropic_tools_keep_input_schema():
    wire = AnthropicProvider._tools_to_wire(TOOLS)
    assert set(wire[0]) == {"name", "description", "input_schema"}


def test_anthropic_batches_all_results_into_one_user_message():
    """Splitting them would train the model out of parallel tool calls."""
    wire = AnthropicProvider._history_to_wire(multi_tool_history())
    assert [m["role"] for m in wire] == ["user", "assistant", "user"]

    results = wire[2]["content"]
    assert len(results) == 2, "both tool_results must ride in ONE user message"
    assert [r["tool_use_id"] for r in results] == ["tc_1", "tc_2"]
    assert all(r["type"] == "tool_result" for r in results)


def test_anthropic_replays_assistant_content_verbatim():
    """Thinking blocks and tool_use ids must survive unchanged."""
    native_blocks = [{"type": "thinking", "thinking": "..."}, {"type": "tool_use", "id": "tc_1"}]
    response = llm_response(tool_calls=[tool_call("x", {}, "tc_1")])
    response = type(response)(**{**response.__dict__, "raw": native_blocks})

    wire = AnthropicProvider._history_to_wire([UserTurn("hi"), AssistantTurn(response)])
    assert wire[1]["content"] is native_blocks


def test_anthropic_parses_text_thinking_and_tool_use_blocks():
    from types import SimpleNamespace

    response = SimpleNamespace(
        stop_reason="tool_use",
        content=[
            SimpleNamespace(type="thinking", thinking="I should check the guide."),
            SimpleNamespace(type="text", text="Let me look."),
            SimpleNamespace(type="tool_use", id="tc_1", name="search_destination_guide", input={"query": "Tokyo"}),
        ],
        usage=SimpleNamespace(input_tokens=20, output_tokens=8),
    )
    parsed = AnthropicProvider._parse(response)
    assert parsed.thinking == "I should check the guide."
    assert parsed.text == "Let me look."
    assert parsed.tool_calls[0].arguments == {"query": "Tokyo"}
    assert parsed.stop_reason == "tool_use"


# --- the factory ----------------------------------------------------------


def test_factory_rejects_an_unknown_provider():
    config = Config(provider="hal9000", api_key="k", model="m")
    with pytest.raises(ProviderError, match="Unknown provider"):
        build_provider(config)


def test_factory_requires_a_key_naming_the_right_env_var():
    from tripmate.config import ConfigError

    config = Config(provider="groq", api_key="", model="m")
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        build_provider(config)

    config = Config(provider="anthropic", api_key="", model="m")
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        build_provider(config)


def test_config_picks_the_default_model_per_provider(monkeypatch):
    monkeypatch.delenv("TRIPMATE_MODEL", raising=False)
    assert Config(provider="groq", api_key="k").model == "openai/gpt-oss-120b"
    assert Config(provider="anthropic", api_key="k").model == "claude-opus-5"


def test_config_lets_the_model_be_overridden(monkeypatch):
    monkeypatch.setenv("TRIPMATE_MODEL", "some-other-model")
    assert Config(provider="groq", api_key="k").model == "some-other-model"
