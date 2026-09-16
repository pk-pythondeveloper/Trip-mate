"""Module 1/4/5 tests: orchestration, multi-tool flow, and scope awareness.

These drive a `ScriptedProvider`, so they verify the *orchestrator's* handling
of each path -- single-tool, multi-tool, no-tool, and every failure mode -- with
no network and no flakiness. They are provider-agnostic by construction.

They do not verify the model's own judgement; that needs a live call and lives
in `test_live_selection.py`.
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import FailingProvider, ScriptedProvider, llm_response, tool_call
from tripmate.agent.orchestrator import TripMateAgent
from tripmate.config import Config
from tripmate.llm.base import AssistantTurn, ToolResultsTurn, UserTurn


def build_agent(config, registry, provider) -> TripMateAgent:
    return TripMateAgent(config, registry=registry, provider=provider)


# --- single-tool ---------------------------------------------------------


def test_single_tool_flow_calls_only_the_rag_tool(config, registry):
    provider = ScriptedProvider(
        [
            llm_response(
                thinking="A visa question -- the guide covers this; no weather needed.",
                tool_calls=[tool_call("search_destination_guide", {"query": "Japan visa"}, "tc_a")],
            ),
            llm_response("Most visitors can enter Japan visa-free for up to 90 days."),
        ]
    )
    response = build_agent(config, registry, provider).run("Do I need a visa for Japan?")

    assert response.tools_used == ["search_destination_guide"]
    assert "visa-free" in response.answer
    assert response.trace.llm_turns == 2


def test_the_tool_result_reaches_the_second_call(config, registry):
    """The retrieved text must actually be fed back to the model."""
    provider = ScriptedProvider(
        [
            llm_response(
                tool_calls=[tool_call("search_destination_guide", {"query": "Japan visa"}, "tc_a")]
            ),
            llm_response("Answer."),
        ]
    )
    build_agent(config, registry, provider).run("Do I need a visa for Japan?")

    history = provider.calls[1]["history"]
    assert isinstance(history[0], UserTurn)
    assert isinstance(history[1], AssistantTurn)

    results_turn = history[2]
    assert isinstance(results_turn, ToolResultsTurn)
    result = results_turn.results[0]
    assert result.id == "tc_a"
    assert result.is_error is False
    assert "visa-free" in json.loads(result.content)["results"][0]["text"]


def test_the_system_prompt_and_tools_are_sent_on_every_call(config, registry):
    provider = ScriptedProvider(
        [
            llm_response(tool_calls=[tool_call("search_destination_guide", {"query": "Tokyo"})]),
            llm_response("Answer."),
        ]
    )
    build_agent(config, registry, provider).run("Tell me about Tokyo")

    for call in provider.calls:
        assert "TripMate" in call["system"]
        assert {t.name for t in call["tools"]} == {
            "search_destination_guide",
            "get_weather_forecast",
        }


# --- multi-tool (Module 4) ----------------------------------------------


def test_packing_question_runs_both_tools_and_synthesises_one_answer(config, registry):
    provider = ScriptedProvider(
        [
            llm_response(
                thinking="Packing needs the guide's advice AND the real December conditions.",
                tool_calls=[
                    tool_call("search_destination_guide", {"query": "Reykjavik packing"}, "tc_1"),
                    tool_call(
                        "get_weather_forecast",
                        {"city": "Reykjavik", "date_or_month": "December"},
                        "tc_2",
                    ),
                ],
            ),
            llm_response(
                "Pack waterproof and windproof outer layers, warm mid-layers, a hat and "
                "gloves: December in Reykjavik runs about -2 to 3C with high precipitation."
            ),
        ]
    )
    response = build_agent(config, registry, provider).run(
        "What should I pack for Reykjavik in December?"
    )

    assert set(response.tools_used) == {"search_destination_guide", "get_weather_forecast"}
    assert response.trace.llm_turns == 2, "both tools should resolve in a single round trip"


def test_parallel_tool_results_travel_as_one_turn(config, registry):
    """One ToolResultsTurn carries every result; providers shape it to the wire."""
    provider = ScriptedProvider(
        [
            llm_response(
                tool_calls=[
                    tool_call("search_destination_guide", {"query": "Reykjavik packing"}, "tc_1"),
                    tool_call(
                        "get_weather_forecast",
                        {"city": "Reykjavik", "date_or_month": "December"},
                        "tc_2",
                    ),
                ]
            ),
            llm_response("Answer."),
        ]
    )
    build_agent(config, registry, provider).run("What should I pack for Reykjavik in December?")

    history = provider.calls[1]["history"]
    assert len(history) == 3, "user turn, assistant turn, one combined results turn"
    results_turn = history[2]
    assert isinstance(results_turn, ToolResultsTurn)
    assert [r.id for r in results_turn.results] == ["tc_1", "tc_2"]

    # Both tools actually ran and returned usable payloads.
    guide, weather = results_turn.results
    assert json.loads(guide.content)["results"][0]["city"] == "Reykjavik"
    assert json.loads(weather.content)["temp_range_c"] == [-2, 3]


# --- no-tool (Module 5) --------------------------------------------------


def test_out_of_scope_request_uses_no_tools(config, registry):
    provider = ScriptedProvider(
        [
            llm_response(
                "I can't book flights -- I have no booking ability. I can help you plan "
                "when to go and what to pack."
            )
        ]
    )
    response = build_agent(config, registry, provider).run("Can you book my flight to Tokyo?")

    assert response.tools_used == []
    assert response.trace.llm_turns == 1
    assert "book" in response.answer.lower()


def test_the_system_prompt_forbids_fabricated_actions(config, registry):
    """Module 5 is a prompt concern, so assert the prompt actually carries it."""
    from tripmate.agent.prompts import SYSTEM_PROMPT

    # Collapse wrapping so the assertions match phrases, not line breaks.
    prompt = " ".join(SYSTEM_PROMPT.lower().split())
    assert "cannot book" in prompt
    assert "cannot take actions in the world" in prompt
    assert "never imply an action has been taken" in prompt
    for city in ("tokyo", "reykjavik", "bangkok", "barcelona"):
        assert city in prompt


# --- input validation ----------------------------------------------------


@pytest.mark.parametrize("bad_query", ["", "   ", None])
def test_empty_input_is_rejected_before_any_api_call(config, registry, bad_query):
    provider = ScriptedProvider([])  # any call would raise
    response = build_agent(config, registry, provider).run(bad_query)
    assert "did not receive a question" in response.answer
    assert response.tools_used == []
    assert provider.calls == []


def test_an_over_long_query_is_rejected(config, registry):
    provider = ScriptedProvider([])
    response = build_agent(config, registry, provider).run("a" * 5000)
    assert "shorten" in response.answer
    assert provider.calls == []


# --- failure handling ----------------------------------------------------


def test_provider_failure_yields_a_graceful_message_not_a_crash(config, registry):
    provider = FailingProvider(ConnectionError("network unreachable"))
    response = build_agent(config, registry, provider).run("Do I need a visa for Japan?")
    assert "could not reach the language model" in response.answer.lower()
    assert any(s.kind == "error" for s in response.trace.steps)


def test_a_failing_tool_is_reported_to_the_model_and_the_loop_continues(config, registry):
    """An erroring tool must produce an is_error result, not end the run."""
    provider = ScriptedProvider(
        [
            llm_response(
                tool_calls=[tool_call("get_weather_forecast", {"city": "Tokyo"}, "tc_1")]  # no month
            ),
            llm_response("Which month were you thinking of?"),
        ]
    )
    response = build_agent(config, registry, provider).run("What's the weather in Tokyo?")

    result = provider.calls[1]["history"][2].results[0]
    assert result.is_error is True
    assert "date_or_month" in result.content
    assert "which month" in response.answer.lower()


def test_an_unknown_tool_name_does_not_break_the_loop(config, registry):
    provider = ScriptedProvider(
        [
            llm_response(tool_calls=[tool_call("book_flight", {"to": "Tokyo"}, "tc_1")]),
            llm_response("I can't book flights."),
        ]
    )
    response = build_agent(config, registry, provider).run("Book me a flight")

    result = provider.calls[1]["history"][2].results[0]
    assert result.is_error is True
    assert "unknown_tool" in result.content
    assert "can't book" in response.answer.lower()


def test_the_loop_guard_stops_a_runaway_conversation(config, registry):
    from dataclasses import replace

    capped = replace(config, max_iterations=3)
    provider = ScriptedProvider(
        [llm_response(tool_calls=[tool_call("search_destination_guide", {"query": "Tokyo"})])] * 3
    )

    response = build_agent(capped, registry, provider).run("Tell me about Tokyo")
    assert "could not settle on a final answer" in response.answer
    assert len(provider.calls) == 3


def test_max_tokens_truncation_is_disclosed(config, registry):
    provider = ScriptedProvider([llm_response("A partial answer", stop_reason="max_tokens")])
    response = build_agent(config, registry, provider).run("Tell me about Tokyo")
    assert "cut short" in response.answer


def test_a_refusal_is_handled(config, registry):
    provider = ScriptedProvider([llm_response("", stop_reason="refusal")])
    response = build_agent(config, registry, provider).run("Something disallowed")
    assert "not able to help" in response.answer


def test_an_empty_response_does_not_produce_a_blank_answer(config, registry):
    provider = ScriptedProvider([llm_response("")])
    response = build_agent(config, registry, provider).run("Tell me about Tokyo")
    assert "rephrase" in response.answer


# --- trace (Module 1: visible reasoning) ---------------------------------


def test_the_trace_records_the_full_decision_path(config, registry):
    provider = ScriptedProvider(
        [
            llm_response(
                thinking="Needs both the guide and the forecast.",
                tool_calls=[
                    tool_call("search_destination_guide", {"query": "Reykjavik packing"}, "tc_1"),
                    tool_call(
                        "get_weather_forecast",
                        {"city": "Reykjavik", "date_or_month": "December"},
                        "tc_2",
                    ),
                ],
            ),
            llm_response("Pack waterproof layers."),
        ]
    )
    response = build_agent(config, registry, provider).run(
        "What should I pack for Reykjavik in December?"
    )

    assert [s.kind for s in response.trace.steps] == [
        "user_query",
        "llm_turn",
        "tool_call",
        "tool_call",
        "llm_turn",
        "final_answer",
    ]

    payload = response.trace.to_dict()
    assert payload["tools_used"] == ["search_destination_guide", "get_weather_forecast"]
    assert payload["usage"]["input"] > 0 and payload["usage"]["output"] > 0

    rendered = response.trace.render()
    assert "REASONING TRACE" in rendered
    assert "Needs both the guide and the forecast." in rendered
    assert "get_weather_forecast" in rendered


def test_the_trace_names_the_provider_and_model(config, registry):
    provider = ScriptedProvider([llm_response("Hello.")])
    response = build_agent(config, registry, provider).run("hi")
    assert response.trace.model == "scripted · test-model"


def test_the_trace_serialises_to_json(config, registry):
    provider = ScriptedProvider([llm_response("Hello.")])
    response = build_agent(config, registry, provider).run("hi")
    assert json.loads(response.trace.to_json())["query"] == "hi"


# --- multi-turn conversation ---------------------------------------------


def test_a_follow_up_sees_the_previous_exchange(config, registry):
    """Without this the REPL restarts cold every turn, and the system prompt's
    promise of answerable follow-ups is a lie."""
    first = ScriptedProvider([llm_response("Tokyo is warm in May.")])
    turn_one = build_agent(config, registry, first).run("How warm is Tokyo in May?")

    second = ScriptedProvider([llm_response("Bangkok is hotter.")])
    build_agent(config, registry, second).run("And Bangkok?", history=turn_one.history)

    sent = second.calls[0]["history"]
    assert [type(t).__name__ for t in sent] == ["UserTurn", "AssistantTurn", "UserTurn"]
    assert sent[0].text == "How warm is Tokyo in May?"
    assert sent[2].text == "And Bangkok?"


def test_a_tool_using_turn_is_replayable_in_full(config, registry):
    provider = ScriptedProvider(
        [
            llm_response(tool_calls=[tool_call("search_destination_guide", {"query": "Tokyo visa"}, "tc_a")]),
            llm_response("Visa-free for 90 days."),
        ]
    )
    result = build_agent(config, registry, provider).run("Do I need a visa for Japan?")

    kinds = [type(t).__name__ for t in result.history]
    assert kinds == ["UserTurn", "AssistantTurn", "ToolResultsTurn", "AssistantTurn"]
    # Every tool call must be answered, or both wire formats reject the replay.
    call_ids = {c.id for t in result.history if isinstance(t, AssistantTurn) for c in t.response.tool_calls}
    result_ids = {r.id for t in result.history if isinstance(t, ToolResultsTurn) for r in t.results}
    assert call_ids == result_ids


def test_history_is_trimmed_at_user_turn_boundaries(config, registry):
    """Trimming mid-exchange would orphan a tool call from its result."""
    trimmed = Config(provider="groq", api_key="k", model="m", max_history_turns=2)
    history = []
    for n in range(4):
        provider = ScriptedProvider(
            [
                llm_response(tool_calls=[tool_call("search_destination_guide", {"query": f"q{n}"}, f"tc_{n}")]),
                llm_response(f"Answer {n}."),
            ]
        )
        history = build_agent(trimmed, registry, provider).run(f"Question {n}?", history=history).history

    user_turns = [t for t in history if isinstance(t, UserTurn)]
    assert len(user_turns) <= 3  # 2 carried forward + the current one
    assert isinstance(history[0], UserTurn), "a replay must start on a user turn"

    call_ids = {c.id for t in history if isinstance(t, AssistantTurn) for c in t.response.tool_calls}
    result_ids = {r.id for t in history if isinstance(t, ToolResultsTurn) for r in t.results}
    assert call_ids == result_ids, "trimming must not orphan a tool call"


def test_a_failed_turn_does_not_poison_the_conversation(config, registry):
    """A turn that died mid-tool-call cannot be replayed, so it is dropped."""
    good = ScriptedProvider([llm_response("Tokyo is warm in May.")])
    healthy = build_agent(config, registry, good).run("How warm is Tokyo in May?").history

    broken = build_agent(config, registry, FailingProvider(ConnectionError("down")))
    after = broken.run("And Bangkok?", history=healthy)

    assert after.history == healthy, "the failed exchange must leave no trace in history"


def test_an_empty_assistant_answer_is_not_carried_forward(config, registry):
    """An assistant message with no content is rejected by both wire formats."""
    provider = ScriptedProvider([llm_response("")])
    result = build_agent(config, registry, provider).run("Hello?")
    assert result.history == []
    assert result.answer  # the user still gets a usable fallback


def test_a_rejected_query_leaves_history_untouched(config, registry):
    good = ScriptedProvider([llm_response("Tokyo is warm in May.")])
    healthy = build_agent(config, registry, good).run("How warm is Tokyo in May?").history

    idle = ScriptedProvider([])
    after = build_agent(config, registry, idle).run("   ", history=healthy)
    assert after.history == healthy
    assert idle.calls == [], "a rejected query must never reach the provider"


def test_run_still_defaults_to_a_fresh_conversation(config, registry):
    provider = ScriptedProvider([llm_response("Answer.")])
    build_agent(config, registry, provider).run("Tell me about Tokyo")
    assert len(provider.calls[0]["history"]) == 1
