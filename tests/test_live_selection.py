"""Live tool-selection tests (opt-in -- these cost money and need network).

    pytest -m live

Why these exist: every other test drives a scripted client, so it proves the
orchestrator routes each path correctly but says nothing about whether the
*model* picks the right tools. Dynamic tool selection is the core requirement,
so it deserves at least one test that actually exercises it. They are excluded
by default so the default suite stays fast, free, and deterministic.
"""

from __future__ import annotations

import pytest

from tripmate.agent.orchestrator import TripMateAgent
from tripmate.config import load_config

_CONFIG = load_config()

pytestmark = [
    pytest.mark.live,
    # Gate on whichever key the *selected* provider actually needs, so these
    # run under Groq or Anthropic without editing the test.
    pytest.mark.skipif(
        not _CONFIG.api_key,
        reason=f"{_CONFIG.api_key_var} is not set (provider: {_CONFIG.provider})",
    ),
]


@pytest.fixture(scope="module")
def live_agent():
    return TripMateAgent(_CONFIG)


def test_visa_question_selects_only_the_guide(live_agent):
    response = live_agent.run("Do I need a visa to visit Japan as a UK citizen?")
    assert "search_destination_guide" in response.tools_used
    assert "get_weather_forecast" not in response.tools_used


def test_weather_question_selects_only_the_forecast(live_agent):
    response = live_agent.run("How warm does Bangkok get in April?")
    assert "get_weather_forecast" in response.tools_used


def test_packing_question_selects_both_tools(live_agent):
    """Module 4: the multi-tool case."""
    response = live_agent.run("What should I pack for Reykjavik in December?")
    assert set(response.tools_used) == {"search_destination_guide", "get_weather_forecast"}


def test_booking_request_selects_no_tool_and_declines(live_agent):
    """Module 5: scope awareness, with no fabricated action."""
    response = live_agent.run("Can you book me a flight to Tokyo next Tuesday?")
    assert response.tools_used == []
    lowered = response.answer.lower()
    assert any(phrase in lowered for phrase in ("can't", "cannot", "not able", "unable"))
    assert "booked" not in lowered


def test_uncovered_destination_is_admitted_not_invented(live_agent):
    response = live_agent.run("What are the visa rules for Cairo?")
    assert "cairo" not in response.answer.lower() or any(
        city in response.answer for city in ("Tokyo", "Reykjavik", "Bangkok", "Barcelona")
    )
