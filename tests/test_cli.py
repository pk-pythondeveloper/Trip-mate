"""CLI wiring.

The agent supports follow-ups; these tests check the REPL actually *uses*
that. The mechanism being correct while the caller never passes the history
is a silent, invisible failure -- it just looks like a forgetful model.
"""

from __future__ import annotations

import argparse

import pytest

from tests.conftest import ScriptedProvider, llm_response
from tripmate import cli
from tripmate.agent.orchestrator import TripMateAgent


def _args(**overrides) -> argparse.Namespace:
    base = dict(verbose=False, trace_json=None, interactive=True)
    base.update(overrides)
    return argparse.Namespace(**base)


def _feed(monkeypatch, lines: list[str]) -> None:
    queued = iter(lines)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(queued))


def test_the_repl_carries_history_between_turns(config, registry, monkeypatch, capsys):
    provider = ScriptedProvider([llm_response("Tokyo is warm."), llm_response("Bangkok is hotter.")])
    agent = TripMateAgent(config, registry=registry, provider=provider)

    _feed(monkeypatch, ["How warm is Tokyo in May?", "And Bangkok?", "exit"])
    cli._repl(agent, _args())

    second_turn = provider.calls[1]["history"]
    assert len(second_turn) == 3, "the follow-up must see the previous exchange"
    assert second_turn[0].text == "How warm is Tokyo in May?"

    out = capsys.readouterr().out
    assert "Tokyo is warm." in out and "Bangkok is hotter." in out


def test_new_clears_the_conversation(config, registry, monkeypatch, capsys):
    provider = ScriptedProvider([llm_response("First."), llm_response("Second.")])
    agent = TripMateAgent(config, registry=registry, provider=provider)

    _feed(monkeypatch, ["Tell me about Tokyo", "new", "Tell me about Bangkok", "exit"])
    cli._repl(agent, _args())

    assert len(provider.calls[1]["history"]) == 1, "'new' must reset the history"
    assert "[conversation cleared]" in capsys.readouterr().out


@pytest.mark.parametrize("word", ["exit", "quit"])
def test_the_repl_exits_on_command(config, registry, monkeypatch, word):
    agent = TripMateAgent(config, registry=registry, provider=ScriptedProvider([]))
    _feed(monkeypatch, [word])
    cli._repl(agent, _args())  # must return, not hang or raise


def test_a_blank_line_does_not_reach_the_provider(config, registry, monkeypatch):
    provider = ScriptedProvider([])
    agent = TripMateAgent(config, registry=registry, provider=provider)
    _feed(monkeypatch, ["", "   ", "exit"])
    cli._repl(agent, _args())
    assert provider.calls == []
