"""Structured logging: redaction and level handling.

The logs are the trace. A formatter that over-redacts or a bad level that
takes the whole CLI down both destroy the thing the logs exist for.
"""

from __future__ import annotations

import json
import logging

import pytest





from tripmate.logging_setup import JsonFormatter, resolve_level, setup_logging


def _format(**extra) -> dict:
    record = logging.LogRecord("t", logging.INFO, "f", 1, "event.name", (), None)
    record.__dict__.update(extra)
    return json.loads(JsonFormatter().format(record))


def test_credentials_are_redacted():
    payload = _format(api_key="sk-secret", auth_token="t", password="hunter2")
    assert payload["api_key"] == "[REDACTED]"
    assert payload["auth_token"] == "[REDACTED]"
    assert payload["password"] == "[REDACTED]"


def test_token_counts_and_chunk_ids_survive():
    """A substring match on 'token'/'key' silently guts the trace fields."""
    payload = _format(input_tokens=120, output_tokens=40, chunk_ids=["tokyo:visa_entry"])
    assert payload["input_tokens"] == 120
    assert payload["output_tokens"] == 40
    assert payload["chunk_ids"] == ["tokyo:visa_entry"]


def test_the_event_name_and_level_are_always_present():
    payload = _format(tool="get_weather_forecast")
    assert payload["event"] == "event.name"
    assert payload["level"] == "INFO"
    assert payload["tool"] == "get_weather_forecast"


@pytest.mark.parametrize(
    "given,expected",
    [("DEBUG", logging.DEBUG), ("info", logging.INFO), (" warning ", logging.WARNING)],
)
def test_known_levels_resolve(given, expected):
    assert resolve_level(given) == expected


def test_an_unknown_level_falls_back_instead_of_crashing():
    """`TRIPMATE_LOG_LEVEL=verbose` should not take down the CLI."""
    assert resolve_level("verbose") == logging.INFO
    setup_logging("verbose")  # must not raise
    assert logging.getLogger().level == logging.INFO
    setup_logging("CRITICAL")  # restore the quiet suite default
