"""Unit tests for Module 3: the weather forecast tool."""

from __future__ import annotations

import json

import pytest

from tripmate.forecast.providers import MockTableProvider
from tripmate.forecast.service import DateParseError, Forecast, Unavailable, parse_month
from tripmate.tools import weather_tool
from tripmate.tools.errors import ToolInputError

# --- date parsing --------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("December", 12),
        ("december", 12),
        ("Dec", 12),
        ("sept", 9),
        ("2026-12-25", 12),
        ("2026-4-1", 4),
        ("12/25", 12),
        ("25 December 2026", 12),
        ("early April", 4),
        ("7", 7),
    ],
)
def test_parse_month_accepts_common_formats(raw, expected):
    assert parse_month(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "banana", "13", "0", "2026-13-01"])
def test_parse_month_rejects_unusable_input(raw):
    with pytest.raises(DateParseError):
        parse_month(raw)


@pytest.mark.parametrize("raw", ["summer", "next winter", "the rainy season"])
def test_parse_month_refuses_to_guess_a_season(raw):
    # Seasons span months and flip by hemisphere -- the agent must ask.
    with pytest.raises(DateParseError, match="season"):
        parse_month(raw)


# --- provider ------------------------------------------------------------


def test_provider_covers_all_four_cities_for_all_twelve_months():
    provider = MockTableProvider()
    assert provider.cities() == ["Bangkok", "Barcelona", "Reykjavik", "Tokyo"]
    for city in provider.cities():
        for month in range(1, 13):
            assert provider.lookup(city, month) is not None, f"{city}/{month} missing"


def test_temperature_ranges_are_ordered_low_then_high():
    provider = MockTableProvider()
    for city in provider.cities():
        for month in range(1, 13):
            low, high = provider.lookup(city, month).temp_range_c
            assert low < high, f"{city}/{month} has an inverted range"


# --- service -------------------------------------------------------------


def test_forecast_returns_expected_shape(weather_service):
    result = weather_service.get_forecast("Reykjavik", "December")
    assert isinstance(result, Forecast)
    payload = result.to_dict()
    assert payload["available"] is True
    assert payload["city"] == "Reykjavik"
    assert payload["month"] == "December"
    assert payload["temp_range_c"] == [-2, 3]
    assert isinstance(payload["conditions"], str) and payload["conditions"]


def test_city_lookup_is_case_insensitive(weather_service):
    assert weather_service.get_forecast("tOkYo", "April").to_dict()["city"] == "Tokyo"


def test_unsupported_city_is_reported_not_invented(weather_service):
    result = weather_service.get_forecast("Cairo", "June")
    assert isinstance(result, Unavailable)
    payload = result.to_dict()
    assert payload["available"] is False
    assert payload["reason"] == "unsupported_city"
    assert "Bangkok" in payload["supported_cities"]


def test_unparseable_date_is_reported_not_guessed(weather_service):
    payload = weather_service.get_forecast("Tokyo", "someday").to_dict()
    assert payload["available"] is False
    assert payload["reason"] == "unparseable_date"


def test_empty_city_is_rejected(weather_service):
    payload = weather_service.get_forecast("", "June").to_dict()
    assert payload["available"] is False
    assert payload["reason"] == "invalid_input"


def test_provider_failure_does_not_escape_the_service():
    """A tool failure or timeout must surface as data, never a crash."""

    class ExplodingProvider:
        name = "exploding"

        def cities(self):
            return ["Tokyo"]

        def lookup(self, city, month):
            raise TimeoutError("upstream weather API timed out")

    from tripmate.forecast.service import WeatherService

    service = WeatherService(provider=ExplodingProvider())
    with pytest.raises(TimeoutError):
        service.get_forecast("Tokyo", "June")
    # ...and the registry converts that into an error result; see test_agent.py.


# --- the tool wrapper ----------------------------------------------------


def test_tool_returns_json_payload(weather_service):
    handler = weather_tool.make_handler(weather_service)
    payload = json.loads(handler(city="Bangkok", date_or_month="April"))
    assert payload["available"] is True
    assert payload["temp_range_c"] == [27, 36]


def test_tool_reports_unavailable_as_a_successful_call(weather_service):
    handler = weather_tool.make_handler(weather_service)
    payload = json.loads(handler(city="Cairo", date_or_month="April"))
    assert payload["available"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"city": "", "date_or_month": "April"},
        {"city": "Tokyo", "date_or_month": ""},
        {"city": None, "date_or_month": "April"},
    ],
)
def test_tool_validates_its_arguments(weather_service, kwargs):
    handler = weather_tool.make_handler(weather_service)
    with pytest.raises(ToolInputError):
        handler(**kwargs)
