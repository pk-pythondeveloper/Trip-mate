"""The `get_weather_forecast` tool: schema + adapter over the weather service."""

from __future__ import annotations

import json

from tripmate.forecast.service import Forecast, WeatherService
from tripmate.logging_setup import get_logger
from tripmate.tools.errors import ToolInputError

log = get_logger(__name__)

NAME = "get_weather_forecast"

DESCRIPTION = (
    "Get typical weather conditions for a city in a given month: temperature "
    "range in Celsius, general conditions, precipitation level, and daylight "
    "hours. Covers exactly four cities: Tokyo, Reykjavik, Bangkok, and Barcelona.\n\n"
    "Use this whenever the answer depends on what the weather will actually be "
    "like -- including packing questions, 'is X a good time to visit', and any "
    "question mentioning a month or travel date.\n\n"
    "Returns monthly climate normals, not a live day-by-day forecast, so it "
    "cannot answer 'will it rain next Tuesday'. If the user names a season "
    "rather than a month, ask which month they mean rather than guessing."
)

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {
            "type": "string",
            "description": "City name, e.g. 'Reykjavik'. One city per call.",
        },
        "date_or_month": {
            "type": "string",
            "description": (
                "A month name or a date. Accepts 'December', 'Dec', '2026-12-25', "
                "or '12/25'. Must resolve to a single month."
            ),
        },
    },
    "required": ["city", "date_or_month"],
    "additionalProperties": False,
}


def make_handler(service: WeatherService):
    """Bind the tool to a weather service instance."""

    def handler(**kwargs) -> str:
        city = kwargs.get("city")
        date_or_month = kwargs.get("date_or_month")

        if not isinstance(city, str) or not city.strip():
            raise ToolInputError("'city' must be a non-empty string.")
        if not isinstance(date_or_month, str) or not date_or_month.strip():
            raise ToolInputError("'date_or_month' must be a non-empty string.")

        result = service.get_forecast(city, date_or_month)
        payload = result.to_dict()

        # An unavailable result is a *successful* tool call reporting a gap --
        # not an error. The model needs it to say "I don't have that data".
        if not isinstance(result, Forecast):
            log.info(
                "weather.unavailable",
                extra={"city": city, "date_or_month": date_or_month, "reason": payload["reason"]},
            )

        return json.dumps(payload, ensure_ascii=False)

    return handler
