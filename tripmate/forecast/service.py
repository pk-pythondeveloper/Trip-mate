"""Weather forecast business logic.

Responsibilities: normalise the loose `date_or_month` string an LLM will send,
look the answer up through a provider, and report *unavailable* honestly
instead of guessing. Knows nothing about tool schemas or the agent loop.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field

from tripmate.forecast.providers import MockTableProvider, WeatherProvider
from tripmate.logging_setup import get_logger

log = get_logger(__name__)

_MONTHS: dict[str, int] = {}
for _i in range(1, 13):
    _MONTHS[calendar.month_name[_i].casefold()] = _i  # "december"
    _MONTHS[calendar.month_abbr[_i].casefold()] = _i  # "dec"
_MONTHS["sept"] = 9

_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})(?:-(\d{1,2}))?\b")
_SLASH_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")
_BARE_MONTH_RE = re.compile(r"^\s*(\d{1,2})\s*$")

# Seasons are hemisphere-dependent and span three months, so we refuse to
# silently pick one. The agent is told to ask the user instead.
_SEASON_WORDS = {"spring", "summer", "autumn", "fall", "winter", "monsoon", "rainy season"}


class DateParseError(ValueError):
    """The date_or_month argument could not be resolved to a single month."""


@dataclass(frozen=True)
class Forecast:
    """Result of a successful lookup."""

    city: str
    month: int
    month_name: str
    conditions: str
    temp_range_c: tuple[int, int]
    precipitation: str
    daylight_hours: float
    source: str

    def to_dict(self) -> dict:
        return {
            "available": True,
            "city": self.city,
            "month": self.month_name,
            "conditions": self.conditions,
            "temp_range_c": list(self.temp_range_c),
            "precipitation": self.precipitation,
            "daylight_hours": self.daylight_hours,
            "source": self.source,
            "note": "Typical monthly climate normals, not a live forecast.",
        }


@dataclass(frozen=True)
class Unavailable:
    """A lookup that could not be answered. Never a fabricated forecast."""

    reason: str
    detail: str
    supported_cities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = {"available": False, "reason": self.reason, "detail": self.detail}
        if self.supported_cities:
            payload["supported_cities"] = self.supported_cities
        return payload


def parse_month(date_or_month: str) -> int:
    """Resolve a loose date reference to a month number (1-12).

    Accepts: "December", "dec", "2026-12-25", "12/25", "25 December 2026",
    and a bare "12". Raises `DateParseError` on anything ambiguous.
    """
    if not date_or_month or not date_or_month.strip():
        raise DateParseError("No date or month was provided.")

    text = date_or_month.strip()
    lowered = text.casefold()

    # Named month anywhere in the string handles most LLM output.
    for name, number in _MONTHS.items():
        if re.search(rf"\b{name}\b", lowered):
            return number

    iso = _ISO_RE.search(text)
    if iso:
        month = int(iso.group(2))
        if 1 <= month <= 12:
            return month
        raise DateParseError(f"'{text}' contains month {month}, which is not between 1 and 12.")

    slash = _SLASH_RE.search(text)
    if slash:
        month = int(slash.group(1))  # US-style MM/DD
        if 1 <= month <= 12:
            return month
        raise DateParseError(f"'{text}' contains month {month}, which is not between 1 and 12.")

    bare = _BARE_MONTH_RE.match(text)
    if bare:
        month = int(bare.group(1))
        if 1 <= month <= 12:
            return month
        raise DateParseError(f"'{text}' is not a month number between 1 and 12.")

    if any(word in lowered for word in _SEASON_WORDS):
        raise DateParseError(
            f"'{text}' names a season rather than a month. Seasons span several months and "
            "differ by hemisphere -- ask the user which month they mean."
        )

    raise DateParseError(
        f"Could not interpret '{text}' as a date or month. Use a month name "
        "(e.g. 'December') or an ISO date (e.g. '2026-12-25')."
    )


class WeatherService:
    """Coordinates date parsing and provider lookup."""

    def __init__(self, provider: WeatherProvider | None = None) -> None:
        self._provider = provider or MockTableProvider()

    @property
    def supported_cities(self) -> list[str]:
        return self._provider.cities()

    def get_forecast(self, city: str, date_or_month: str) -> Forecast | Unavailable:
        if not city or not city.strip():
            return Unavailable(
                reason="invalid_input",
                detail="No city was provided.",
                supported_cities=self.supported_cities,
            )

        try:
            month = parse_month(date_or_month)
        except DateParseError as exc:
            log.info(
                "weather.date_parse_failed",
                extra={"city": city, "date_or_month": date_or_month, "detail": str(exc)},
            )
            return Unavailable(reason="unparseable_date", detail=str(exc))

        climate = self._provider.lookup(city.strip(), month)
        if climate is None:
            log.info(
                "weather.no_data",
                extra={"city": city, "month": month, "provider": self._provider.name},
            )
            return Unavailable(
                reason="unsupported_city",
                detail=(
                    f"No weather data is held for '{city.strip()}'. "
                    "This assistant covers a fixed set of destinations."
                ),
                supported_cities=self.supported_cities,
            )

        forecast = Forecast(
            city=city.strip().title(),
            month=month,
            month_name=calendar.month_name[month],
            conditions=climate.conditions,
            temp_range_c=climate.temp_range_c,
            precipitation=climate.precipitation,
            daylight_hours=climate.daylight_hours,
            source=self._provider.name,
        )
        log.info(
            "weather.lookup_ok",
            extra={
                "city": forecast.city,
                "month": forecast.month_name,
                "temp_range_c": list(forecast.temp_range_c),
            },
        )
        return forecast
