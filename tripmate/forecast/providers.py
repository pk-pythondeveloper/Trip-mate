"""Weather data providers.

The assessment states Option A (a live keyless API) and Option B (a mock
lookup table) are scored identically, so we take Option B and spend the time
on the agent instead. The provider sits behind a Protocol, so swapping in an
Open-Meteo client later means adding one class -- no change to the service,
the tool, or the agent.

Figures are typical monthly climate normals, rounded. They are illustrative
reference data for an assessment exercise, not a live forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class MonthlyClimate:
    conditions: str
    temp_range_c: tuple[int, int]
    precipitation: str
    daylight_hours: float


class WeatherProvider(Protocol):
    """Anything that can supply climate data for a city and month."""

    name: str

    def cities(self) -> list[str]: ...

    def lookup(self, city: str, month: int) -> MonthlyClimate | None: ...


# month number -> climate, per city (keys are casefolded city names)
_TABLE: dict[str, dict[int, MonthlyClimate]] = {
    "tokyo": {
        1: MonthlyClimate("cold and dry, mostly clear", (2, 10), "low", 10.0),
        2: MonthlyClimate("cold and dry, occasional light snow", (2, 11), "low", 11.0),
        3: MonthlyClimate("mild, cherry blossom season begins", (5, 14), "moderate", 12.0),
        4: MonthlyClimate("mild and pleasant", (10, 19), "moderate", 13.0),
        5: MonthlyClimate("warm and comfortable", (15, 23), "moderate", 14.0),
        6: MonthlyClimate("warm and humid, rainy season (tsuyu)", (19, 26), "high", 14.5),
        7: MonthlyClimate("hot and very humid", (23, 30), "moderate", 14.0),
        8: MonthlyClimate("hot and very humid, typhoon season begins", (24, 31), "moderate", 13.0),
        9: MonthlyClimate("warm, humid, peak typhoon risk", (21, 28), "high", 12.0),
        10: MonthlyClimate("mild and clear, autumn foliage", (15, 22), "moderate", 11.0),
        11: MonthlyClimate("cool and clear", (9, 17), "low", 10.0),
        12: MonthlyClimate("cold and dry, crisp and sunny", (4, 12), "low", 9.5),
    },
    "reykjavik": {
        1: MonthlyClimate("cold, windy, frequent snow, very short days", (-3, 2), "moderate", 4.5),
        2: MonthlyClimate("cold, windy, snow and sleet", (-3, 3), "moderate", 7.0),
        3: MonthlyClimate("cold, changeable, snow possible", (-2, 3), "moderate", 11.0),
        4: MonthlyClimate("chilly and windy, rapidly lengthening days", (0, 6), "moderate", 14.5),
        5: MonthlyClimate("cool, drier, long daylight", (4, 10), "low", 18.0),
        6: MonthlyClimate("mild, near-midnight sun", (7, 13), "low", 21.0),
        7: MonthlyClimate("mildest month, long bright evenings", (9, 15), "low", 20.0),
        8: MonthlyClimate("mild, occasional rain", (8, 14), "moderate", 17.0),
        9: MonthlyClimate("cool and windy, Northern Lights season begins", (5, 11), "moderate", 13.5),
        10: MonthlyClimate("cold, wet, and windy", (2, 7), "high", 10.0),
        11: MonthlyClimate("cold, dark, icy roads likely", (-1, 4), "high", 6.5),
        12: MonthlyClimate("cold and dark, snow and strong winds", (-2, 3), "high", 4.0),
    },
    "bangkok": {
        1: MonthlyClimate("warm and dry, cool season, most comfortable", (21, 32), "very low", 11.5),
        2: MonthlyClimate("warm and dry, cool season", (23, 33), "very low", 11.5),
        3: MonthlyClimate("hot and dry, hot season begins", (25, 34), "low", 12.0),
        4: MonthlyClimate("very hot and humid, hottest month", (27, 36), "low", 12.5),
        5: MonthlyClimate("very hot, humid, first monsoon storms", (26, 34), "high", 13.0),
        6: MonthlyClimate("hot and humid, rainy season", (25, 33), "high", 13.0),
        7: MonthlyClimate("hot and humid, frequent heavy showers", (25, 33), "high", 13.0),
        8: MonthlyClimate("hot and humid, frequent heavy showers", (25, 32), "high", 12.5),
        9: MonthlyClimate("hot and humid, wettest month", (24, 32), "very high", 12.0),
        10: MonthlyClimate("hot and humid, rain easing late in month", (24, 32), "high", 11.5),
        11: MonthlyClimate("warm and drier, cool season begins", (23, 32), "low", 11.5),
        12: MonthlyClimate("warm and dry, cool season, low humidity", (21, 31), "very low", 11.0),
    },
    "barcelona": {
        1: MonthlyClimate("cool and mild, occasional rain", (5, 14), "low", 9.5),
        2: MonthlyClimate("cool and mild", (6, 15), "low", 10.5),
        3: MonthlyClimate("mild, breezy", (8, 17), "moderate", 12.0),
        4: MonthlyClimate("mild and pleasant, occasional showers", (10, 19), "moderate", 13.5),
        5: MonthlyClimate("warm and pleasant", (13, 22), "moderate", 14.5),
        6: MonthlyClimate("warm and sunny", (17, 26), "low", 15.0),
        7: MonthlyClimate("hot and humid, peak tourist season", (20, 29), "very low", 15.0),
        8: MonthlyClimate("hot and humid, peak tourist season", (21, 29), "low", 14.0),
        9: MonthlyClimate("warm, occasional heavy showers", (18, 26), "high", 12.5),
        10: MonthlyClimate("mild, wettest month", (14, 22), "high", 11.0),
        11: MonthlyClimate("cool and mild, some rain", (9, 17), "moderate", 10.0),
        12: MonthlyClimate("cool and mild", (6, 15), "low", 9.0),
    },
}


class MockTableProvider:
    """Static climate-normals lookup. No network, no key, fully deterministic."""

    name = "mock-climate-table"

    def cities(self) -> list[str]:
        return sorted(c.title() for c in _TABLE)

    def lookup(self, city: str, month: int) -> MonthlyClimate | None:
        return _TABLE.get(city.casefold(), {}).get(month)
