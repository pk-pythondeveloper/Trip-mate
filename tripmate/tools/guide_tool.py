"""The `search_destination_guide` tool: schema + adapter over the retriever.

The description below is not documentation -- it is the routing logic. It is
the only thing the model reads when deciding whether this tool applies, so it
names the covered cities and the covered topics explicitly, and says what the
tool does *not* do.
"""

from __future__ import annotations

import json

from tripmate.logging_setup import get_logger
from tripmate.rag.retriever import DestinationRetriever
from tripmate.tools.errors import ToolInputError

log = get_logger(__name__)

NAME = "search_destination_guide"

DESCRIPTION = (
    "Search TripMate's curated destination guides and return the most relevant "
    "excerpts. Covers exactly four cities: Tokyo, Reykjavik, Bangkok, and Barcelona.\n\n"
    "Each guide covers five topics: visa and entry requirements, the best time of "
    "year to visit, local customs and etiquette, packing tips, and safety and health.\n\n"
    "Use this for any question about what a destination is like, what to expect "
    "there, or how to behave there. Include the city name in your query so the "
    "search can target the right guide.\n\n"
    "This tool returns reference text only. It does NOT provide temperatures or "
    "forecasts for a specific month -- use get_weather_forecast for that. For "
    "packing questions, call BOTH tools: the guide gives destination-specific "
    "advice, the forecast gives the actual conditions to pack for."
)

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "A natural-language question or topic, including the city name. "
                "Example: 'Do I need a visa for Japan?' or 'Reykjavik winter packing'."
            ),
        }
    },
    "required": ["query"],
    "additionalProperties": False,
}


def make_handler(retriever: DestinationRetriever, top_k: int):
    """Bind the tool to a retriever instance, returning the callable handler."""

    def handler(**kwargs) -> str:
        query = kwargs.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolInputError("'query' must be a non-empty string.")

        hits = retriever.search(query.strip(), top_k=top_k)

        if not hits:
            # An honest miss. Naming the covered cities lets the model explain
            # the gap instead of inventing coverage we do not have.
            return json.dumps(
                {
                    "results": [],
                    "message": (
                        "No destination guide content matched this query. The knowledge "
                        "base covers only the cities listed in covered_cities."
                    ),
                    "covered_cities": retriever.cities,
                },
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "results": [
                    {
                        "city": h.chunk.city,
                        "section": h.chunk.section_title.title(),
                        "text": h.chunk.text,
                        "relevance": round(h.score, 3),
                        "citation": h.chunk.citation(),
                    }
                    for h in hits
                ],
                "covered_cities": retriever.cities,
            },
            ensure_ascii=False,
        )

    return handler
