"""Load destination guides from disk and split them into retrievable chunks.

Chunking strategy: **one chunk per section per document** (4 cities x 5
sections = 20 chunks).

Why: the guides are already authored as short, self-contained, topically pure
sections whose headings map almost one-to-one onto the questions users ask
("visa", "when should I go", "what do I pack"). Splitting on a boundary the
author already drew beats any fixed window: no sentence is cut in half, no
chunk mixes visa rules with packing advice, and every chunk carries a city and
a section label we can cite back to the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Section headings as they appear in the data pack, mapped to a stable key.
SECTION_HEADINGS: dict[str, str] = {
    "VISA & ENTRY": "visa_entry",
    "BEST TIME TO VISIT": "best_time_to_visit",
    "LOCAL CUSTOMS": "local_customs",
    "PACKING TIPS": "packing_tips",
    "SAFETY & HEALTH": "safety_health",
}

# How many times the section heading is repeated in a chunk's embedding text.
# See Chunk.embedding_text for why, and the README for the measurement.
SECTION_TITLE_BOOST = 3

_TITLE_RE = re.compile(
    r"^DESTINATION GUIDE:\s*(?P<city>[^,]+),\s*(?P<country>.+)$", re.MULTILINE
)
_HEADING_RE = re.compile(
    r"^(" + "|".join(re.escape(h) for h in SECTION_HEADINGS) + r")$", re.MULTILINE
)


class IngestionError(RuntimeError):
    """Raised when a guide file cannot be parsed into chunks."""


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of the knowledge base."""

    chunk_id: str
    city: str
    country: str
    section: str  # stable key, e.g. "packing_tips"
    section_title: str  # display heading, e.g. "PACKING TIPS"
    text: str
    source: str  # originating filename

    def embedding_text(self) -> str:
        """Text handed to the embedder.

        Two deliberate choices:

        1. The city and country are prepended, so the vector encodes *which
           destination* the passage describes. Without this the four "PACKING
           TIPS" chunks embed almost identically and the retriever cannot tell
           Tokyo from Bangkok.
        2. The section heading is repeated `SECTION_TITLE_BOOST` times -- field
           boosting, the classic trick for weighting a title above body text
           when the ranker has no notion of fields. It matters: a section
           heading is one token inside a ~60-word body, so without a boost
           "what should I pack" loses to any chunk that merely happens to
           mention the same month. Measured on the real corpus, a boost of 3
           moves PACKING TIPS from rank 2 to rank 1 with clear separation.
        """
        heading = " ".join([self.section_title] * SECTION_TITLE_BOOST)
        return f"{self.city}, {self.country} - {heading}: {self.text}"

    def citation(self) -> str:
        return f"{self.city} guide / {self.section_title.title()}"


def parse_guide(text: str, source: str) -> list[Chunk]:
    """Split one guide document into per-section chunks."""
    title = _TITLE_RE.search(text)
    if not title:
        raise IngestionError(
            f"{source}: missing 'DESTINATION GUIDE: <City>, <Country>' title line"
        )

    city = title.group("city").strip().title()
    country = title.group("country").strip().title()

    # Split keeping the headings: [preamble, HEADING, body, HEADING, body, ...]
    parts = _HEADING_RE.split(text)
    stem = source[:-4] if source.endswith(".txt") else source

    chunks: list[Chunk] = []
    for heading, body in zip(parts[1::2], parts[2::2]):
        heading = heading.strip()
        body = body.strip()
        if not body:
            continue
        section = SECTION_HEADINGS[heading]
        chunks.append(
            Chunk(
                chunk_id=f"{stem}:{section}",
                city=city,
                country=country,
                section=section,
                section_title=heading,
                text=body,
                source=source,
            )
        )

    if not chunks:
        raise IngestionError(f"{source}: no known section headings found")
    return chunks


def load_chunks(data_dir: Path) -> list[Chunk]:
    """Ingest every destination guide in `data_dir`."""
    if not data_dir.is_dir():
        raise IngestionError(f"Data directory not found: {data_dir}")

    files = sorted(p for p in data_dir.glob("*.txt") if p.name.lower() != "readme.txt")
    if not files:
        raise IngestionError(f"No destination guides (*.txt) found in {data_dir}")

    chunks: list[Chunk] = []
    for path in files:
        chunks.extend(parse_guide(path.read_text(encoding="utf-8"), path.name))
    return chunks
