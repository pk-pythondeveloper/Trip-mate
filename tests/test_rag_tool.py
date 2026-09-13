"""Unit tests for Module 2: the destination knowledge (RAG) tool."""

from __future__ import annotations

import json

import pytest

from tripmate.rag.embedder import LexicalEmbedder, _stem
from tripmate.rag.ingest import IngestionError, SECTION_HEADINGS, load_chunks, parse_guide
from tripmate.tools import guide_tool

# --- ingestion / chunking ------------------------------------------------


def test_ingest_produces_one_chunk_per_section_per_city(config):
    chunks = load_chunks(config.data_dir)
    assert len(chunks) == 20, "4 cities x 5 sections"
    assert {c.city for c in chunks} == {"Tokyo", "Reykjavik", "Bangkok", "Barcelona"}
    for city in {c.city for c in chunks}:
        sections = {c.section for c in chunks if c.city == city}
        assert sections == set(SECTION_HEADINGS.values()), f"{city} is missing sections"


def test_chunk_ids_are_unique(config):
    chunks = load_chunks(config.data_dir)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_embedding_text_carries_city_and_boosted_section(config):
    chunk = next(c for c in load_chunks(config.data_dir) if c.chunk_id == "tokyo:packing_tips")
    embedded = chunk.embedding_text()
    assert "Tokyo" in embedded and "Japan" in embedded
    # Field boosting: the heading appears more than once on purpose.
    assert embedded.count("PACKING TIPS") == 3
    assert chunk.text in embedded


def test_parse_guide_rejects_a_document_with_no_title():
    with pytest.raises(IngestionError, match="title line"):
        parse_guide("VISA & ENTRY\nSome text.", "broken.txt")


def test_parse_guide_rejects_a_document_with_no_sections():
    with pytest.raises(IngestionError, match="no known section headings"):
        parse_guide("DESTINATION GUIDE: Atlantis, Nowhere\n\nJust prose.", "broken.txt")


def test_load_chunks_rejects_a_missing_directory(tmp_path):
    with pytest.raises(IngestionError, match="Data directory not found"):
        load_chunks(tmp_path / "does-not-exist")


def test_load_chunks_rejects_an_empty_directory(tmp_path):
    with pytest.raises(IngestionError, match="No destination guides"):
        load_chunks(tmp_path)


# --- embedder ------------------------------------------------------------


@pytest.mark.parametrize(
    "word,expected",
    [("packing", "pack"), ("customs", "custom"), ("tips", "tip"), ("visas", "visa"), ("is", "is")],
)
def test_stemmer_collapses_inflections(word, expected):
    assert _stem(word) == expected


def test_lexical_embedder_returns_normalised_vectors():
    embedder = LexicalEmbedder().fit(["warm winter coat", "light summer shirt"])
    vectors = embedder.encode(["warm winter coat"])
    assert pytest.approx(1.0, abs=1e-5) == float((vectors[0] ** 2).sum())


def test_lexical_embedder_requires_fit_first():
    with pytest.raises(RuntimeError, match="fit()"):
        LexicalEmbedder().encode(["anything"])


# --- retrieval -----------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected_chunk",
    [
        ("Do I need a visa for Japan?", "tokyo:visa_entry"),
        ("What should I pack for Reykjavik in December?", "reykjavik:packing_tips"),
        ("Is tipping expected in Iceland?", "reykjavik:local_customs"),
        ("Is Barcelona safe for tourists?", "barcelona:safety_health"),
        ("When is the best time to visit Tokyo?", "tokyo:best_time_to_visit"),
    ],
)
def test_retriever_ranks_the_right_section_first(retriever, query, expected_chunk):
    hits = retriever.search(query, top_k=3)
    assert hits, f"expected a match for {query!r}"
    assert hits[0].chunk.chunk_id == expected_chunk


@pytest.mark.parametrize(
    "query,expected_city",
    [
        ("visa for Japan", "Tokyo"),
        ("customs in Thailand", "Bangkok"),
        ("weather in Reykjavik", "Reykjavik"),
        ("is Spain expensive", "Barcelona"),
        ("what about Cairo", None),
    ],
)
def test_city_detection_handles_city_and_country_names(retriever, query, expected_city):
    assert retriever.detect_city(query) == expected_city


def test_retrieval_is_scoped_to_the_named_city(retriever):
    hits = retriever.search("What should I pack for Bangkok?", top_k=5)
    assert hits
    assert {h.chunk.city for h in hits} == {"Bangkok"}


def test_unknown_destination_returns_no_results(retriever):
    assert retriever.search("Tell me about Cairo") == []


def test_gibberish_returns_no_results(retriever):
    assert retriever.search("asdfghjkl qwertyuiop") == []


def test_empty_query_returns_no_results(retriever):
    assert retriever.search("   ") == []


# --- the tool wrapper ----------------------------------------------------


def test_tool_returns_structured_results(retriever, config):
    handler = guide_tool.make_handler(retriever, config.top_k)
    payload = json.loads(handler(query="Do I need a visa for Japan?"))
    assert payload["results"], "expected at least one result"
    top = payload["results"][0]
    assert top["city"] == "Tokyo"
    assert top["section"] == "Visa & Entry"
    assert "citation" in top and 0.0 <= top["relevance"] <= 1.0


def test_tool_reports_scope_honestly_on_a_miss(retriever, config):
    handler = guide_tool.make_handler(retriever, config.top_k)
    payload = json.loads(handler(query="Tell me about Cairo"))
    assert payload["results"] == []
    # The model must be able to name what it *does* cover.
    assert set(payload["covered_cities"]) == {"Tokyo", "Reykjavik", "Bangkok", "Barcelona"}


def test_tool_rejects_an_empty_query(retriever, config):
    from tripmate.tools.errors import ToolInputError

    handler = guide_tool.make_handler(retriever, config.top_k)
    with pytest.raises(ToolInputError):
        handler(query="  ")
