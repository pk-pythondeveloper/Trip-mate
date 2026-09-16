"""Unit tests for the tool registry -- the orchestration/business boundary."""

from __future__ import annotations

import json

from tripmate.tools.registry import Tool, ToolRegistry


def test_registry_exposes_both_tools(registry):
    assert set(registry.names) == {"search_destination_guide", "get_weather_forecast"}


def test_schemas_have_the_shape_the_messages_api_expects(registry):
    for schema in registry.schemas():
        assert set(schema) == {"name", "description", "input_schema"}
        assert schema["description"].strip(), "a tool with no description cannot be routed to"
        assert schema["input_schema"]["type"] == "object"
        assert schema["input_schema"]["required"]


def test_descriptions_name_the_covered_cities(registry):
    """The description is the routing logic, so scope must be stated in it."""
    for schema in registry.schemas():
        for city in ("Tokyo", "Reykjavik", "Bangkok", "Barcelona"):
            assert city in schema["description"]


def test_dispatch_returns_a_successful_outcome(registry):
    outcome = registry.dispatch("get_weather_forecast", {"city": "Tokyo", "date_or_month": "April"})
    assert outcome.is_error is False
    assert json.loads(outcome.content)["available"] is True
    assert outcome.duration_ms >= 0


def test_unknown_tool_is_an_error_outcome_not_an_exception(registry):
    outcome = registry.dispatch("book_flight", {"to": "Paris"})
    assert outcome.is_error is True
    assert "unknown_tool" in outcome.content
    # The model is told what it *can* call, so it can recover.
    assert "search_destination_guide" in outcome.content


def test_missing_required_argument_is_reported(registry):
    outcome = registry.dispatch("get_weather_forecast", {"city": "Tokyo"})
    assert outcome.is_error is True
    assert "invalid_input" in outcome.content
    assert "date_or_month" in outcome.content


def test_unexpected_argument_is_reported(registry):
    outcome = registry.dispatch(
        "search_destination_guide", {"query": "visas", "temperature": "hot"}
    )
    assert outcome.is_error is True
    assert "temperature" in outcome.content


def test_a_crashing_tool_becomes_an_error_outcome(registry):
    """A tool that raises must not kill the agent loop."""

    def exploding_handler(**kwargs):
        raise TimeoutError("upstream API timed out")

    crashing = ToolRegistry(
        [
            Tool(
                name="flaky_tool",
                description="Always fails.",
                input_schema={"type": "object", "properties": {}, "required": []},
                handler=exploding_handler,
            )
        ]
    )
    outcome = crashing.dispatch("flaky_tool", {})
    assert outcome.is_error is True
    assert "execution_failed" in outcome.content
    assert "TimeoutError" in outcome.content


def test_dispatch_never_raises_on_malformed_arguments(registry):
    outcome = registry.dispatch("search_destination_guide", {"query": None})
    assert outcome.is_error is True


# --- schema type enforcement ---------------------------------------------


def test_a_wrongly_typed_argument_is_rejected_with_the_expected_type(registry):
    """The model needs to be told the type it should have sent, to retry."""
    outcome = registry.dispatch("get_weather_forecast", {"city": 42, "date_or_month": "May"})
    assert outcome.is_error is True
    assert "invalid_input" in outcome.content
    assert "must be of type string" in outcome.content
    assert "int" in outcome.content


def test_a_null_argument_is_rejected_as_a_type_error(registry):
    outcome = registry.dispatch("search_destination_guide", {"query": None})
    assert outcome.is_error is True
    assert "must be of type string" in outcome.content


def test_a_boolean_does_not_pass_as_a_number():
    """`bool` is an `int` subclass in Python; JSON schema does not agree."""
    numeric = ToolRegistry(
        [
            Tool(
                name="scale",
                description="Scales.",
                input_schema={
                    "type": "object",
                    "properties": {"factor": {"type": "number"}},
                    "required": ["factor"],
                },
                handler=lambda **kw: "ok",
            )
        ]
    )
    assert numeric.dispatch("scale", {"factor": True}).is_error is True
    assert numeric.dispatch("scale", {"factor": 1.5}).is_error is False
    assert numeric.dispatch("scale", {"factor": 2}).is_error is False


def test_an_out_of_enum_value_is_rejected():
    enumerated = ToolRegistry(
        [
            Tool(
                name="pick",
                description="Picks.",
                input_schema={
                    "type": "object",
                    "properties": {"unit": {"type": "string", "enum": ["c", "f"]}},
                    "required": ["unit"],
                },
                handler=lambda **kw: "ok",
            )
        ]
    )
    outcome = enumerated.dispatch("pick", {"unit": "kelvin"})
    assert outcome.is_error is True
    assert "must be one of: c, f" in outcome.content
    assert enumerated.dispatch("pick", {"unit": "f"}).is_error is False


def test_a_schema_without_a_declared_type_is_left_to_the_handler():
    """Absent `type` means we make no claim -- we must not invent one."""
    untyped = ToolRegistry(
        [
            Tool(
                name="anything",
                description="Takes anything.",
                input_schema={
                    "type": "object",
                    "properties": {"value": {"description": "no type declared"}},
                    "required": ["value"],
                },
                handler=lambda **kw: "ok",
            )
        ]
    )
    assert untyped.dispatch("anything", {"value": {"nested": 1}}).is_error is False
