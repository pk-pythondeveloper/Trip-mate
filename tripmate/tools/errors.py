"""Tool-layer error types.

Design rule: a tool failure must never propagate out of the tool layer.
Raising past the registry kills the agent loop; returning a `tool_result` with
`is_error: true` lets the model see what went wrong and recover -- by
rephrasing, by trying the other tool, or by telling the user plainly.
"""

from __future__ import annotations


class ToolError(Exception):
    """Base class for recoverable tool failures reported back to the model."""

    code = "tool_error"


class ToolNotFoundError(ToolError):
    """The model called a tool name that is not registered."""

    code = "unknown_tool"


class ToolInputError(ToolError):
    """Arguments failed validation before the tool ever ran."""

    code = "invalid_input"


class ToolExecutionError(ToolError):
    """The tool raised while executing."""

    code = "execution_failed"
