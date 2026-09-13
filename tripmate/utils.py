"""Input validation utilities."""

from __future__ import annotations

import re


def sanitize_query(query: str) -> str:
    """Remove null bytes and control characters from input."""
    if not isinstance(query, str):
        return ""
    
    query = query.replace('\x00', '')
    query = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', query)
    
    return query
