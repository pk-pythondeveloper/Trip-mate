"""Structured JSON logging.

Every tool call, agent decision, and error is emitted as one JSON object per
line so traces are greppable and machine-readable. Human-facing output is the
CLI's job, not the logger's.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Attributes present on every LogRecord; anything else was passed via `extra`
# and is therefore part of our structured payload.
_STANDARD_ATTRS = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"asctime", "message", "taskName"}


# Structured fields whose *name* implies a credential. Matched on whole
# underscore-separated words, not substrings: a bare `in` test also redacts
# `input_tokens` ("token") and `chunk_ids` -- silently gutting the very trace
# fields the logs exist for.
_SENSITIVE_WORDS = {"apikey", "key", "token", "secret", "password", "credential", "authorization"}


def _is_sensitive(field_name: str) -> bool:
    words = field_name.lower().replace("-", "_").split("_")
    return any(w in _SENSITIVE_WORDS for w in words)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS:
                payload[key] = "[REDACTED]" if _is_sensitive(key) else value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def force_utf8_output() -> None:
    """Configure stdout/stderr for UTF-8 on Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def resolve_level(level: str, *, fallback: int = logging.INFO) -> int:
    """Turn a level name into a level number, tolerating a bad one.

    `TRIPMATE_LOG_LEVEL=verbose` is a plausible typo, and a logging
    misconfiguration bringing down the whole CLI with a traceback is the wrong
    trade. Fall back to INFO and say so once we have somewhere to say it.
    """
    resolved = logging.getLevelName(str(level).strip().upper())
    return resolved if isinstance(resolved, int) else fallback


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Initialize logging with JSON formatter."""
    resolved = resolve_level(level)
    root = logging.getLogger()
    root.setLevel(resolved)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(JsonFormatter())
    root.addHandler(stream)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

    if resolve_level(level, fallback=-1) == -1:
        logging.getLogger(__name__).warning(
            "logging.unknown_level", extra={"requested": level, "using": "INFO"}
        )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
