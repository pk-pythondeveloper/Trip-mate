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


class JsonFormatter(logging.Formatter):
    _SENSITIVE_KEYS = {'api_key', 'key', 'token', 'secret', 'password'}
    
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS:
                if any(s in key.lower() for s in self._SENSITIVE_KEYS):
                    value = "[REDACTED]"
                payload[key] = value
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


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Initialize logging with JSON formatter."""
    root = logging.getLogger()
    root.setLevel(level.upper())
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


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
