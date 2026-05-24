"""Structured logging bootstrap (console or JSON)."""

from __future__ import annotations

import logging
import re
from typing import Any

import structlog

_SECRET_KEY_PATTERN = re.compile(
    r"(api[_-]?key|token|authorization|password|secret)",
    re.IGNORECASE,
)


def _redact_secrets(_logger: object, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Redact values whose keys look like secrets."""

    def redact_value(key: str, value: Any) -> Any:
        if _SECRET_KEY_PATTERN.search(key):
            return "***REDACTED***"
        if isinstance(value, dict):
            return {k: redact_value(k, v) for k, v in value.items()}
        return value

    return {k: redact_value(k, v) for k, v in event_dict.items()}


def configure_structlog(*, level: int = logging.INFO, json_output: bool = False) -> None:
    renderer: structlog.types.Processor
    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=False),
            structlog.processors.StackInfoRenderer(),
            _redact_secrets,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def resolve_log_level(name: str) -> int:
    return getattr(logging, name.upper(), logging.INFO)
