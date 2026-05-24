"""Structured logging bootstrap (console or JSON)."""

from __future__ import annotations

import logging

import structlog


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
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def resolve_log_level(name: str) -> int:
    return getattr(logging, name.upper(), logging.INFO)
