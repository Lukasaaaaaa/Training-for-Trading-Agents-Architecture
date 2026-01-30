"""Logging configuration for the trading orchestrator."""

import sys
import structlog
from structlog.typing import EventDict
from typing import Any
from .config import settings


def add_workflow_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add workflow context to log entries."""
    if "workflow_id" in event_dict:
        event_dict["workflow_id"] = event_dict["workflow_id"]
    if "agent" in event_dict:
        event_dict["agent"] = event_dict["agent"]
    if "stage" in event_dict:
        event_dict["stage"] = event_dict["stage"]
    return event_dict


def configure_logging() -> None:
    """Configure structured logging for the application."""

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        add_workflow_context,
    ]

    if settings.log_format == "json":
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ]
    else:
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=True)
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog.stdlib.logging, settings.log_level.upper(), 20)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """Get a configured logger instance."""
    return structlog.get_logger(name)
