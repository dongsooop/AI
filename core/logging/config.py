from __future__ import annotations

import logging
import os
from typing import Final

from core.logging.context import get_request_id


_DEFAULT_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)s | %(service)s | %(request_id)s | %(name)s | %(message)s"
)
_CONFIGURED = False
_SERVICE_NAME = "dongsooop"


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        record.service = _SERVICE_NAME
        return True


def configure_logging(service_name: str, level: str | None = None) -> None:
    global _CONFIGURED, _SERVICE_NAME

    _SERVICE_NAME = service_name
    log_level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if _CONFIGURED:
        for handler in root_logger.handlers:
            handler.setLevel(log_level)
        return

    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    handler.addFilter(RequestContextFilter())

    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
