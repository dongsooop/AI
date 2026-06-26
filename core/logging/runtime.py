from __future__ import annotations

from enum import Enum
from typing import Any, Final


class RuntimeComponent(str, Enum):
    CHATBOT = "chatbot"
    OCR = "ocr"
    TEXT_FILTERING = "text_filtering"


class RuntimeOperation(str, Enum):
    REQUEST = "request"
    RETRIEVAL = "retrieval"
    LLM = "llm"
    GRID_DETECTION = "grid_detection"
    OCR = "ocr"
    RULE_FILTER = "rule_filter"
    ML_FILTER = "ml_filter"


class RuntimeStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    FALLBACK = "fallback"
    SKIPPED = "skipped"


RUNTIME_LOG_FIELDS: Final[tuple[str, ...]] = (
    "event",
    "component",
    "operation",
    "status",
    "duration_ms",
    "result_count",
    "fallback",
    "fallback_reason",
    "error_code",
)


def _format_log_value(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    if not text:
        return "none"
    return text.replace(" ", "_").replace("=", ":")


def runtime_log_message(event: str, **fields: Any) -> str:
    all_fields = {"event": event, **fields}
    ordered_keys = [k for k in RUNTIME_LOG_FIELDS if k in all_fields]
    extra_keys = [k for k in all_fields.keys() if k not in RUNTIME_LOG_FIELDS]
    keys = ordered_keys + extra_keys
    return " ".join(f"{key}={_format_log_value(all_fields[key])}" for key in keys)