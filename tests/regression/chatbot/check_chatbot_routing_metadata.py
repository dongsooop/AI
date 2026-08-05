#!/usr/bin/env python3
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from LLM.OSS.routing import RoutingMetadata


def check_routing_metadata() -> list[str]:
    errors = []

    metadata = RoutingMetadata(
        intent="oss",
        stage="oss_fast_path",
        tool="confident_search_answer",
        confidence=0.85,
        decision_source="retrieval",
        source_urls=("https://www.dongyang.ac.kr/contact",),
    )
    log_fields = metadata.to_log_fields()

    expected_fields = {
        "intent": "oss",
        "route_stage": "oss_fast_path",
        "tool": "confident_search_answer",
        "confidence": 0.85,
        "decision_source": "retrieval",
        "source_count": 1,
        "has_source": True,
        "fallback": False,
        "fallback_reason": None,
        "llm_required": False,
    }
    if log_fields != expected_fields:
        errors.append(f"log_fields_unexpected:{log_fields}")
    if "source_urls" in log_fields or "https://" in str(log_fields):
        errors.append("raw_source_url_exposed_in_log_fields")

    fallback = RoutingMetadata(
        intent="oss",
        stage="grounded_llm",
        tool="rag_context_required",
        confidence=0.5,
        decision_source="retrieval",
        fallback_used=True,
        fallback_reason="no_deterministic_tool_match",
        llm_required=True,
    )
    fallback_fields = fallback.to_log_fields()
    if not fallback_fields["fallback"] or not fallback_fields["llm_required"]:
        errors.append(f"fallback_flags_unexpected:{fallback_fields}")

    try:
        metadata.intent = "fast"
        errors.append("metadata_is_mutable")
    except FrozenInstanceError:
        pass

    invalid_cases = (
        {"intent": "", "stage": "classified"},
        {"intent": "oss", "stage": ""},
        {"intent": "oss", "stage": "classified", "decision_source": ""},
        {"intent": "oss", "stage": "classified", "confidence": -0.01},
        {"intent": "oss", "stage": "classified", "confidence": 1.01},
    )
    for index, kwargs in enumerate(invalid_cases):
        try:
            RoutingMetadata(**kwargs)
            errors.append(f"invalid_metadata_accepted:{index}")
        except ValueError:
            pass

    return errors


def main() -> int:
    errors = check_routing_metadata()
    print(json.dumps({"ok": not errors, "errors": errors}, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
