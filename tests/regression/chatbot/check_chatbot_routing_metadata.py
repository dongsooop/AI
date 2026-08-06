#!/usr/bin/env python3
import ast
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from LLM.OSS.routing import RoutingMetadata


def check_service_metadata_propagation() -> list[str]:
    errors = []
    service_path = ROOT_DIR / "LLM" / "OSS" / "service.py"
    tree = ast.parse(service_path.read_text(encoding="utf-8"), filename=str(service_path))
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

    summary_calls = []
    cache_return_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id == "_log_chatbot_summary":
            summary_calls.append(node)
        elif node.func.id == "cache_and_return":
            cache_return_calls.append(node)

    if not summary_calls:
        errors.append("chatbot_summary_calls_missing")
    for call in summary_calls:
        has_routing = len(call.args) >= 4 or any(keyword.arg == "routing" for keyword in call.keywords)
        if not has_routing:
            errors.append(f"summary_routing_metadata_missing:line_{call.lineno}")
        legacy_keywords = {
            keyword.arg
            for keyword in call.keywords
            if keyword.arg in {"fallback", "fallback_reason", "direct_answer_route"}
        }
        if legacy_keywords:
            errors.append(f"summary_legacy_metadata_args:line_{call.lineno}:{sorted(legacy_keywords)}")

    if not cache_return_calls:
        errors.append("cache_return_calls_missing")
    for call in cache_return_calls:
        has_routing = len(call.args) >= 2 or any(keyword.arg == "routing" for keyword in call.keywords)
        if not has_routing:
            errors.append(f"cache_return_routing_metadata_missing:line_{call.lineno}")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Return):
            continue
        owner = parents.get(node)
        while owner is not None and not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
            owner = parents.get(owner)
        if owner is None or owner.name != "chat_with_oss":
            continue

        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "cache_and_return"
        ):
            continue

        parent = parents.get(node)
        siblings = next(
            (
                value
                for _, value in ast.iter_fields(parent)
                if isinstance(value, list) and node in value
            ),
            [],
        )
        index = siblings.index(node) if siblings else -1
        previous = siblings[index - 1] if index > 0 else None
        has_summary_before_return = (
            isinstance(previous, ast.Expr)
            and isinstance(previous.value, ast.Call)
            and isinstance(previous.value.func, ast.Name)
            and previous.value.func.id == "_log_chatbot_summary"
        )
        if not has_summary_before_return:
            errors.append(f"return_summary_metadata_missing:line_{node.lineno}")

    return errors


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
    errors = check_routing_metadata() + check_service_metadata_propagation()
    print(json.dumps({"ok": not errors, "errors": errors}, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
