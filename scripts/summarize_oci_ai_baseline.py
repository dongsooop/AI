#!/usr/bin/env python3
"""Create JSON and Markdown summaries without raw prompts, tokens, URLs, or logs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = ROOT_DIR / "tests" / "reports" / "operations" / "oci_ai_runtime_baseline.json"
DEFAULT_OCR = ROOT_DIR / "tests" / "reports" / "timetable" / "timetable_ocr_performance_baseline.json"
DEFAULT_RAG = ROOT_DIR / "tests" / "reports" / "chatbot" / "rag_eval_report.json"
DEFAULT_TEXT_FILTER = ROOT_DIR / "tests" / "reports" / "text_filtering" / "text_filter_performance_baseline.json"
DEFAULT_JSON_OUT = ROOT_DIR / "tests" / "reports" / "operations" / "oci_ai_baseline_summary.json"
DEFAULT_MD_OUT = ROOT_DIR / "tests" / "reports" / "operations" / "oci_ai_baseline_summary.md"


def load_optional(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "not_found"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid_json:{type(exc).__name__}"
    if not isinstance(payload, dict):
        return None, "root_not_object"
    return payload, None


def report_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", {})
    return summary if isinstance(summary, dict) else {}


def rag_environment_valid(payload: dict[str, Any]) -> bool:
    results = payload.get("results", [])
    if not isinstance(results, list):
        return False
    for result in results:
        errors = result.get("errors", {}) if isinstance(result, dict) else {}
        if isinstance(errors, dict) and errors.get("import"):
            return False
    return True


def baseline_rows(
    ocr: dict[str, Any] | None,
    rag: dict[str, Any] | None,
    text_filter: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if ocr:
        summary = report_summary(ocr)
        metrics = summary.get("metrics", {})
        task_cells = int(metrics.get("ocr_task_cell_count", 0) or 0)
        fallback_cells = int(metrics.get("fallback_cell_count", 0) or 0)
        rows.append({
            "service": "ocr",
            "scenario": "offline-baseline",
            "concurrency": None,
            "status": summary.get("status"),
            "p50_ms": metrics.get("p50_total_duration_ms"),
            "p95_ms": metrics.get("p95_total_duration_ms"),
            "max_ms": metrics.get("max_total_duration_ms"),
            "peak_rss_mb": metrics.get("peak_rss_mb"),
            "error_rate": None,
            "fallback_rate": round(fallback_cells / task_cells, 4) if task_cells else None,
        })
    if rag:
        summary = report_summary(rag)
        valid = rag_environment_valid(rag)
        rows.append({
            "service": "chatbot-retrieval",
            "scenario": "offline-quality",
            "concurrency": None,
            "status": summary.get("status") if valid else "environment_invalid",
            "p50_ms": summary.get("p50_retrieval_latency_ms") if valid else None,
            "p95_ms": summary.get("p95_retrieval_latency_ms") if valid else None,
            "max_ms": summary.get("max_retrieval_latency_ms") if valid else None,
            "peak_rss_mb": summary.get("peak_rss_mb") if valid else None,
            "error_rate": None,
            "fallback_rate": None,
            "top1_url_accuracy": summary.get("top1_url_accuracy") if valid else None,
            "top3_url_accuracy": summary.get("top3_url_accuracy") if valid else None,
        })
    if text_filter:
        summary = report_summary(text_filter)
        metrics = summary.get("metrics", {})
        rows.append({
            "service": "text-filter",
            "scenario": "offline-baseline",
            "concurrency": None,
            "status": summary.get("status"),
            "p50_ms": metrics.get("request_p50_latency_ms"),
            "p95_ms": metrics.get("request_p95_latency_ms"),
            "max_ms": metrics.get("request_max_latency_ms"),
            "peak_rss_mb": metrics.get("peak_rss_mb"),
            "error_rate": None,
            "fallback_rate": None,
            "false_positive_count": metrics.get("false_positive_count"),
            "false_negative_count": metrics.get("false_negative_count"),
        })
    return rows


def load_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        scenario = str(payload.get("scenario", "unknown"))
        concurrency = payload.get("concurrency_per_service")
        summaries = payload.get("summary", {})
        if not isinstance(summaries, dict):
            continue
        for service, summary in summaries.items():
            if not isinstance(summary, dict):
                continue
            latency = summary.get("latency_ms", {})
            rows.append({
                "service": service,
                "scenario": scenario,
                "concurrency": concurrency,
                "cache_state": payload.get("cache_state"),
                "query_variants": payload.get("chatbot_query_variant_count"),
                "status": payload.get("status"),
                "p50_ms": latency.get("p50"),
                "p95_ms": latency.get("p95"),
                "max_ms": latency.get("max"),
                "peak_rss_mb": None,
                "error_rate": summary.get("error_rate"),
                "fallback_rate": None,
                "throughput_rps": summary.get("throughput_rps"),
                "stop_reason": payload.get("stop_reason"),
            })
    return rows


def sanitized_runtime(runtime: dict[str, Any] | None) -> dict[str, Any]:
    if not runtime:
        return {}
    host = runtime.get("host", {})
    profile = runtime.get("deployment_profile", {})
    processes = runtime.get("processes", {})
    logs = runtime.get("logs", {})
    runtime_events: dict[str, Any] = {}
    if isinstance(logs, dict):
        for service, log in logs.items():
            if isinstance(log, dict):
                runtime_events[service] = log.get("runtime_events", {})
    return {
        "host": {
            "architecture": host.get("architecture"),
            "cpu_model": host.get("cpu_model"),
            "oci_ocpu_count": host.get("oci_ocpu_count"),
            "ram_mb": host.get("ram_mb"),
        },
        "deployment_profile": profile,
        "processes": {
            name: {
                "cpu_percent": metrics.get("cpu_percent"),
                "peak_rss_mb": metrics.get("peak_rss_mb"),
            }
            for name, metrics in processes.items()
            if isinstance(metrics, dict)
        },
        "runtime_events": runtime_events,
    }


def markdown_value(value: Any) -> str:
    return "-" if value is None else str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# OCI AI Performance Baseline Summary",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "Raw prompts, bearer tokens, base URLs, process IDs, and raw log lines are intentionally excluded.",
        "",
        "| Service | Scenario | Concurrency | Cache | Query variants | Status | p50 ms | p95 ms | Max ms | Peak RSS MB | Error rate | Fallback rate |",
        "| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        lines.append("| " + " | ".join(markdown_value(row.get(key)) for key in (
            "service", "scenario", "concurrency", "cache_state", "query_variants", "status", "p50_ms", "p95_ms", "max_ms",
            "peak_rss_mb", "error_rate", "fallback_rate",
        )) + " |")
    lines.extend(["", "## Missing or invalid inputs", ""])
    missing = summary.get("input_status", {})
    problems = [f"- `{name}`: {status}" for name, status in missing.items() if status != "loaded"]
    lines.extend(problems or ["- None"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize OCI AI baseline reports without sensitive payloads")
    parser.add_argument("--runtime", default=str(DEFAULT_RUNTIME))
    parser.add_argument("--ocr", default=str(DEFAULT_OCR))
    parser.add_argument("--rag", default=str(DEFAULT_RAG))
    parser.add_argument("--text-filter", default=str(DEFAULT_TEXT_FILTER))
    parser.add_argument("--load-report", action="append", default=[])
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MD_OUT))
    args = parser.parse_args()

    named_paths = {
        "runtime": Path(args.runtime),
        "ocr": Path(args.ocr),
        "rag": Path(args.rag),
        "text_filter": Path(args.text_filter),
    }
    payloads: dict[str, dict[str, Any] | None] = {}
    input_status: dict[str, str] = {}
    for name, path in named_paths.items():
        payload, error = load_optional(path)
        payloads[name] = payload
        input_status[name] = error or "loaded"
    if payloads["rag"] and not rag_environment_valid(payloads["rag"]):
        input_status["rag"] = "environment_invalid"

    load_payloads: list[dict[str, Any]] = []
    for index, value in enumerate(args.load_report, start=1):
        payload, error = load_optional(Path(value))
        input_status[f"load_{index}"] = error or "loaded"
        if payload:
            load_payloads.append(payload)

    rows = baseline_rows(payloads["ocr"], payloads["rag"], payloads["text_filter"])
    rows.extend(load_rows(load_payloads))
    summary = {
        "schema_version": 1,
        "suite": "oci_ai_baseline_summary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_status": input_status,
        "rows": rows,
        "runtime": sanitized_runtime(payloads["runtime"]),
        "privacy": {
            "raw_prompts_included": False,
            "tokens_included": False,
            "base_urls_included": False,
            "process_ids_included": False,
            "raw_log_lines_included": False,
        },
    }
    json_out = Path(args.json_out)
    markdown_out = Path(args.markdown_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_out.write_text(render_markdown(summary), encoding="utf-8")
    print("[OK] OCI AI baseline summary")
    print(json.dumps({"input_status": input_status, "row_count": len(rows)}, ensure_ascii=False))
    print(f"json_report={json_out}")
    print(f"markdown_report={markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
