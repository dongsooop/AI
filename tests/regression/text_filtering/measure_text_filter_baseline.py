#!/usr/bin/env python3
"""Measure text-filter cold loading, inference latency, quality, CPU, and RSS."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_CASES_PATH = ROOT_DIR / "tests" / "regression" / "text_filtering" / "cases"
DEFAULT_REPORT_PATH = ROOT_DIR / "tests" / "reports" / "text_filtering" / "text_filter_performance_baseline.json"


def percentile(values: list[float], value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((value / 100) * len(ordered)) - 1))
    return round(ordered[index], 2)


def peak_rss_mb() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        rss *= 1024
    return round(rss / (1024 * 1024), 2)


def load_cases(path: Path) -> list[dict[str, Any]]:
    paths = sorted(path.glob("*.json")) if path.is_dir() else [path]
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for case_path in paths:
        payload = json.loads(case_path.read_text(encoding="utf-8"))
        items = payload.get("cases", payload) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise ValueError(f"cases must be a list: {case_path}")
        for item in items:
            case = dict(item)
            case_id = str(case.get("id", ""))
            if not case_id or case_id in seen_ids:
                raise ValueError(f"missing or duplicate case id: {case_id or case_path}")
            seen_ids.add(case_id)
            case["source_file"] = str(case_path.relative_to(ROOT_DIR))
            cases.append(case)
    if not cases:
        raise ValueError(f"no cases found: {path}")
    return cases


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def skipped_report(out_path: Path, cases_path: Path, reason: str) -> int:
    report = {
        "schema_version": 1,
        "suite": "text_filter_performance_baseline",
        "service": "text_filtering",
        "summary": {"status": "skipped", "total": 0, "passed": 0, "failed": 0, "skipped": 1, "metrics": {}},
        "cases_path": str(cases_path),
        "skipped": {"reason": reason},
        "case_results": [],
    }
    write_report(out_path, report)
    print(f"[SKIP] {reason}")
    print(f"report={out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a text-filter performance and quality baseline report")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH), help="case JSON file or directory")
    parser.add_argument("--out", default=str(DEFAULT_REPORT_PATH), help="output report path")
    parser.add_argument("--repeat", type=int, default=3, help="measured repetitions per case")
    parser.add_argument("--warmup", type=int, default=1, help="unmeasured full-suite warmups")
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0:
        raise ValueError("--repeat must be >= 1 and --warmup must be >= 0")

    cases_path = Path(args.cases)
    out_path = Path(args.out)
    cases = load_cases(cases_path)
    try:
        service = importlib.import_module("text_filtering.service")
    except ModuleNotFoundError as exc:
        return skipped_report(out_path, cases_path, f"dependency missing: {exc.name}")
    except Exception as exc:
        return skipped_report(out_path, cases_path, f"service import failed: {type(exc).__name__}: {exc}")

    readiness = service.get_text_filter_readiness()
    if readiness.get("status") != "ready":
        return skipped_report(out_path, cases_path, f"model is not ready: {readiness}")

    service.get_text_filter_model.cache_clear()
    load_started = time.perf_counter()
    service.get_text_filter_model()
    model_load_ms = round((time.perf_counter() - load_started) * 1000, 2)

    for _ in range(args.warmup):
        for case in cases:
            service.analyze_text_labels(str(case.get("text", "")))

    cpu_started = time.process_time()
    case_results: list[dict[str, Any]] = []
    all_request_latencies: list[float] = []
    all_sentence_latencies: list[float] = []
    false_positive_count = 0
    false_negative_count = 0

    for case in cases:
        text = str(case.get("text", ""))
        sentence_count = max(1, len(service.split_sentences(text)))
        latencies: list[float] = []
        labels: list[str] = []
        for _ in range(args.repeat):
            started = time.perf_counter()
            labels = service.analyze_text_labels(text)
            latency_ms = (time.perf_counter() - started) * 1000
            latencies.append(latency_ms)
            all_request_latencies.append(latency_ms)
            all_sentence_latencies.append(latency_ms / sentence_count)

        expected = bool(case.get("expected", {}).get("has_profanity"))
        actual = any(label == "비속어" for label in labels)
        if actual and not expected:
            false_positive_count += 1
        if expected and not actual:
            false_negative_count += 1
        case_results.append({
            "id": case.get("id"),
            "category": case.get("category"),
            "source_file": case.get("source_file"),
            "status": "passed" if actual == expected else "failed",
            "expected_has_profanity": expected,
            "actual_has_profanity": actual,
            "sentence_count": sentence_count,
            "latency_ms": {
                "p50": percentile(latencies, 50),
                "p95": percentile(latencies, 95),
                "max": round(max(latencies), 2),
            },
        })

    failed = false_positive_count + false_negative_count
    metrics = {
        "model_initial_load_ms": model_load_ms,
        "average_inference_per_sentence_ms": round(sum(all_sentence_latencies) / len(all_sentence_latencies), 2),
        "request_p50_latency_ms": percentile(all_request_latencies, 50),
        "request_p95_latency_ms": percentile(all_request_latencies, 95),
        "request_max_latency_ms": round(max(all_request_latencies), 2),
        "process_cpu_seconds": round(time.process_time() - cpu_started, 3),
        "peak_rss_mb": peak_rss_mb(),
        "false_positive_count": false_positive_count,
        "false_negative_count": false_negative_count,
    }
    summary = {
        "schema_version": 1,
        "suite": "text_filter_performance_baseline",
        "service": "text_filtering",
        "status": "passed" if failed == 0 else "failed",
        "total": len(cases),
        "passed": len(cases) - failed,
        "failed": failed,
        "skipped": 0,
        "metrics": metrics,
        "errors": [],
    }
    report = {
        "schema_version": 1,
        "suite": "text_filter_performance_baseline",
        "service": "text_filtering",
        "summary": summary,
        "cases_path": str(cases_path),
        "repeat": args.repeat,
        "warmup": args.warmup,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "case_results": case_results,
    }
    write_report(out_path, report)
    print("[OK] text-filter performance baseline" if failed == 0 else "[WARN] text-filter quality mismatch")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
