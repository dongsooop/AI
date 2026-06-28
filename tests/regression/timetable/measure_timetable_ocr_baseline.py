#!/usr/bin/env python3
import argparse
import importlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_CASES_PATH = ROOT_DIR / "tests" / "regression" / "timetable" / "timetable_ocr_performance_cases.json"
DEFAULT_REPORT_PATH = ROOT_DIR / "tests" / "reports" / "timetable" / "timetable_ocr_performance_baseline.json"
SUITE = "timetable_ocr_performance_baseline"
SERVICE = "timetable"


def load_cases(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"no timetable OCR performance cases found in {path}")
    return payload


def write_report(out_path: Path, output: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def make_summary(
    status: str,
    total: int,
    passed: int,
    failed: int,
    skipped: int,
    metrics: Dict[str, Any],
    errors: List[str],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "suite": SUITE,
        "service": SERVICE,
        "status": status,
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "metrics": metrics,
        "errors": errors,
    }


def write_skipped_report(out_path: Path, reason: str, cases_path: Path) -> None:
    summary = make_summary(
        status="skipped",
        total=0,
        passed=0,
        failed=0,
        skipped=1,
        metrics={},
        errors=[],
    )
    output = {
        "schema_version": 1,
        "suite": SUITE,
        "service": SERVICE,
        "summary": summary,
        "cases_path": str(cases_path),
        "skipped": {
            "reason": reason,
        },
        "case_results": [],
    }
    write_report(out_path, output)
    print(f"[SKIP] {reason}")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"report={out_path}")


def load_ocr_dependencies(out_path: Path, cases_path: Path):
    try:
        cv2 = importlib.import_module("cv2")
        pytesseract = importlib.import_module("pytesseract")
        ocr_engine = importlib.import_module("image_analysis.ocr_engine")
    except ModuleNotFoundError as exc:
        write_skipped_report(out_path, f"timetable OCR baseline dependency is missing: {exc.name}", cases_path)
        raise SystemExit(0) from exc

    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        write_skipped_report(out_path, f"Tesseract runtime is unavailable: {exc}", cases_path)
        raise SystemExit(0) from exc

    return cv2, ocr_engine


def percentile(values: List[int], percentile_value: float) -> Optional[int]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((percentile_value / 100.0) * len(ordered)) - 1))
    return ordered[index]


def average_int(values: List[int]) -> Optional[int]:
    if not values:
        return None
    return int(round(sum(values) / len(values)))


def resolve_case_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def extract_run_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    diagnostics = report.get("diagnostics", {})
    runtime = diagnostics.get("runtime", {})
    grid = diagnostics.get("grid", {})
    ocr = diagnostics.get("ocr", {})
    schedules = report.get("schedules", [])
    return {
        "total_duration_ms": int(runtime.get("total_duration_ms", 0) or 0),
        "grid_detection_duration_ms": int(runtime.get("grid_detection_duration_ms", 0) or 0),
        "ocr_duration_ms": int(runtime.get("ocr_duration_ms", 0) or 0),
        "schedule_count": len(schedules),
        "failure_reason": diagnostics.get("failure_reason"),
        "x_line_count": int(grid.get("x_line_count", 0) or 0),
        "y_line_count": int(grid.get("y_line_count", 0) or 0),
        "accepted_cell_count": int(ocr.get("accepted_cells", 0) or 0),
        "total_cell_count": int(ocr.get("total_cells", 0) or 0),
        "ocr_task_cell_count": int(ocr.get("ocr_task_cells", 0) or 0),
        "text_cell_count": int(ocr.get("text_cells", 0) or 0),
        "skipped_empty_cell_count": int(ocr.get("skipped_empty_cells", 0) or 0),
        "fallback_cell_count": int(ocr.get("fallback_cells", 0) or 0),
    }


def summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_durations = [run["total_duration_ms"] for run in runs]
    grid_durations = [run["grid_detection_duration_ms"] for run in runs]
    ocr_durations = [run["ocr_duration_ms"] for run in runs]
    schedule_counts = [run["schedule_count"] for run in runs]
    failure_reasons = sorted({run["failure_reason"] for run in runs if run.get("failure_reason")})
    first_run = runs[0] if runs else {}
    return {
        "average_total_duration_ms": average_int(total_durations),
        "max_total_duration_ms": max(total_durations) if total_durations else None,
        "p95_total_duration_ms": percentile(total_durations, 95.0),
        "average_grid_detection_duration_ms": average_int(grid_durations),
        "max_grid_detection_duration_ms": max(grid_durations) if grid_durations else None,
        "average_ocr_duration_ms": average_int(ocr_durations),
        "max_ocr_duration_ms": max(ocr_durations) if ocr_durations else None,
        "average_schedule_count": average_int(schedule_counts),
        "max_schedule_count": max(schedule_counts) if schedule_counts else None,
        "failure_reasons": failure_reasons,
        "x_line_count": first_run.get("x_line_count"),
        "y_line_count": first_run.get("y_line_count"),
        "accepted_cell_count": first_run.get("accepted_cell_count"),
        "total_cell_count": first_run.get("total_cell_count"),
        "ocr_task_cell_count": first_run.get("ocr_task_cell_count"),
        "text_cell_count": first_run.get("text_cell_count"),
        "skipped_empty_cell_count": first_run.get("skipped_empty_cell_count"),
        "fallback_cell_count": first_run.get("fallback_cell_count"),
    }


def evaluate_case(case: Dict[str, Any], repeat: int, warmup: int, cv2, ocr_engine) -> Dict[str, Any]:
    case_id = str(case.get("id") or case.get("path") or "unnamed_case")
    path_value = str(case.get("path") or "")
    image_path = resolve_case_path(path_value)
    result: Dict[str, Any] = {
        "id": case_id,
        "path": path_value,
        "absolute_path": str(image_path),
    }

    if not image_path.exists():
        result.update({
            "status": "skipped",
            "reason": "image_not_found",
            "runs": [],
            "metrics": {},
        })
        return result

    image = cv2.imread(str(image_path))
    if image is None:
        result.update({
            "status": "failed",
            "errors": ["image_decode_failed"],
            "runs": [],
            "metrics": {},
        })
        return result

    runs: List[Dict[str, Any]] = []
    errors: List[str] = []
    for warmup_index in range(warmup):
        try:
            ocr_engine.extract_schedule_runtime_report(image)
        except Exception as exc:
            errors.append(f"warmup_{warmup_index + 1}:{type(exc).__name__}:{exc}")

    for run_index in range(repeat):
        try:
            report = ocr_engine.extract_schedule_runtime_report(image)
        except Exception as exc:
            errors.append(f"run_{run_index + 1}:{type(exc).__name__}:{exc}")
            continue
        run_metrics = extract_run_metrics(report)
        run_metrics["run"] = run_index + 1
        runs.append(run_metrics)

    if errors and not runs:
        status = "failed"
    elif errors:
        status = "failed"
    else:
        status = "passed"

    result.update({
        "status": status,
        "runs": runs,
        "metrics": summarize_runs(runs),
    })
    if errors:
        result["errors"] = errors
    return result


def apply_runtime_profile(ocr_engine, profile: str, ocr_workers: Optional[int]) -> Dict[str, Any]:
    original_ocr_workers = int(getattr(ocr_engine, "OCR_THREAD_WORKERS", 0) or 0)
    effective_ocr_workers = ocr_workers
    if effective_ocr_workers is None and profile == "oci-constrained":
        effective_ocr_workers = 2

    if effective_ocr_workers is not None:
        if effective_ocr_workers < 1:
            raise ValueError("--ocr-workers must be at least 1")
        setattr(ocr_engine, "OCR_THREAD_WORKERS", effective_ocr_workers)

    return {
        "profile": profile,
        "original_ocr_thread_workers": original_ocr_workers,
        "effective_ocr_thread_workers": int(getattr(ocr_engine, "OCR_THREAD_WORKERS", 0) or 0),
        "process_cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }


def aggregate_metrics(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    runs = [
        run
        for result in case_results
        if result.get("status") == "passed"
        for run in result.get("runs", [])
    ]
    total_durations = [run["total_duration_ms"] for run in runs]
    grid_durations = [run["grid_detection_duration_ms"] for run in runs]
    ocr_durations = [run["ocr_duration_ms"] for run in runs]
    failure_reasons = [
        run["failure_reason"]
        for run in runs
        if run.get("failure_reason")
    ]
    return {
        "measured_run_count": len(runs),
        "average_total_duration_ms": average_int(total_durations),
        "max_total_duration_ms": max(total_durations) if total_durations else None,
        "p95_total_duration_ms": percentile(total_durations, 95.0),
        "average_grid_detection_duration_ms": average_int(grid_durations),
        "max_grid_detection_duration_ms": max(grid_durations) if grid_durations else None,
        "average_ocr_duration_ms": average_int(ocr_durations),
        "max_ocr_duration_ms": max(ocr_durations) if ocr_durations else None,
        "total_extracted_schedule_count": sum(int(run.get("schedule_count", 0)) for run in runs),
        "total_cell_count": sum(int(run.get("total_cell_count", 0)) for run in runs),
        "ocr_task_cell_count": sum(int(run.get("ocr_task_cell_count", 0)) for run in runs),
        "skipped_empty_cell_count": sum(int(run.get("skipped_empty_cell_count", 0)) for run in runs),
        "failure_count": len(failure_reasons),
        "failure_reasons": sorted(set(failure_reasons)),
        "skipped_case_count": sum(1 for result in case_results if result.get("status") == "skipped"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a timetable OCR runtime performance baseline report")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH), help="performance cases path")
    parser.add_argument("--out", default=str(DEFAULT_REPORT_PATH), help="output report path")
    parser.add_argument("--repeat", type=int, default=1, help="number of runs per image; keep low on OCI")
    parser.add_argument("--warmup", type=int, default=0, help="unmeasured warmup runs per image")
    parser.add_argument(
        "--profile",
        choices=("local", "production", "oci-constrained"),
        default="production",
        help="measurement profile; oci-constrained defaults OCR cell workers to 2",
    )
    parser.add_argument(
        "--ocr-workers",
        type=int,
        default=None,
        help="override image_analysis.ocr_engine.OCR_THREAD_WORKERS for this measurement process",
    )
    args = parser.parse_args()

    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be at least 0")

    out_path = Path(args.out)
    cases_path = Path(args.cases)
    case_payload = load_cases(cases_path)
    cases = case_payload["cases"]
    cv2, ocr_engine = load_ocr_dependencies(out_path, cases_path)
    runtime_profile = apply_runtime_profile(ocr_engine, args.profile, args.ocr_workers)

    case_results = [evaluate_case(case, args.repeat, args.warmup, cv2, ocr_engine) for case in cases]
    errors = [
        f"{result['id']}:{error}"
        for result in case_results
        for error in result.get("errors", [])
    ]
    failed = sum(1 for result in case_results if result.get("status") == "failed")
    skipped = sum(1 for result in case_results if result.get("status") == "skipped")
    passed = len(case_results) - failed - skipped
    status = "failed" if failed else "passed"
    summary = make_summary(
        status=status,
        total=len(case_results),
        passed=passed,
        failed=failed,
        skipped=skipped,
        metrics=aggregate_metrics(case_results),
        errors=errors,
    )
    output = {
        "schema_version": 1,
        "suite": SUITE,
        "service": SERVICE,
        "summary": summary,
        "cases_path": str(cases_path),
        "repeat": args.repeat,
        "warmup": args.warmup,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "runtime_profile": runtime_profile,
        "performance_targets": case_payload.get("performance_targets", {}),
        "case_results": case_results,
    }
    write_report(out_path, output)

    for result in case_results:
        if result.get("status") == "skipped":
            print(f"[SKIP] {result['id']} {result.get('reason')}")
    if failed:
        for error in errors:
            print(f"[FAIL] {error}")
        print(json.dumps(summary, ensure_ascii=False))
        print(f"report={out_path}")
        return 1

    print("[OK] timetable OCR performance baseline")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
