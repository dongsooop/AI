#!/usr/bin/env python3
import argparse
import importlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_CASES_PATH = ROOT_DIR / "tests" / "regression" / "timetable" / "timetable_ocr_performance_cases.json"
DEFAULT_REPORT_PATH = ROOT_DIR / "tests" / "reports" / "timetable" / "timetable_ocr_empty_cell_skip_analysis.json"
DEFAULT_THRESHOLDS = "0.001,0.002,0.003,0.005,0.008,0.01,0.015,0.02,0.03,0.05,0.075,0.1"
SUITE = "timetable_ocr_empty_cell_skip_analysis"
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
        write_skipped_report(out_path, f"timetable OCR empty-cell analysis dependency is missing: {exc.name}", cases_path)
        raise SystemExit(0) from exc

    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        write_skipped_report(out_path, f"Tesseract runtime is unavailable: {exc}", cases_path)
        raise SystemExit(0) from exc

    return cv2, ocr_engine


def parse_thresholds(value: str) -> List[float]:
    thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not thresholds:
        raise ValueError("--thresholds must include at least one value")
    if any(threshold < 0 for threshold in thresholds):
        raise ValueError("--thresholds values must be non-negative")
    return thresholds


def resolve_case_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def prepare_cell_candidates(img, cv2, ocr_engine) -> Dict[str, Any]:
    base_gray = ocr_engine._make_base_gray(img)
    x_lines, vertical_mask = ocr_engine._vertical_lines(base_gray)
    x_lines = ocr_engine._prune_x_lines(x_lines, base_gray)
    y_lines, horizontal_mask = ocr_engine._horizontal_lines(base_gray, vertical_mask=vertical_mask)

    if ocr_engine.TRIM_OUTER:
        if len(x_lines) > 2 and (x_lines[1] - x_lines[0] < ocr_engine.TRIM_X_GAP):
            x_lines = x_lines[1:]
        if len(x_lines) > 2 and (x_lines[-1] - x_lines[-2] < ocr_engine.TRIM_X_GAP):
            x_lines = x_lines[:-1]
        if len(y_lines) > 2 and (y_lines[1] - y_lines[0] < ocr_engine.TRIM_Y_GAP):
            y_lines = y_lines[1:]
        if len(y_lines) > 2 and (y_lines[-1] - y_lines[-2] < ocr_engine.TRIM_Y_GAP):
            y_lines = y_lines[:-1]

    cells: List[Dict[str, Any]] = []
    tasks: List[bytes] = []
    task_indexes: List[int] = []
    for row, col, roi_gray in ocr_engine._iterate_cells_autogrid(base_gray, x_lines, y_lines):
        if row == 0:
            continue
        cell = {
            "row": row,
            "col": col + 1,
            "foreground_density": ocr_engine._foreground_density(roi_gray),
            "ocr_status": "not_run",
        }
        ok, buf = cv2.imencode(".png", roi_gray)
        if ok:
            tasks.append(buf.tobytes())
            task_indexes.append(len(cells))
        else:
            cell["ocr_status"] = "roi_encode_failed"
        cells.append(cell)

    if tasks:
        max_workers = min(ocr_engine.OCR_THREAD_WORKERS, len(tasks)) or 1
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="empty_cell_analysis") as executor:
            for index, lines in zip(task_indexes, executor.map(ocr_engine._ocr_task, tasks)):
                cells[index]["lines"] = lines
                if not lines:
                    cells[index]["ocr_status"] = "empty"
                    continue
                course = lines[0]
                professor = lines[1] if len(lines) > 1 else ""
                room = lines[2] if len(lines) > 2 else ""
                if len(course) > 15 or len(professor) > 8 or len(room) > 10:
                    cells[index]["ocr_status"] = "rejected_field_too_long"
                elif ocr_engine._is_valid_course(course) and ocr_engine._is_valid_professor(professor):
                    cells[index]["ocr_status"] = "accepted"
                else:
                    cells[index]["ocr_status"] = "rejected_invalid_course_or_professor"

    return {
        "grid": {
            "x_line_count": len(x_lines),
            "y_line_count": len(y_lines),
            "cell_columns": max(0, len(x_lines) - 1),
            "cell_rows": max(0, len(y_lines) - 1),
        },
        "cells": cells,
    }


def summarize_thresholds(cells: List[Dict[str, Any]], thresholds: List[float], sample_limit: int) -> List[Dict[str, Any]]:
    accepted_cells = [cell for cell in cells if cell.get("ocr_status") == "accepted"]
    text_cells = [cell for cell in cells if cell.get("ocr_status") not in {"empty", "roi_encode_failed", "not_run"}]
    results: List[Dict[str, Any]] = []
    for threshold in thresholds:
        skipped = [cell for cell in cells if float(cell["foreground_density"]) <= threshold]
        skipped_accepted = [cell for cell in skipped if cell.get("ocr_status") == "accepted"]
        skipped_text = [cell for cell in skipped if cell.get("ocr_status") not in {"empty", "roi_encode_failed", "not_run"}]
        results.append({
            "threshold": threshold,
            "would_skip_cell_count": len(skipped),
            "would_skip_cell_ratio": round(len(skipped) / len(cells), 4) if cells else 0.0,
            "would_skip_accepted_cell_count": len(skipped_accepted),
            "would_skip_text_cell_count": len(skipped_text),
            "estimated_ocr_task_count_after_skip": max(0, len(cells) - len(skipped)),
            "estimated_ocr_task_reduction_ratio": round(len(skipped) / len(cells), 4) if cells else 0.0,
            "sample_skipped_accepted_cells": [
                {
                    "row": cell["row"],
                    "col": cell["col"],
                    "foreground_density": cell["foreground_density"],
                }
                for cell in skipped_accepted[:sample_limit]
            ],
        })

    accepted_densities = [float(cell["foreground_density"]) for cell in accepted_cells]
    text_densities = [float(cell["foreground_density"]) for cell in text_cells]
    if accepted_densities:
        results.append({
            "threshold": "accepted_density_floor",
            "min_accepted_foreground_density": min(accepted_densities),
            "max_accepted_foreground_density": max(accepted_densities),
            "min_text_foreground_density": min(text_densities) if text_densities else None,
        })
    return results


def evaluate_case(case: Dict[str, Any], thresholds: List[float], sample_limit: int, cv2, ocr_engine) -> Dict[str, Any]:
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
            "metrics": {},
            "threshold_results": [],
        })
        return result

    image = cv2.imread(str(image_path))
    if image is None:
        result.update({
            "status": "failed",
            "errors": ["image_decode_failed"],
            "metrics": {},
            "threshold_results": [],
        })
        return result

    prepared = prepare_cell_candidates(image, cv2, ocr_engine)
    cells = prepared["cells"]
    threshold_results = summarize_thresholds(cells, thresholds, sample_limit)
    accepted_count = sum(1 for cell in cells if cell.get("ocr_status") == "accepted")
    empty_count = sum(1 for cell in cells if cell.get("ocr_status") == "empty")
    text_count = sum(1 for cell in cells if cell.get("ocr_status") not in {"empty", "roi_encode_failed", "not_run"})
    safe_thresholds = [
        item["threshold"]
        for item in threshold_results
        if isinstance(item["threshold"], float)
        and item["would_skip_accepted_cell_count"] == 0
    ]
    no_text_loss_thresholds = [
        item["threshold"]
        for item in threshold_results
        if isinstance(item["threshold"], float)
        and item["would_skip_text_cell_count"] == 0
    ]

    result.update({
        "status": "passed",
        "grid": prepared["grid"],
        "metrics": {
            "total_cell_count": len(cells),
            "accepted_cell_count": accepted_count,
            "text_cell_count": text_count,
            "empty_cell_count": empty_count,
            "best_safe_threshold_no_accepted_loss": max(safe_thresholds) if safe_thresholds else None,
            "best_safe_threshold_no_text_loss": max(no_text_loss_thresholds) if no_text_loss_thresholds else None,
        },
        "threshold_results": threshold_results,
    })
    return result


def aggregate_metrics(case_results: List[Dict[str, Any]], thresholds: List[float]) -> Dict[str, Any]:
    passed_results = [result for result in case_results if result.get("status") == "passed"]
    threshold_summary: List[Dict[str, Any]] = []
    for threshold in thresholds:
        items = [
            item
            for result in passed_results
            for item in result.get("threshold_results", [])
            if item.get("threshold") == threshold
        ]
        total_cells = sum(int(result.get("metrics", {}).get("total_cell_count", 0)) for result in passed_results)
        skipped = sum(int(item.get("would_skip_cell_count", 0)) for item in items)
        skipped_accepted = sum(int(item.get("would_skip_accepted_cell_count", 0)) for item in items)
        skipped_text = sum(int(item.get("would_skip_text_cell_count", 0)) for item in items)
        threshold_summary.append({
            "threshold": threshold,
            "would_skip_cell_count": skipped,
            "would_skip_cell_ratio": round(skipped / total_cells, 4) if total_cells else 0.0,
            "would_skip_accepted_cell_count": skipped_accepted,
            "would_skip_text_cell_count": skipped_text,
            "safe_no_accepted_loss": skipped_accepted == 0,
        })

    safe_thresholds = [
        item["threshold"]
        for item in threshold_summary
        if item["safe_no_accepted_loss"]
    ]
    no_text_loss_thresholds = [
        item["threshold"]
        for item in threshold_summary
        if item["would_skip_text_cell_count"] == 0
    ]
    return {
        "measured_case_count": len(passed_results),
        "total_cell_count": sum(int(result.get("metrics", {}).get("total_cell_count", 0)) for result in passed_results),
        "accepted_cell_count": sum(int(result.get("metrics", {}).get("accepted_cell_count", 0)) for result in passed_results),
        "text_cell_count": sum(int(result.get("metrics", {}).get("text_cell_count", 0)) for result in passed_results),
        "empty_cell_count": sum(int(result.get("metrics", {}).get("empty_cell_count", 0)) for result in passed_results),
        "best_safe_threshold_no_accepted_loss": max(safe_thresholds) if safe_thresholds else None,
        "best_safe_threshold_no_text_loss": max(no_text_loss_thresholds) if no_text_loss_thresholds else None,
        "threshold_summary": threshold_summary,
        "skipped_case_count": sum(1 for result in case_results if result.get("status") == "skipped"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze candidate blank-cell thresholds before changing timetable OCR runtime")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH), help="performance cases path")
    parser.add_argument("--out", default=str(DEFAULT_REPORT_PATH), help="output report path")
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS, help="comma-separated foreground-density thresholds")
    parser.add_argument("--sample-limit", type=int, default=8, help="max skipped accepted-cell samples per threshold")
    args = parser.parse_args()

    if args.sample_limit < 0:
        raise ValueError("--sample-limit must be non-negative")

    out_path = Path(args.out)
    cases_path = Path(args.cases)
    thresholds = parse_thresholds(args.thresholds)
    case_payload = load_cases(cases_path)
    cases = case_payload["cases"]
    cv2, ocr_engine = load_ocr_dependencies(out_path, cases_path)

    case_results = [evaluate_case(case, thresholds, args.sample_limit, cv2, ocr_engine) for case in cases]
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
        metrics=aggregate_metrics(case_results, thresholds),
        errors=errors,
    )
    output = {
        "schema_version": 1,
        "suite": SUITE,
        "service": SERVICE,
        "summary": summary,
        "cases_path": str(cases_path),
        "thresholds": thresholds,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "This is a dry-run analysis only. It does not change runtime OCR behavior.",
            "A threshold is considered safer when would_skip_accepted_cell_count remains 0 across the sample set.",
        ],
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

    print("[OK] timetable OCR empty-cell skip analysis")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
