#!/usr/bin/env python3
import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_CASES_PATH = ROOT_DIR / "tests" / "regression" / "timetable" / "timetable_ocr_diagnostic_cases.json"
DEFAULT_REPORT_PATH = ROOT_DIR / "tests" / "reports" / "timetable" / "timetable_ocr_diagnostics_report.json"
SUITE = "timetable_ocr_diagnostics"
SERVICE = "timetable"


def load_cases(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"no timetable OCR cases found in {path}")
    return cases


def write_report(out_path: Path, output: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def make_summary(status: str, total: int, passed: int, failed: int, skipped: int, metrics: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
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
        np = importlib.import_module("numpy")
        pytesseract = importlib.import_module("pytesseract")
        ocr_engine = importlib.import_module("image_analysis.ocr_engine")
    except ModuleNotFoundError as exc:
        write_skipped_report(out_path, f"timetable OCR diagnostics dependency is missing: {exc.name}", cases_path)
        raise SystemExit(0) from exc

    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        write_skipped_report(out_path, f"Tesseract runtime is unavailable: {exc}", cases_path)
        raise SystemExit(0) from exc

    return cv2, np, ocr_engine


def make_synthetic_timetable(case: Dict[str, Any], cv2, np):
    spec = case.get("input", {}).get("synthetic_grid", {})
    width = int(spec.get("width", 1000))
    height = int(spec.get("height", 1600))
    background = int(spec.get("background", 255))
    image = np.full((height, width, 3), background, dtype=np.uint8)

    line_color = tuple(spec.get("line_color", [0, 0, 0]))
    line_thickness = int(spec.get("line_thickness", 3))
    vertical_line_thickness = int(spec.get("vertical_line_thickness", line_thickness))
    horizontal_line_thickness = int(spec.get("horizontal_line_thickness", line_thickness))
    horizontal_x_start = int(spec.get("horizontal_x_start", spec.get("x_start", 80)))
    horizontal_x_end = int(spec.get("horizontal_x_end", spec.get("x_end", 920)))
    for x in range(int(spec.get("x_start", 80)), int(spec.get("x_stop", 921)), int(spec.get("x_step", 140))):
        cv2.line(
            image,
            (x, int(spec.get("y_start", 80))),
            (x, int(spec.get("y_end", 1480))),
            line_color,
            vertical_line_thickness,
        )
    for y in range(int(spec.get("y_start", 80)), int(spec.get("y_stop", 1481)), int(spec.get("y_step", 100))):
        cv2.line(image, (horizontal_x_start, y), (horizontal_x_end, y), line_color, horizontal_line_thickness)

    label = spec.get("label")
    if label:
        cv2.putText(
            image,
            str(label.get("text", "class")),
            (int(label.get("x", 380)), int(label.get("y", 410))),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(label.get("scale", 1.0)),
            line_color,
            int(label.get("thickness", 2)),
        )
    return image


def build_mock_ocr(case: Dict[str, Any]):
    mock = case.get("mock_ocr", {})
    primary = mock.get("primary", {})
    fallback = mock.get("fallback", {})

    def image_to_string(*args, **kwargs) -> str:
        config = kwargs.get("config", "")
        lines = fallback.get("lines", []) if "--psm 11" in config else primary.get("lines", [])
        return "\n".join(lines) + ("\n" if lines else "")

    def image_to_data(*args, **kwargs) -> dict:
        config = kwargs.get("config", "")
        conf = fallback.get("confidence", []) if "--psm 11" in config else primary.get("confidence", [])
        return {"conf": [str(value) for value in conf]}

    return image_to_string, image_to_data


def grid_detection_success(grid: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    min_x = int(expected.get("min_x_lines", 3))
    min_y = int(expected.get("min_y_lines", 3))
    return int(grid.get("x_line_count", 0)) >= min_x and int(grid.get("y_line_count", 0)) >= min_y


def empty_cell_ratio(ocr: Dict[str, Any]) -> Optional[float]:
    total = int(ocr.get("total_cells", 0))
    if total <= 0:
        return None
    text_cells = int(ocr.get("text_cells", 0))
    return round((total - text_cells) / total, 4)


def evaluate_case(case: Dict[str, Any], cv2, np, ocr_engine) -> Dict[str, Any]:
    image = make_synthetic_timetable(case, cv2, np)
    quality = ocr_engine.assess_image_quality(image)
    expected = case.get("expected", {})
    errors: List[str] = []

    if quality["width"] != case.get("input", {}).get("synthetic_grid", {}).get("width", 1000):
        errors.append(f"unexpected_quality_width:{quality}")
    if quality["height"] != case.get("input", {}).get("synthetic_grid", {}).get("height", 1600):
        errors.append(f"unexpected_quality_height:{quality}")

    mock_string, mock_data = build_mock_ocr(case)
    with patch("pytesseract.image_to_string", side_effect=mock_string), patch(
        "pytesseract.image_to_data",
        side_effect=mock_data,
    ):
        report = ocr_engine.extract_schedule_with_diagnostics(image)

    diagnostics = report["diagnostics"]
    schedules = report.get("schedules", [])
    grid = diagnostics.get("grid", {})
    ocr = diagnostics.get("ocr", {})
    metrics = {
        "grid_detection_success": grid_detection_success(grid, expected),
        "extracted_course_count": len(schedules),
        "empty_cell_ratio": empty_cell_ratio(ocr),
        "ocr_confidence": ocr.get("average_confidence"),
        "fallback_cell_count": ocr.get("fallback_cells", 0),
        "accepted_cell_count": ocr.get("accepted_cells", 0),
        "rejected_cell_count": len(ocr.get("rejected_cells", [])),
        "low_confidence_cell_count": ocr.get("low_confidence_cells", 0),
    }

    if not metrics["grid_detection_success"]:
        errors.append(f"grid_detection_failed:{grid}")
    if metrics["extracted_course_count"] < int(expected.get("min_extracted_course_count", 1)):
        errors.append(f"course_count_below_expected:{metrics['extracted_course_count']}")
    if metrics["fallback_cell_count"] < int(expected.get("min_fallback_cell_count", 0)):
        errors.append(f"fallback_count_below_expected:{metrics['fallback_cell_count']}")
    min_confidence = expected.get("min_ocr_confidence")
    if min_confidence is not None and (metrics["ocr_confidence"] is None or metrics["ocr_confidence"] < float(min_confidence)):
        errors.append(f"ocr_confidence_below_expected:{metrics['ocr_confidence']}")
    max_empty_ratio = expected.get("max_empty_cell_ratio")
    if max_empty_ratio is not None and metrics["empty_cell_ratio"] is not None and metrics["empty_cell_ratio"] > float(max_empty_ratio):
        errors.append(f"empty_cell_ratio_above_expected:{metrics['empty_cell_ratio']}")

    return {
        "id": case.get("id"),
        "name": case.get("name"),
        "status": "failed" if errors else "passed",
        "metrics": metrics,
        "errors": errors,
        "schedules": schedules,
        "diagnostics": diagnostics,
    }


def aggregate_metrics(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not case_results:
        return {}

    confidence_values = [
        result["metrics"]["ocr_confidence"]
        for result in case_results
        if result["metrics"].get("ocr_confidence") is not None
    ]
    empty_ratios = [
        result["metrics"]["empty_cell_ratio"]
        for result in case_results
        if result["metrics"].get("empty_cell_ratio") is not None
    ]
    return {
        "grid_detection_success_rate": round(
            sum(1 for result in case_results if result["metrics"].get("grid_detection_success")) / len(case_results),
            4,
        ),
        "extracted_course_count": sum(int(result["metrics"].get("extracted_course_count", 0)) for result in case_results),
        "average_empty_cell_ratio": round(sum(empty_ratios) / len(empty_ratios), 4) if empty_ratios else None,
        "average_ocr_confidence": round(sum(confidence_values) / len(confidence_values), 2) if confidence_values else None,
        "fallback_cell_count": sum(int(result["metrics"].get("fallback_cell_count", 0)) for result in case_results),
        "accepted_cell_count": sum(int(result["metrics"].get("accepted_cell_count", 0)) for result in case_results),
        "rejected_cell_count": sum(int(result["metrics"].get("rejected_cell_count", 0)) for result in case_results),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Write a timetable OCR diagnostics report")
    ap.add_argument("--cases", default=str(DEFAULT_CASES_PATH), help="golden OCR diagnostic cases path")
    ap.add_argument("--out", default=str(DEFAULT_REPORT_PATH), help="output report path")
    args = ap.parse_args()

    out_path = Path(args.out)
    cases_path = Path(args.cases)
    cases = load_cases(cases_path)
    cv2, np, ocr_engine = load_ocr_dependencies(out_path, cases_path)

    case_results = [evaluate_case(case, cv2, np, ocr_engine) for case in cases]
    errors = [
        f"{result['id']}:{error}"
        for result in case_results
        for error in result.get("errors", [])
    ]
    failed = sum(1 for result in case_results if result["status"] == "failed")
    passed = len(case_results) - failed
    summary = make_summary(
        status="failed" if failed else "passed",
        total=len(case_results),
        passed=passed,
        failed=failed,
        skipped=0,
        metrics=aggregate_metrics(case_results),
        errors=errors,
    )
    output = {
        "schema_version": 1,
        "suite": SUITE,
        "service": SERVICE,
        "summary": summary,
        "cases_path": str(cases_path),
        "case_results": case_results,
    }
    write_report(out_path, output)

    if failed:
        for error in errors:
            print(f"[FAIL] {error}")
        print(json.dumps(summary, ensure_ascii=False))
        print(f"report={out_path}")
        return 1

    print("[OK] timetable OCR diagnostics report")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
