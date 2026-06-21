#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_CASES_PATH = ROOT_DIR / "tests" / "regression" / "text_filtering" / "text_filter_quality_cases.json"
DEFAULT_REPORT_PATH = ROOT_DIR / "tests" / "reports" / "text_filtering" / "text_filter_quality_report.json"
SUITE = "text_filter_quality"
SERVICE = "text_filtering"


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"no text filtering cases found in {path}")
    return cases


def validate_cases(cases: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()

    for idx, case in enumerate(cases, start=1):
        case_id = str(case.get("id", "") or "")
        prefix = case_id or f"case#{idx}"

        if not case_id:
            errors.append(f"{prefix}:missing_id")
        elif case_id in seen_ids:
            errors.append(f"{prefix}:duplicate_id")
        seen_ids.add(case_id)

        if not str(case.get("category", "") or "").strip():
            errors.append(f"{prefix}:missing_category")
        if not str(case.get("text", "") or "").strip():
            errors.append(f"{prefix}:missing_text")

        expected = case.get("expected")
        if not isinstance(expected, dict):
            errors.append(f"{prefix}:missing_expected")
            continue
        if not isinstance(expected.get("has_profanity"), bool):
            errors.append(f"{prefix}:expected_has_profanity_must_be_bool")

    return errors


def write_report(out_path: Path, output: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def make_summary(
    status: str,
    total: int,
    passed: int,
    failed: int,
    skipped: int,
    metrics: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
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
        "notes": [
            "The report only observes existing text filtering logic and does not write feedback logs.",
        ],
    }
    write_report(out_path, output)
    print(f"[SKIP] {reason}")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"report={out_path}")


def load_text_filter_service(out_path: Path, cases_path: Path):
    try:
        from text_filtering import service as text_filter_service
    except ModuleNotFoundError as exc:
        write_skipped_report(out_path, f"text filtering dependency is missing: {exc.name}", cases_path)
        raise SystemExit(0) from exc
    except Exception as exc:
        write_skipped_report(out_path, f"text filtering service import failed: {type(exc).__name__}: {exc}", cases_path)
        raise SystemExit(0) from exc

    readiness = text_filter_service.get_text_filter_readiness()
    if readiness.get("status") != "ready":
        write_skipped_report(out_path, f"text filter model is not ready: {readiness}", cases_path)
        raise SystemExit(0)

    return text_filter_service


def evaluate_case(case: dict[str, Any], text_filter_service) -> dict[str, Any]:
    text = str(case.get("text", "") or "")
    expected_has_profanity = bool(case.get("expected", {}).get("has_profanity"))
    labels = text_filter_service.analyze_text_labels(text)
    actual_has_profanity = any(label == "비속어" for label in labels)
    passed = actual_has_profanity == expected_has_profanity

    errors: list[str] = []
    if not passed:
        if actual_has_profanity:
            errors.append("false_positive")
        else:
            errors.append("false_negative")

    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "text": text,
        "expected": {
            "has_profanity": expected_has_profanity,
        },
        "actual": {
            "has_profanity": actual_has_profanity,
            "labels": labels,
        },
        "status": "passed" if passed else "failed",
        "errors": errors,
    }


def aggregate_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(case_results)
    if total == 0:
        return {}

    passed = sum(1 for result in case_results if result["status"] == "passed")
    false_positive_count = sum(1 for result in case_results if "false_positive" in result.get("errors", []))
    false_negative_count = sum(1 for result in case_results if "false_negative" in result.get("errors", []))

    by_category: dict[str, dict[str, Any]] = {}
    for result in case_results:
        category = str(result.get("category", "") or "uncategorized")
        bucket = by_category.setdefault(category, {"total": 0, "passed": 0, "failed": 0})
        bucket["total"] += 1
        if result["status"] == "passed":
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1

    for bucket in by_category.values():
        bucket["pass_rate"] = round(bucket["passed"] / bucket["total"], 4) if bucket["total"] else 0.0

    pass_rate = round(passed / total, 4)
    return {
        "false_positive_count": false_positive_count,
        "false_negative_count": false_negative_count,
        "pass_rate": pass_rate,
        "ml_filter_pass_rate": pass_rate,
        "rule_filter_pass_rate": None,
        "rule_endpoint_pass_rate": pass_rate,
        "by_category": by_category,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Write a text filtering quality report")
    ap.add_argument("--cases", default=str(DEFAULT_CASES_PATH), help="golden text filtering cases path")
    ap.add_argument("--out", default=str(DEFAULT_REPORT_PATH), help="output report path")
    ap.add_argument("--validate-only", action="store_true", help="validate golden cases without loading the model")
    ap.add_argument("--strict", action="store_true", help="return a non-zero exit code when evaluated cases fail")
    args = ap.parse_args()

    cases_path = Path(args.cases)
    out_path = Path(args.out)
    cases = load_cases(cases_path)
    validation_errors = validate_cases(cases)

    if validation_errors:
        summary = make_summary(
            status="failed",
            total=len(cases),
            passed=0,
            failed=len(validation_errors),
            skipped=0,
            metrics={},
            errors=validation_errors,
        )
        output = {
            "schema_version": 1,
            "suite": SUITE,
            "service": SERVICE,
            "summary": summary,
            "cases_path": str(cases_path),
            "case_results": [],
        }
        write_report(out_path, output)
        for error in validation_errors:
            print(f"[FAIL] {error}")
        print(json.dumps(summary, ensure_ascii=False))
        print(f"report={out_path}")
        return 1

    if args.validate_only:
        summary = make_summary(
            status="passed",
            total=len(cases),
            passed=len(cases),
            failed=0,
            skipped=0,
            metrics={"validated_case_count": len(cases)},
            errors=[],
        )
        output = {
            "schema_version": 1,
            "suite": SUITE,
            "service": SERVICE,
            "summary": summary,
            "cases_path": str(cases_path),
            "case_results": [],
        }
        write_report(out_path, output)
        print("[OK] text filtering quality cases validated")
        print(json.dumps(summary, ensure_ascii=False))
        print(f"report={out_path}")
        return 0

    text_filter_service = load_text_filter_service(out_path, cases_path)
    case_results = [evaluate_case(case, text_filter_service) for case in cases]
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
        "notes": [
            "This report calls analyze_text_labels() only, so test cases are not appended to data/bad_text_sample.txt.",
            "rule_endpoint_pass_rate is reported for the current API contract, which uses the shared ML-backed analyzer.",
        ],
    }
    write_report(out_path, output)

    if failed:
        for error in errors:
            print(f"[FAIL] {error}")
        print(json.dumps(summary, ensure_ascii=False))
        print(f"report={out_path}")
        return 1 if args.strict else 0

    print("[OK] text filtering quality report")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
