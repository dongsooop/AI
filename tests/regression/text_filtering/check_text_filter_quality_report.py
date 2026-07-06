#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_CASES_PATH = ROOT_DIR / "tests" / "regression" / "text_filtering" / "cases"
DEFAULT_REPORT_PATH = ROOT_DIR / "tests" / "reports" / "text_filtering" / "text_filter_quality_report.json"
SUITE = "text_filter_quality"
SERVICE = "text_filtering"
BASELINE_PHASE = "1"


def load_case_payloads(path: Path) -> list[tuple[Path, dict[str, Any]]]:
    if path.is_dir():
        case_files = sorted(child for child in path.glob("*.json") if child.is_file())
        if not case_files:
            raise ValueError(f"no text filtering case files found in {path}")
        return [(case_file, json.loads(case_file.read_text(encoding="utf-8"))) for case_file in case_files]

    return [(path, json.loads(path.read_text(encoding="utf-8")))]


def load_cases(path: Path) -> list[dict[str, Any]]:
    loaded_cases: list[dict[str, Any]] = []
    for case_file, payload in load_case_payloads(path):
        resolved_case_file = case_file.resolve()
        cases = payload.get("cases", [])
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"no text filtering cases found in {case_file}")
        for case in cases:
            if isinstance(case, dict):
                loaded_case = dict(case)
                loaded_case.setdefault("source_file", str(resolved_case_file.relative_to(ROOT_DIR)))
                loaded_cases.append(loaded_case)

    if not loaded_cases:
        raise ValueError(f"no text filtering cases found in {path}")
    return loaded_cases


def load_case_metadata(path: Path) -> dict[str, Any]:
    payloads = load_case_payloads(path)
    if len(payloads) == 1:
        metadata = payloads[0][1].get("metadata", {})
        return metadata if isinstance(metadata, dict) else {}

    case_files: list[dict[str, Any]] = []
    for case_file, payload in payloads:
        resolved_case_file = case_file.resolve()
        metadata = payload.get("metadata", {})
        cases = payload.get("cases", [])
        case_files.append({
            "path": str(resolved_case_file.relative_to(ROOT_DIR)),
            "metadata": metadata if isinstance(metadata, dict) else {},
            "case_count": len(cases) if isinstance(cases, list) else 0,
        })

    return {
        "purpose": "text_filter_shadow_variant_evaluation",
        "contract": "Do not change production True/False or has_profanity behavior in this phase.",
        "case_files": case_files,
    }


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


def count_cases_by_category(cases: list[dict[str, Any]]) -> dict[str, int]:
    by_category: dict[str, int] = {}
    for case in cases:
        category = str(case.get("category", "") or "uncategorized")
        by_category[category] = by_category.get(category, 0) + 1
    return by_category


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
    metadata = load_case_metadata(cases_path)
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
        "baseline": {
            "phase": BASELINE_PHASE,
            "report_only": True,
            "case_metadata": metadata,
        },
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
    from text_filtering.word_matcher import detect_bad_word_match_dicts

    text = str(case.get("text", "") or "")
    expected_has_profanity = bool(case.get("expected", {}).get("has_profanity"))
    labels = text_filter_service.analyze_text_labels(text)
    actual_has_profanity = any(label == "비속어" for label in labels)
    shadow_matches = detect_bad_word_match_dicts(text)
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
        "source_file": case.get("source_file"),
        "text": text,
        "expected": {
            "has_profanity": expected_has_profanity,
        },
        "actual": {
            "has_profanity": actual_has_profanity,
            "labels": labels,
        },
        "shadow": {
            "has_match": bool(shadow_matches),
            "match_count": len(shadow_matches),
            "matches": shadow_matches,
        },
        "status": "passed" if passed else "failed",
        "errors": errors,
    }


def aggregate_shadow_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(case_results)
    shadow_matched_case_count = 0
    shadow_match_count = 0
    shadow_detected_false_negative_count = 0
    shadow_false_positive_candidate_count = 0
    shadow_true_positive_candidate_count = 0
    shadow_strong_rule_candidate_match_count = 0
    shadow_strong_rule_candidate_case_count = 0
    shadow_strong_rule_detected_false_negative_count = 0
    shadow_strong_rule_false_positive_candidate_count = 0
    pattern_counts: dict[str, int] = {}
    strong_rule_pattern_counts: dict[str, int] = {}
    by_category: dict[str, dict[str, Any]] = {}

    for result in case_results:
        category = str(result.get("category", "") or "uncategorized")
        expected_has_profanity = bool(result.get("expected", {}).get("has_profanity"))
        actual_has_profanity = bool(result.get("actual", {}).get("has_profanity"))
        shadow = result.get("shadow", {})
        matches = shadow.get("matches", []) if isinstance(shadow, dict) else []
        match_count = len(matches)
        has_match = match_count > 0
        strong_matches = [match for match in matches if match.get("strong_rule_candidate") is True]
        strong_match_count = len(strong_matches)
        has_strong_match = strong_match_count > 0

        bucket = by_category.setdefault(
            category,
            {
                "total": 0,
                "shadow_matched": 0,
                "shadow_unmatched": 0,
                "shadow_false_positive_candidates": 0,
                "shadow_detected_false_negatives": 0,
                "shadow_strong_rule_matched": 0,
                "shadow_strong_rule_false_positive_candidates": 0,
                "shadow_strong_rule_detected_false_negatives": 0,
            },
        )
        bucket["total"] += 1
        if has_match:
            shadow_matched_case_count += 1
            bucket["shadow_matched"] += 1
        else:
            bucket["shadow_unmatched"] += 1

        shadow_match_count += match_count
        for match in matches:
            pattern_id = str(match.get("pattern_id", "") or "unknown")
            pattern_counts[pattern_id] = pattern_counts.get(pattern_id, 0) + 1
            if match.get("strong_rule_candidate") is True:
                shadow_strong_rule_candidate_match_count += 1
                strong_rule_pattern_counts[pattern_id] = strong_rule_pattern_counts.get(pattern_id, 0) + 1

        if has_strong_match:
            shadow_strong_rule_candidate_case_count += 1
            bucket["shadow_strong_rule_matched"] += 1

        if expected_has_profanity and has_match:
            shadow_true_positive_candidate_count += 1
        if not expected_has_profanity and has_match:
            shadow_false_positive_candidate_count += 1
            bucket["shadow_false_positive_candidates"] += 1
        if expected_has_profanity and not actual_has_profanity and has_match:
            shadow_detected_false_negative_count += 1
            bucket["shadow_detected_false_negatives"] += 1
        if not expected_has_profanity and has_strong_match:
            shadow_strong_rule_false_positive_candidate_count += 1
            bucket["shadow_strong_rule_false_positive_candidates"] += 1
        if expected_has_profanity and not actual_has_profanity and has_strong_match:
            shadow_strong_rule_detected_false_negative_count += 1
            bucket["shadow_strong_rule_detected_false_negatives"] += 1

    for bucket in by_category.values():
        bucket["shadow_match_rate"] = round(bucket["shadow_matched"] / bucket["total"], 4) if bucket["total"] else 0.0
        bucket["shadow_strong_rule_match_rate"] = (
            round(bucket["shadow_strong_rule_matched"] / bucket["total"], 4) if bucket["total"] else 0.0
        )

    return {
        "shadow_match_count": shadow_match_count,
        "shadow_matched_case_count": shadow_matched_case_count,
        "shadow_match_rate": round(shadow_matched_case_count / total, 4) if total else 0.0,
        "shadow_true_positive_candidate_count": shadow_true_positive_candidate_count,
        "shadow_false_positive_candidate_count": shadow_false_positive_candidate_count,
        "shadow_detected_false_negative_count": shadow_detected_false_negative_count,
        "shadow_pattern_counts": pattern_counts,
        "shadow_strong_rule_candidate_match_count": shadow_strong_rule_candidate_match_count,
        "shadow_strong_rule_candidate_case_count": shadow_strong_rule_candidate_case_count,
        "shadow_strong_rule_detected_false_negative_count": shadow_strong_rule_detected_false_negative_count,
        "shadow_strong_rule_false_positive_candidate_count": shadow_strong_rule_false_positive_candidate_count,
        "shadow_strong_rule_pattern_counts": strong_rule_pattern_counts,
        "shadow_by_category": by_category,
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
        **aggregate_shadow_metrics(case_results),
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
    case_metadata = load_case_metadata(cases_path)
    cases = load_cases(cases_path)
    validation_errors = validate_cases(cases)
    case_category_counts = count_cases_by_category(cases)

    if validation_errors:
        failed_case_ids = {
            (err.split(":", 1)[0] if ":" in err else err)
            for err in validation_errors
        }
        summary = make_summary(
            status="failed",
            total=len(cases),
            passed=0,
            failed=len(failed_case_ids),
            skipped=0,
            metrics={},
            errors=validation_errors,
        )
        output = {
            "schema_version": 1,
            "suite": SUITE,
            "service": SERVICE,
            "baseline": {
                "phase": BASELINE_PHASE,
                "report_only": True,
                "case_metadata": case_metadata,
                "case_category_counts": case_category_counts,
            },
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
            metrics={
                "validated_case_count": len(cases),
                "case_category_counts": case_category_counts,
                "baseline_phase": BASELINE_PHASE,
                "report_only": True,
            },
            errors=[],
        )
        output = {
            "schema_version": 1,
            "suite": SUITE,
            "service": SERVICE,
            "baseline": {
                "phase": BASELINE_PHASE,
                "report_only": True,
                "case_metadata": case_metadata,
                "case_category_counts": case_category_counts,
            },
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
        metrics={
            **aggregate_metrics(case_results),
            "case_category_counts": case_category_counts,
            "baseline_phase": BASELINE_PHASE,
            "report_only": True,
        },
        errors=errors,
    )
    output = {
        "schema_version": 1,
        "suite": SUITE,
        "service": SERVICE,
        "baseline": {
            "phase": BASELINE_PHASE,
            "report_only": True,
            "case_metadata": case_metadata,
            "case_category_counts": case_category_counts,
        },
        "summary": summary,
        "cases_path": str(cases_path),
        "case_results": case_results,
        "notes": [
            "This report calls analyze_text_labels() only, so test cases are not appended to data/bad_text_sample.txt.",
            "rule_endpoint_pass_rate is reported for the current API contract, which uses the shared ML-backed analyzer.",
            "shadow matches are report-only word-level detector results and do not affect pass/fail status.",
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
