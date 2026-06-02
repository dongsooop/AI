#!/usr/bin/env python3
import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DATE_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")
URL_RE = re.compile(r"https?://[^\s)>\]}\"']+")
REFUSAL_RE = re.compile(
    r"("
    r"답변할\s*수\s*없|"
    r"확인할\s*수\s*없|"
    r"제공할\s*수\s*없|"
    r"제공하기\s*어렵|"
    r"찾을\s*수\s*없|"
    r"찾지\s*못|"
    r"자료가\s*없|"
    r"정보가\s*없|"
    r"알\s*수\s*없|"
    r"모르겠|"
    r"범위\s*밖|"
    r"관련\s*정보.*없"
    r")"
)


DEFAULT_CASES_PATH = Path(__file__).with_name("rag_eval_cases.json")
DEFAULT_REPORT_PATH = ROOT_DIR / "tests" / "reports" / "chatbot" / "rag_eval_report.json"


def _load_cases_file(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    default_category = ""
    if isinstance(data, dict):
        default_category = str(data.get("category", "") or "")
        cases = data.get("cases")
    else:
        cases = data

    if not isinstance(cases, list):
        raise ValueError(f"RAG eval cases must be a JSON array or an object with cases: {path}")

    loaded = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError(f"RAG eval case must be an object: {path}")
        item = dict(case)
        if default_category and not item.get("category"):
            item["category"] = default_category
        try:
            case_file = str(path.relative_to(ROOT_DIR))
        except ValueError:
            case_file = str(path)
        item.setdefault("case_file", case_file)
        loaded.append(item)
    return loaded


def load_cases(path: Path) -> list[dict]:
    if path.is_dir():
        files = sorted(path.glob("*.json"))
        if not files:
            raise ValueError(f"no RAG eval case JSON files found in {path}")
        cases = []
        for file_path in files:
            cases.extend(_load_cases_file(file_path))
        return cases
    return _load_cases_file(path)


def _is_list_of_str(value) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def validate_cases(cases: list[dict]) -> list[str]:
    errors = []
    seen_ids = set()
    list_fields = [
        "expected_url_contains",
        "expected_title_contains",
        "answer_must_contain_any",
        "answer_must_contain_all",
    ]
    bool_fields = ["requires_source_url", "requires_date", "expects_refusal"]

    for idx, case in enumerate(cases, start=1):
        case_id = str(case.get("id", "") or "")
        kind = case.get("kind", "search")
        prefix = case_id or f"case#{idx}"

        if not case_id:
            errors.append(f"{prefix}:missing_id")
        elif case_id in seen_ids:
            errors.append(f"{prefix}:duplicate_id")
        seen_ids.add(case_id)

        if kind not in {"search", "schedule"}:
            errors.append(f"{prefix}:unknown_kind:{kind}")
        if not str(case.get("query", "") or "").strip():
            errors.append(f"{prefix}:missing_query")
        if not str(case.get("category", "") or "").strip():
            errors.append(f"{prefix}:missing_category")

        for field in list_fields:
            if field in case and not _is_list_of_str(case.get(field)):
                errors.append(f"{prefix}:{field}_must_be_list_of_strings")

        for field in bool_fields:
            if field in case and not isinstance(case.get(field), bool):
                errors.append(f"{prefix}:{field}_must_be_bool")

        if kind == "search":
            has_expected_source = bool(case.get("expected_url_contains") or case.get("expected_title_contains"))
            if not has_expected_source and not case.get("expects_refusal", False):
                errors.append(f"{prefix}:missing_expected_source")
            if case.get("requires_source_url") and not case.get("expected_url_contains"):
                errors.append(f"{prefix}:requires_source_without_expected_url")

        if not (
            case.get("answer_must_contain_any")
            or case.get("answer_must_contain_all")
            or case.get("expects_refusal", False)
        ):
            errors.append(f"{prefix}:missing_answer_expectation")
    return errors


def configure_default_artifact_env() -> None:
    """Make local evaluation runnable without requiring a private .env file."""
    defaults = {
        "DATA_JSON": ROOT_DIR / "data" / "school_info" / "old_data" / "dmu_documents_cleaned.json",
        "ART_DIR": ROOT_DIR / "model" / "artifacts",
        "SEARCH_DF_PATH": ROOT_DIR / "model" / "artifacts" / "search_df.parquet",
        "EMB_PATH": ROOT_DIR / "model" / "artifacts" / "embeddings.npy",
        "BM25_PATH": ROOT_DIR / "model" / "artifacts" / "bm25.pkl",
        "TOK_PATH": ROOT_DIR / "model" / "artifacts" / "tokenized_corpus.json.gz",
        "CONTACTS_CSV": ROOT_DIR / "model" / "artifacts" / "contact_docs.csv",
        "META_PATH": ROOT_DIR / "model" / "artifacts" / "meta.json",
        "CONTACT_KWS_PATH": ROOT_DIR / "data" / "sub_model_data" / "contact_kws.txt",
        "SCHEDULE_CSV_PATH": ROOT_DIR / "data" / "schedule" / "학사일정_년도추가.csv",
    }
    for key, path in defaults.items():
        if os.getenv(key):
            continue
        if path.exists():
            os.environ[key] = str(path)


def contains_any(text: str, needles: list[str]) -> bool:
    if not needles:
        return True
    return any(needle and needle in text for needle in needles)


def contains_all(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def is_refusal_answer(text: str) -> bool:
    return bool(REFUSAL_RE.search(text or ""))


def urls_from_hits(hits) -> list[str]:
    if hits is None or getattr(hits, "empty", True) or "url" not in hits.columns:
        return []
    return [str(url or "") for url in hits["url"].head(10).tolist()]


def url_contains(url: str, fragments: list[str]) -> bool:
    return contains_any(url, fragments)


def rank_for_expected_url(hits, fragments: list[str]) -> int | None:
    if not fragments or hits is None or getattr(hits, "empty", True) or "url" not in hits.columns:
        return None
    for idx, url in enumerate(hits["url"].astype(str).tolist(), start=1):
        if url_contains(url, fragments):
            return idx
    return None


def title_matches(hits, fragments: list[str]) -> bool:
    if not fragments or hits is None or getattr(hits, "empty", True) or "title" not in hits.columns:
        return bool(not fragments)
    top_title = str(hits.iloc[0].get("title", "") or "")
    return contains_any(top_title, fragments)


def hallucination_flags(answer: str, hits, expected_url_fragments: list[str]) -> list[str]:
    """Detect obvious generated-source drift without treating extra official links as errors.

    build_answer() may include reranked official school URLs beyond the metric top_k, so the
    proxy only flags non-DMU URLs here. Stricter context-vs-answer checks belong in an API/LLM
    evaluator where the exact prompt context is captured.
    """
    flags = []
    allowed_urls = urls_from_hits(hits)
    answer_urls = URL_RE.findall(answer or "")
    for url in answer_urls:
        if "dongyang.ac.kr" in url:
            continue
        if url in allowed_urls:
            continue
        if any(fragment and fragment in url for fragment in expected_url_fragments):
            continue
        flags.append(f"unknown_url:{url}")
    return flags


def evaluate_search_case(case: dict, top_k: int, answer_top_k: int) -> dict:
    query = case.get("query", "")
    try:
        from LLM.sub_model.query_index import build_answer, hybrid_search
        import_error = ""
    except Exception as exc:
        import_error = f"{type(exc).__name__}:{exc}"
        return {
            "id": case.get("id"),
            "category": case.get("category", ""),
            "kind": "search",
            "query": query,
            "top1_url_match": False,
            "top3_url_match": False,
            "top1_title_match": False,
            "answer_keyword_match": False,
            "source_url_match": False if case.get("requires_source_url", False) else True,
            "source_url_required": bool(case.get("requires_source_url", False)),
            "date_match": True,
            "refused": False,
            "expected_refusal": bool(case.get("expects_refusal", False)),
            "refusal_match": not bool(case.get("expects_refusal", False)),
            "hallucination_detected": False,
            "hallucination_flags": [],
            "expected_url_rank": None,
            "top1_title": "",
            "top1_url": "",
            "answer": "",
            "errors": {
                "import": import_error,
                "retrieval": "",
                "answer": "",
            },
            "retrieval_latency_ms": 0.0,
            "answer_latency_ms": 0.0,
            "passed": False,
        }

    started = time.perf_counter()
    try:
        hits = hybrid_search(query, top_k=top_k)
        retrieval_error = ""
    except Exception as exc:
        hits = None
        retrieval_error = f"{type(exc).__name__}:{exc}"
    retrieval_ms = round((time.perf_counter() - started) * 1000, 2)

    answer_started = time.perf_counter()
    try:
        answer_payload = build_answer(query, top_k=answer_top_k) or {}
        answer_error = ""
    except Exception as exc:
        answer_payload = {}
        answer_error = f"{type(exc).__name__}:{exc}"
    answer_ms = round((time.perf_counter() - answer_started) * 1000, 2)

    answer = str(answer_payload.get("answer", "") or "")
    expected_urls = case.get("expected_url_contains", [])
    url_rank = rank_for_expected_url(hits, expected_urls)
    top1_url_match = url_rank == 1
    top3_url_match = bool(url_rank and url_rank <= 3)
    answer_url_match = contains_any(answer, expected_urls) if expected_urls else True
    source_url_match = answer_url_match if case.get("requires_source_url", False) else True
    answer_any = contains_any(answer, case.get("answer_must_contain_any", []))
    answer_all = contains_all(answer, case.get("answer_must_contain_all", []))
    date_pass = bool(DATE_RE.search(answer)) if case.get("requires_date", False) else True
    refused = is_refusal_answer(answer)
    expected_refusal = bool(case.get("expects_refusal", False))
    refusal_match = refused if expected_refusal else not refused
    hallu = hallucination_flags(answer, hits, expected_urls)

    return {
        "id": case.get("id"),
        "category": case.get("category", ""),
        "kind": "search",
        "query": query,
        "top1_url_match": top1_url_match,
        "top3_url_match": top3_url_match,
        "top1_title_match": title_matches(hits, case.get("expected_title_contains", [])),
        "answer_keyword_match": answer_any and answer_all,
        "source_url_match": source_url_match,
        "source_url_required": bool(case.get("requires_source_url", False)),
        "date_match": date_pass,
        "refused": refused,
        "expected_refusal": expected_refusal,
        "refusal_match": refusal_match,
        "hallucination_detected": bool(hallu),
        "hallucination_flags": hallu,
        "expected_url_rank": url_rank,
        "top1_title": str(hits.iloc[0].get("title", "") if hits is not None and not hits.empty else ""),
        "top1_url": str(hits.iloc[0].get("url", "") if hits is not None and not hits.empty else ""),
        "answer": answer,
        "errors": {
            "import": "",
            "retrieval": retrieval_error,
            "answer": answer_error,
        },
        "retrieval_latency_ms": retrieval_ms,
        "answer_latency_ms": answer_ms,
        "passed": all([
            not retrieval_error,
            not answer_error,
            top3_url_match if expected_urls and not expected_refusal else True,
            answer_any and answer_all,
            source_url_match,
            date_pass,
            refusal_match,
            not hallu,
        ]),
    }


def evaluate_schedule_case(case: dict, top_k: int) -> dict:
    query = case.get("query", "")
    try:
        from LLM.sub_model.schedule_index import schedule_search
        import_error = ""
    except Exception as exc:
        import_error = f"{type(exc).__name__}:{exc}"
        return {
            "id": case.get("id"),
            "category": case.get("category", ""),
            "kind": "schedule",
            "query": query,
            "top1_url_match": None,
            "top3_url_match": None,
            "top1_title_match": None,
            "answer_keyword_match": False,
            "source_url_match": True,
            "source_url_required": False,
            "date_match": False if case.get("requires_date", False) else True,
            "refused": False,
            "expected_refusal": bool(case.get("expects_refusal", False)),
            "refusal_match": not bool(case.get("expects_refusal", False)),
            "hallucination_detected": False,
            "hallucination_flags": [],
            "expected_url_rank": None,
            "top1_title": "",
            "top1_url": "",
            "answer": "",
            "errors": {
                "import": import_error,
                "schedule": "",
            },
            "retrieval_latency_ms": 0.0,
            "answer_latency_ms": 0.0,
            "passed": False,
        }

    started = time.perf_counter()
    try:
        answer = schedule_search(query, top_k=top_k) or ""
        error = ""
    except Exception as exc:
        answer = ""
        error = f"{type(exc).__name__}:{exc}"
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    answer_any = contains_any(answer, case.get("answer_must_contain_any", []))
    answer_all = contains_all(answer, case.get("answer_must_contain_all", []))
    date_pass = bool(DATE_RE.search(answer)) if case.get("requires_date", False) else True
    refused = is_refusal_answer(answer)
    expected_refusal = bool(case.get("expects_refusal", False))
    refusal_match = refused if expected_refusal else not refused

    return {
        "id": case.get("id"),
        "category": case.get("category", ""),
        "kind": "schedule",
        "query": query,
        "top1_url_match": None,
        "top3_url_match": None,
        "top1_title_match": None,
        "answer_keyword_match": answer_any and answer_all,
        "source_url_match": True,
        "source_url_required": False,
        "date_match": date_pass,
        "refused": refused,
        "expected_refusal": expected_refusal,
        "refusal_match": refusal_match,
        "hallucination_detected": False,
        "hallucination_flags": [],
        "expected_url_rank": None,
        "top1_title": "",
        "top1_url": "",
        "answer": answer,
        "errors": {
            "import": "",
            "schedule": error,
        },
        "retrieval_latency_ms": latency_ms,
        "answer_latency_ms": 0.0,
        "passed": all([not error, answer_any and answer_all, date_pass, refusal_match]),
    }


def pct(numerator: int, denominator: int) -> float:
    return round((numerator / denominator) * 100, 2) if denominator else 0.0


def summarize(results: list[dict]) -> dict:
    search_results = [r for r in results if r["kind"] == "search"]
    schedule_results = [r for r in results if r["kind"] == "schedule"]
    source_required_results = [r for r in search_results if r.get("source_url_required")]
    latencies = [float(r.get("retrieval_latency_ms", 0.0)) for r in results]
    answer_latencies = [float(r.get("answer_latency_ms", 0.0)) for r in search_results]
    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])

    metrics = {
        "recall_at_1": pct(sum(1 for r in search_results if r["top1_url_match"]), len(search_results)),
        "recall_at_3": pct(sum(1 for r in search_results if r["top3_url_match"]), len(search_results)),
        "source_url_pass_rate": pct(
            sum(1 for r in source_required_results if r["source_url_match"]),
            len(source_required_results),
        ),
        "unofficial_url_hallucination_rate": pct(
            sum(1 for r in search_results if r["hallucination_detected"]),
            len(search_results),
        ),
        "refusal_rate": pct(sum(1 for r in results if r.get("refused")), len(results)),
        "refusal_expectation_pass_rate": pct(sum(1 for r in results if r.get("refusal_match")), len(results)),
    }

    summary = {
        "schema_version": 1,
        "suite": "rag_retrieval",
        "service": "chatbot",
        "status": "passed" if failed == 0 else "failed",
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "skipped": 0,
        "pass_rate": pct(passed, len(results)),
        "search_total": len(search_results),
        "schedule_total": len(schedule_results),
        "top1_url_accuracy": metrics["recall_at_1"],
        "top3_url_accuracy": metrics["recall_at_3"],
        "recall_at_1": metrics["recall_at_1"],
        "recall_at_3": metrics["recall_at_3"],
        "top1_title_accuracy": pct(sum(1 for r in search_results if r["top1_title_match"]), len(search_results)),
        "answer_keyword_pass_rate": pct(sum(1 for r in results if r["answer_keyword_match"]), len(results)),
        "source_url_pass_rate": metrics["source_url_pass_rate"],
        "date_pass_rate": pct(sum(1 for r in results if r["date_match"]), len(results)),
        "hallucination_proxy_rate": metrics["unofficial_url_hallucination_rate"],
        "unofficial_url_hallucination_rate": metrics["unofficial_url_hallucination_rate"],
        "refusal_rate": metrics["refusal_rate"],
        "refusal_expectation_pass_rate": metrics["refusal_expectation_pass_rate"],
        "metrics": metrics,
        "errors": [],
        "avg_retrieval_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p95_retrieval_latency_ms": round(statistics.quantiles(latencies, n=20)[18], 2) if len(latencies) >= 20 else 0.0,
        "avg_answer_latency_ms": round(statistics.mean(answer_latencies), 2) if answer_latencies else 0.0,
    }

    by_category = {}
    for category in sorted({r["category"] for r in results}):
        items = [r for r in results if r["category"] == category]
        searches = [r for r in items if r["kind"] == "search"]
        source_required = [r for r in searches if r.get("source_url_required")]
        by_category[category] = {
            "total": len(items),
            "passed": sum(1 for r in items if r["passed"]),
            "failed": sum(1 for r in items if not r["passed"]),
            "pass_rate": pct(sum(1 for r in items if r["passed"]), len(items)),
            "recall_at_1": pct(sum(1 for r in searches if r["top1_url_match"]), len(searches)),
            "recall_at_3": pct(sum(1 for r in searches if r["top3_url_match"]), len(searches)),
            "top3_url_accuracy": pct(sum(1 for r in searches if r["top3_url_match"]), len(searches)),
            "answer_keyword_pass_rate": pct(sum(1 for r in items if r["answer_keyword_match"]), len(items)),
            "source_url_pass_rate": pct(
                sum(1 for r in source_required if r["source_url_match"]),
                len(source_required),
            ),
            "unofficial_url_hallucination_rate": pct(
                sum(1 for r in searches if r["hallucination_detected"]),
                len(searches),
            ),
            "refusal_rate": pct(sum(1 for r in items if r.get("refused")), len(items)),
            "refusal_expectation_pass_rate": pct(sum(1 for r in items if r.get("refusal_match")), len(items)),
        }
    summary["by_category"] = by_category
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate RAG retrieval and source-grounded answer quality")
    ap.add_argument(
        "--cases",
        default=str(DEFAULT_CASES_PATH),
        help="path to a RAG eval cases JSON file or a directory of *.json files",
    )
    ap.add_argument("--top-k", type=int, default=5, help="hybrid_search top_k for retrieval metrics")
    ap.add_argument("--answer-top-k", type=int, default=6, help="build_answer top_k for answer metrics")
    ap.add_argument("--schedule-top-k", type=int, default=5, help="schedule_search top_k for schedule cases")
    ap.add_argument("--out", default=str(DEFAULT_REPORT_PATH), help="output report path")
    ap.add_argument("--fail-on-fail", action="store_true", help="exit 2 when any case fails")
    ap.add_argument("--validate-only", action="store_true", help="validate case schema without importing RAG runtime")
    args = ap.parse_args()

    cases = load_cases(Path(args.cases))
    validation_errors = validate_cases(cases)
    if validation_errors:
        print(json.dumps({"valid": False, "errors": validation_errors}, ensure_ascii=False))
        return 2
    if args.validate_only:
        categories = sorted({case.get("category", "") for case in cases})
        print(json.dumps({"valid": True, "total": len(cases), "categories": categories}, ensure_ascii=False))
        return 0

    configure_default_artifact_env()

    results = []
    for case in cases:
        kind = case.get("kind", "search")
        if kind == "schedule":
            results.append(evaluate_schedule_case(case, args.schedule_top_k))
        elif kind == "search":
            results.append(evaluate_search_case(case, args.top_k, args.answer_top_k))
        else:
            raise ValueError(f"unknown case kind: {kind}")

    report = {
        "cases_file": str(Path(args.cases)),
        "summary": summarize(results),
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"report={out_path}")

    failed = [r for r in results if not r["passed"]]
    if failed:
        print("failed_cases:")
        for r in failed:
            reasons = []
            if r["top3_url_match"] is False:
                reasons.append("top3_url")
            if not r["answer_keyword_match"]:
                reasons.append("answer_keyword")
            if not r["source_url_match"]:
                reasons.append("source_url")
            if not r["date_match"]:
                reasons.append("date")
            if not r.get("refusal_match", True):
                reasons.append("refusal")
            if r["hallucination_detected"]:
                reasons.append("hallucination_proxy")
            for stage, error in (r.get("errors") or {}).items():
                if error:
                    reasons.append(f"{stage}:{error}")
            print(f"- {r['id']}: {', '.join(reasons) or 'failed'}")

    return 2 if failed and args.fail_on_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
