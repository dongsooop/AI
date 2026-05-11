#!/usr/bin/env python3
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DATE_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")
URL_RE = re.compile(r"https?://[^\s)>\]}\"']+")


def load_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list):
        raise ValueError("RAG eval cases must be a JSON array")
    return cases


def contains_any(text: str, needles: list[str]) -> bool:
    if not needles:
        return True
    return any(needle and needle in text for needle in needles)


def contains_all(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


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
    from LLM.sub_model.query_index import build_answer, hybrid_search

    query = case.get("query", "")
    started = time.perf_counter()
    hits = hybrid_search(query, top_k=top_k)
    retrieval_ms = round((time.perf_counter() - started) * 1000, 2)

    answer_started = time.perf_counter()
    answer_payload = build_answer(query, top_k=answer_top_k) or {}
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
        "date_match": date_pass,
        "hallucination_detected": bool(hallu),
        "hallucination_flags": hallu,
        "expected_url_rank": url_rank,
        "top1_title": str(hits.iloc[0].get("title", "") if not hits.empty else ""),
        "top1_url": str(hits.iloc[0].get("url", "") if not hits.empty else ""),
        "answer": answer,
        "retrieval_latency_ms": retrieval_ms,
        "answer_latency_ms": answer_ms,
        "passed": all([
            top3_url_match if expected_urls else True,
            answer_any and answer_all,
            source_url_match,
            date_pass,
            not hallu,
        ]),
    }


def evaluate_schedule_case(case: dict, top_k: int) -> dict:
    from LLM.sub_model.schedule_index import schedule_search

    query = case.get("query", "")
    started = time.perf_counter()
    answer = schedule_search(query, top_k=top_k) or ""
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    answer_any = contains_any(answer, case.get("answer_must_contain_any", []))
    answer_all = contains_all(answer, case.get("answer_must_contain_all", []))
    date_pass = bool(DATE_RE.search(answer)) if case.get("requires_date", False) else True

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
        "date_match": date_pass,
        "hallucination_detected": False,
        "hallucination_flags": [],
        "expected_url_rank": None,
        "top1_title": "",
        "top1_url": "",
        "answer": answer,
        "retrieval_latency_ms": latency_ms,
        "answer_latency_ms": 0.0,
        "passed": all([answer_any and answer_all, date_pass]),
    }


def pct(numerator: int, denominator: int) -> float:
    return round((numerator / denominator) * 100, 2) if denominator else 0.0


def summarize(results: list[dict]) -> dict:
    search_results = [r for r in results if r["kind"] == "search"]
    schedule_results = [r for r in results if r["kind"] == "schedule"]
    latencies = [float(r.get("retrieval_latency_ms", 0.0)) for r in results]
    answer_latencies = [float(r.get("answer_latency_ms", 0.0)) for r in search_results]

    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "failed": sum(1 for r in results if not r["passed"]),
        "pass_rate": pct(sum(1 for r in results if r["passed"]), len(results)),
        "search_total": len(search_results),
        "schedule_total": len(schedule_results),
        "top1_url_accuracy": pct(sum(1 for r in search_results if r["top1_url_match"]), len(search_results)),
        "top3_url_accuracy": pct(sum(1 for r in search_results if r["top3_url_match"]), len(search_results)),
        "top1_title_accuracy": pct(sum(1 for r in search_results if r["top1_title_match"]), len(search_results)),
        "answer_keyword_pass_rate": pct(sum(1 for r in results if r["answer_keyword_match"]), len(results)),
        "source_url_pass_rate": pct(sum(1 for r in search_results if r["source_url_match"]), len(search_results)),
        "date_pass_rate": pct(sum(1 for r in results if r["date_match"]), len(results)),
        "hallucination_proxy_rate": pct(
            sum(1 for r in search_results if r["hallucination_detected"]),
            len(search_results),
        ),
        "avg_retrieval_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p95_retrieval_latency_ms": round(statistics.quantiles(latencies, n=20)[18], 2) if len(latencies) >= 20 else 0.0,
        "avg_answer_latency_ms": round(statistics.mean(answer_latencies), 2) if answer_latencies else 0.0,
    }

    by_category = {}
    for category in sorted({r["category"] for r in results}):
        items = [r for r in results if r["category"] == category]
        searches = [r for r in items if r["kind"] == "search"]
        by_category[category] = {
            "total": len(items),
            "pass_rate": pct(sum(1 for r in items if r["passed"]), len(items)),
            "top3_url_accuracy": pct(sum(1 for r in searches if r["top3_url_match"]), len(searches)),
            "answer_keyword_pass_rate": pct(sum(1 for r in items if r["answer_keyword_match"]), len(items)),
        }
    summary["by_category"] = by_category
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate RAG retrieval and source-grounded answer quality")
    ap.add_argument(
        "--cases",
        default=str(Path(__file__).with_name("rag_eval_cases.json")),
        help="path to RAG eval cases JSON",
    )
    ap.add_argument("--top-k", type=int, default=5, help="hybrid_search top_k for retrieval metrics")
    ap.add_argument("--answer-top-k", type=int, default=6, help="build_answer top_k for answer metrics")
    ap.add_argument("--schedule-top-k", type=int, default=5, help="schedule_search top_k for schedule cases")
    ap.add_argument("--out", default="/tmp/rag_eval_report.json", help="output report path")
    ap.add_argument("--fail-on-fail", action="store_true", help="exit 2 when any case fails")
    args = ap.parse_args()

    cases = load_cases(Path(args.cases))
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
            if r["hallucination_detected"]:
                reasons.append("hallucination_proxy")
            print(f"- {r['id']}: {', '.join(reasons) or 'failed'}")

    return 2 if failed and args.fail_on_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
