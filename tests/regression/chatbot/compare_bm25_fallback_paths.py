#!/usr/bin/env python3
import argparse
import gzip
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_rag_retrieval as rag_eval  # noqa: E402
from LLM.sub_model.index_utils import get_tokenizer  # noqa: E402
from LLM.sub_model.query_index_schema import normalize_search_df_schema  # noqa: E402

DEFAULT_REPORT_PATH = ROOT_DIR / "tests" / "reports" / "chatbot" / "bm25_fallback_comparison_report.json"
BASELINE_MODE = "tokenized_corpus"
MODES = (BASELINE_MODE, "runtime_tokenize")


def pct(numerator: int, denominator: int) -> float:
    return round((numerator / denominator) * 100, 2) if denominator else 0.0


def mean(values: list[float]) -> float:
    return round(statistics.mean(values), 2) if values else 0.0


def p95(values: list[float]) -> float:
    if len(values) < 20:
        return 0.0
    return round(statistics.quantiles(values, n=20)[18], 2)


def configure_default_artifact_env() -> dict[str, str]:
    rag_eval.configure_default_artifact_env()
    keys = [
        "DATA_JSON",
        "ART_DIR",
        "SEARCH_DF_PATH",
        "EMB_PATH",
        "BM25_PATH",
        "CONTACTS_CSV",
        "META_PATH",
        "CONTACT_KWS_PATH",
        "SCHEDULE_CSV_PATH",
    ]
    missing = [key for key in keys if not os.getenv(key)]
    if missing:
        raise FileNotFoundError(f"missing artifact env values: {missing}")
    env = {key: os.environ[key] for key in keys}
    env["TOK_PATH"] = os.getenv(
        "TOK_PATH",
        str(ROOT_DIR / "model" / "artifacts" / "tokenized_corpus.json.gz"),
    )
    return env


def write_tokenized_corpus(search_df_path: Path, out_path: Path) -> dict:
    search_df = normalize_search_df_schema(pd.read_parquet(search_df_path))
    tokenizer = get_tokenizer()
    started = time.perf_counter()
    tokenized = [
        tokenizer(text)
        for text in search_df["text_for_bm25"].astype(str).tolist()
    ]
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(tokenized, f, ensure_ascii=False)
    return {
        "path": str(out_path),
        "documents": len(tokenized),
        "generation_latency_ms": elapsed_ms,
    }


def mode_env(base_env: dict[str, str], mode: str, temp_dir: Path, tokenized_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(base_env)
    missing_bm25 = temp_dir / f"missing_{mode}_bm25.pkl"
    missing_tok = temp_dir / f"missing_{mode}_tokenized_corpus.json.gz"

    if mode == "tokenized_corpus":
        env["BM25_PATH"] = str(missing_bm25)
        env["TOK_PATH"] = str(tokenized_path)
    elif mode == "runtime_tokenize":
        env["BM25_PATH"] = str(missing_bm25)
        env["TOK_PATH"] = str(missing_tok)
    else:
        raise ValueError(f"unknown mode: {mode}")
    return env


def contains_any(text: str, needles: list[str]) -> bool:
    return any(needle and needle in text for needle in needles) if needles else True


def contains_all(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def evaluate_search_case(case: dict, top_k: int, answer_top_k: int) -> dict:
    from LLM.sub_model.query_index import build_answer, hybrid_search

    query = case.get("query", "")

    started = time.perf_counter()
    try:
        hits = hybrid_search(query, top_k=top_k)
        retrieval_error = ""
    except Exception as exc:
        hits = None
        retrieval_error = f"{type(exc).__name__}:{exc}"
    retrieval_ms = round((time.perf_counter() - started) * 1000, 2)

    started = time.perf_counter()
    try:
        answer_payload = build_answer(query, top_k=answer_top_k) or {}
        answer_error = ""
    except Exception as exc:
        answer_payload = {}
        answer_error = f"{type(exc).__name__}:{exc}"
    answer_ms = round((time.perf_counter() - started) * 1000, 2)

    answer = str(answer_payload.get("answer", "") or "")
    expected_urls = case.get("expected_url_contains", [])
    url_rank = rag_eval.rank_for_expected_url(hits, expected_urls)
    answer_any = contains_any(answer, case.get("answer_must_contain_any", []))
    answer_all = contains_all(answer, case.get("answer_must_contain_all", []))
    source_url_match = (
        contains_any(answer, expected_urls)
        if case.get("requires_source_url", False)
        else True
    )
    hallucination_flags = rag_eval.hallucination_flags(answer, hits, expected_urls)

    top_rows = []
    if hits is not None and not getattr(hits, "empty", True):
        for _, row in hits.head(top_k).iterrows():
            top_rows.append(
                {
                    "title": str(row.get("title", "") or ""),
                    "url": str(row.get("url", "") or ""),
                    "score": round(float(row.get("score", 0.0) or 0.0), 8),
                }
            )

    return {
        "id": case.get("id"),
        "category": case.get("category", ""),
        "query": query,
        "top_rows": top_rows,
        "top_urls": [row["url"] for row in top_rows],
        "top1_url": top_rows[0]["url"] if top_rows else "",
        "top1_title": top_rows[0]["title"] if top_rows else "",
        "expected_url_rank": url_rank,
        "top3_url_match": bool(url_rank and url_rank <= 3) if expected_urls else True,
        "top1_title_match": rag_eval.title_matches(hits, case.get("expected_title_contains", [])),
        "answer": answer,
        "answer_keyword_match": answer_any and answer_all,
        "source_url_match": source_url_match,
        "hallucination_flags": hallucination_flags,
        "retrieval_latency_ms": retrieval_ms,
        "answer_latency_ms": answer_ms,
        "errors": {
            "retrieval": retrieval_error,
            "answer": answer_error,
        },
        "passed": all(
            [
                not retrieval_error,
                not answer_error,
                bool(url_rank and url_rank <= 3) if expected_urls else True,
                answer_any and answer_all,
                source_url_match,
                not hallucination_flags,
            ]
        ),
    }


def summarize_mode(mode: str, import_latency_ms: float, results: list[dict]) -> dict:
    retrieval_latencies = [float(result["retrieval_latency_ms"]) for result in results]
    answer_latencies = [float(result["answer_latency_ms"]) for result in results]
    passed = sum(1 for result in results if result["passed"])
    failed = len(results) - passed
    return {
        "mode": mode,
        "status": "passed" if failed == 0 else "failed",
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "pass_rate": pct(passed, len(results)),
        "top3_url_accuracy": pct(sum(1 for result in results if result["top3_url_match"]), len(results)),
        "answer_keyword_pass_rate": pct(sum(1 for result in results if result["answer_keyword_match"]), len(results)),
        "source_url_pass_rate": pct(sum(1 for result in results if result["source_url_match"]), len(results)),
        "hallucination_proxy_rate": pct(sum(1 for result in results if result["hallucination_flags"]), len(results)),
        "query_index_import_latency_ms": import_latency_ms,
        "avg_retrieval_latency_ms": mean(retrieval_latencies),
        "p95_retrieval_latency_ms": p95(retrieval_latencies),
        "avg_answer_latency_ms": mean(answer_latencies),
        "p95_answer_latency_ms": p95(answer_latencies),
    }


def worker_main(args: argparse.Namespace) -> int:
    cases = [
        case
        for case in rag_eval.load_cases(Path(args.cases))
        if case.get("kind", "search") == "search"
    ]
    if args.max_cases:
        cases = cases[: args.max_cases]

    started = time.perf_counter()
    import LLM.sub_model.query_index  # noqa: F401
    import_latency_ms = round((time.perf_counter() - started) * 1000, 2)

    results = [
        evaluate_search_case(case, args.top_k, args.answer_top_k)
        for case in cases
    ]
    payload = {
        "mode": args.mode,
        "summary": summarize_mode(args.mode, import_latency_ms, results),
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def parse_worker_payload(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise ValueError(f"worker JSON payload not found in stdout: {stdout[-500:]}")


def run_worker(script_path: Path, mode: str, env: dict[str, str], args: argparse.Namespace) -> dict:
    cmd = [
        sys.executable,
        str(script_path),
        "--worker",
        "--mode",
        mode,
        "--cases",
        str(args.cases),
        "--top-k",
        str(args.top_k),
        "--answer-top-k",
        str(args.answer_top_k),
    ]
    if args.max_cases:
        cmd.extend(["--max-cases", str(args.max_cases)])

    started = time.perf_counter()
    completed = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    if completed.returncode != 0:
        return {
            "mode": mode,
            "summary": {
                "mode": mode,
                "status": "error",
                "worker_elapsed_ms": elapsed_ms,
            },
            "results": [],
            "worker_error": {
                "returncode": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            },
        }

    payload = parse_worker_payload(completed.stdout)
    payload["summary"]["worker_elapsed_ms"] = elapsed_ms
    payload["stderr"] = completed.stderr[-4000:]
    return payload


def compare_modes(mode_reports: list[dict], baseline_mode: str) -> dict:
    reports_by_mode = {report["mode"]: report for report in mode_reports}
    baseline = reports_by_mode[baseline_mode]
    baseline_by_id = {result["id"]: result for result in baseline["results"]}
    comparisons = {}

    for report in mode_reports:
        mode = report["mode"]
        if mode == baseline_mode:
            continue

        top_urls_changed = []
        top1_changed = []
        answer_changed = []
        passed_changed = []
        missing_cases = []

        for result in report["results"]:
            base = baseline_by_id.get(result["id"])
            if not base:
                missing_cases.append(result["id"])
                continue
            if result["top_urls"] != base["top_urls"]:
                top_urls_changed.append(result["id"])
            if result["top1_url"] != base["top1_url"]:
                top1_changed.append(result["id"])
            if result["answer"] != base["answer"]:
                answer_changed.append(result["id"])
            if result["passed"] != base["passed"]:
                passed_changed.append(result["id"])

        comparisons[mode] = {
            "baseline": baseline_mode,
            "load_latency_delta_ms": round(
                float(report["summary"].get("query_index_import_latency_ms", 0.0))
                - float(baseline["summary"].get("query_index_import_latency_ms", 0.0)),
                2,
            ),
            "top_urls_changed": len(top_urls_changed),
            "top_urls_changed_case_ids": top_urls_changed,
            "top1_changed": len(top1_changed),
            "top1_changed_case_ids": top1_changed,
            "answer_changed": len(answer_changed),
            "answer_changed_case_ids": answer_changed,
            "passed_changed": len(passed_changed),
            "passed_changed_case_ids": passed_changed,
            "missing_cases": missing_cases,
        }
    return comparisons


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare chatbot BM25 tokenized-corpus and runtime-tokenize fallback paths"
    )
    ap.add_argument("--cases", default=str(rag_eval.DEFAULT_CASES_PATH), help="RAG eval case file or directory")
    ap.add_argument("--top-k", type=int, default=5, help="hybrid_search top_k")
    ap.add_argument("--answer-top-k", type=int, default=6, help="build_answer top_k")
    ap.add_argument("--max-cases", type=int, default=0, help="limit search cases for a smoke comparison")
    ap.add_argument("--out", default=str(DEFAULT_REPORT_PATH), help="output report path")
    ap.add_argument("--fail-on-diff", action="store_true", help="exit 2 when runtime fallback results differ from tokenized-corpus fallback")
    ap.add_argument("--fail-on-fail", action="store_true", help="exit 2 when any mode has failing cases")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--mode", choices=MODES, default=BASELINE_MODE, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        return worker_main(args)

    cases = rag_eval.load_cases(Path(args.cases))
    validation_errors = rag_eval.validate_cases(cases)
    if validation_errors:
        print(json.dumps({"valid": False, "errors": validation_errors}, ensure_ascii=False))
        return 2

    base_env = configure_default_artifact_env()
    script_path = Path(__file__).resolve()

    with tempfile.TemporaryDirectory(prefix="bm25_fallback_compare_") as td:
        temp_dir = Path(td)
        tokenized_info = write_tokenized_corpus(
            Path(base_env["SEARCH_DF_PATH"]),
            temp_dir / "tokenized_corpus.json.gz",
        )
        mode_reports = [
            run_worker(script_path, mode, mode_env(base_env, mode, temp_dir, Path(tokenized_info["path"])), args)
            for mode in MODES
        ]

    comparisons = compare_modes(mode_reports, baseline_mode=BASELINE_MODE)
    worker_errors = [report for report in mode_reports if report["summary"].get("status") == "error"]
    failed_modes = [report for report in mode_reports if report["summary"].get("failed", 0)]
    changed = [
        mode
        for mode, comparison in comparisons.items()
        if comparison["top_urls_changed"]
        or comparison["top1_changed"]
        or comparison["answer_changed"]
        or comparison["passed_changed"]
    ]

    report = {
        "schema_version": 1,
        "suite": "bm25_fallback_comparison",
        "service": "chatbot",
        "cases_file": str(Path(args.cases)),
        "search_cases_evaluated": mode_reports[0]["summary"].get("total", 0) if mode_reports else 0,
        "tokenized_corpus_fixture": tokenized_info,
        "summary": {
            "status": "failed" if worker_errors or failed_modes or changed else "passed",
            "modes": [report["summary"] for report in mode_reports],
            "changed_modes": changed,
            "worker_error_modes": [report["mode"] for report in worker_errors],
            "failed_case_modes": [report["mode"] for report in failed_modes],
        },
        "comparisons": comparisons,
        "mode_reports": mode_reports,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"report={out_path}")

    if worker_errors:
        return 2
    if args.fail_on_fail and failed_modes:
        return 2
    if args.fail_on_diff and changed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
