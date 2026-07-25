#!/usr/bin/env python3
"""Compare PyTorch, ONNX FP32, and ONNX INT8 text-filter inference in isolation."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CASES_DIR = ROOT_DIR / "tests" / "regression" / "text_filtering" / "cases"
DEFAULT_MODEL_DIR = ROOT_DIR / "model" / "my_electra_finetuned"
DEFAULT_ONNX_DIR = ROOT_DIR / "model" / "generated" / "text_filter_onnx"
DEFAULT_REPORT_PATH = ROOT_DIR / "tests" / "reports" / "text_filtering" / "text_filter_onnx_shadow_report.json"
DEFAULT_MARKDOWN_PATH = ROOT_DIR / "tests" / "reports" / "text_filtering" / "text_filter_onnx_shadow_report.md"
ENGLISH_BAD_WORDS_PATH = ROOT_DIR / "data" / "eng_bad_text.txt"
BACKENDS = ("pytorch", "onnx-fp32", "onnx-int8")
SENTENCE_ENDINGS = (
    "다", "요", "죠", "네", "습니다", "습니까", "해요", "했어요", "하였습니다", "하네요", "해봐요",
)


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


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"([.!?])\s+", r"\1\n", text)
    for ending in SENTENCE_ENDINGS:
        text = re.sub(rf"({ending})(?=\s)", r"\1\n", text)
    return [sentence.strip() for sentence in text.split("\n") if sentence.strip()]


def load_english_bad_words() -> set[str]:
    return {
        line.strip().lower()
        for line in ENGLISH_BAD_WORDS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def load_cases(path: Path) -> list[dict[str, Any]]:
    paths = sorted(path.glob("*.json")) if path.is_dir() else [path]
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case_path in paths:
        payload = json.loads(case_path.read_text(encoding="utf-8"))
        items = payload.get("cases", payload) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise ValueError(f"cases must be a list: {case_path}")
        for item in items:
            case = dict(item)
            case_id = str(case.get("id", ""))
            if not case_id or case_id in seen:
                raise ValueError(f"missing or duplicate case id: {case_id or case_path}")
            seen.add(case_id)
            case["source_file"] = str(case_path.relative_to(ROOT_DIR))
            cases.append(case)
    if not cases:
        raise ValueError(f"no cases found: {path}")
    return cases


def make_predictor(
    backend: str,
    model_dir: Path,
    onnx_dir: Path,
) -> tuple[Any, Callable[[str], int]]:
    from transformers import ElectraTokenizer

    tokenizer = ElectraTokenizer.from_pretrained(model_dir, local_files_only=True)
    if backend == "pytorch":
        import torch
        from transformers import ElectraForSequenceClassification

        model = ElectraForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        model.to("cpu")
        model.eval()

        def predict(sentence: str) -> int:
            encoded = tokenizer(
                sentence,
                add_special_tokens=True,
                max_length=64,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                ).logits
            return int(torch.argmax(logits, dim=-1).item())

        return model, predict

    try:
        import numpy as np
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"missing {exc.name}; install requirements-onnx-shadow.txt"
        ) from exc

    model_name = "text_filter_fp32.onnx" if backend == "onnx-fp32" else "text_filter_int8.onnx"
    session = ort.InferenceSession(str(onnx_dir / model_name), providers=["CPUExecutionProvider"])

    def predict(sentence: str) -> int:
        encoded = tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        logits = session.run(
            ["logits"],
            {
                "input_ids": encoded["input_ids"].astype(np.int64, copy=False),
                "attention_mask": encoded["attention_mask"].astype(np.int64, copy=False),
            },
        )[0]
        return int(np.argmax(logits, axis=-1)[0])

    return session, predict


def run_worker(args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.cases))
    english_bad_words = load_english_bad_words()
    load_started = time.perf_counter()
    runtime, predict = make_predictor(args.backend, Path(args.model_dir), Path(args.onnx_dir))
    cold_start_ms = round((time.perf_counter() - load_started) * 1000, 2)

    def evaluate(text: str) -> tuple[bool, list[int]]:
        labels = [predict(sentence) for sentence in split_sentences(text)]
        english_override = any(word in text.lower() for word in english_bad_words)
        return any(label == 1 for label in labels) or english_override, labels

    for _ in range(args.warmup):
        for case in cases:
            evaluate(str(case.get("text", "")))

    all_latencies: list[float] = []
    case_results: list[dict[str, Any]] = []
    false_positive_count = 0
    false_negative_count = 0
    for case in cases:
        text = str(case.get("text", ""))
        latencies: list[float] = []
        actual = False
        labels: list[int] = []
        for _ in range(args.repeat):
            started = time.perf_counter()
            actual, labels = evaluate(text)
            latency_ms = (time.perf_counter() - started) * 1000
            latencies.append(latency_ms)
            all_latencies.append(latency_ms)
        expected = bool(case.get("expected", {}).get("has_profanity"))
        false_positive_count += int(actual and not expected)
        false_negative_count += int(expected and not actual)
        case_results.append({
            "id": case.get("id"),
            "category": case.get("category"),
            "source_file": case.get("source_file"),
            "expected_has_profanity": expected,
            "actual_has_profanity": actual,
            "labels": labels,
            "latency_ms": {
                "p50": percentile(latencies, 50),
                "p95": percentile(latencies, 95),
                "max": round(max(latencies), 2),
            },
        })

    failed = false_positive_count + false_negative_count
    result = {
        "backend": args.backend,
        "status": "completed",
        "total": len(cases),
        "passed": len(cases) - failed,
        "failed": failed,
        "pass_rate": round((len(cases) - failed) / len(cases), 4),
        "false_positive_count": false_positive_count,
        "false_negative_count": false_negative_count,
        "cold_start_ms": cold_start_ms,
        "latency_ms": {
            "average": round(statistics.mean(all_latencies), 2),
            "p50": percentile(all_latencies, 50),
            "p95": percentile(all_latencies, 95),
            "max": round(max(all_latencies), 2),
        },
        "peak_rss_mb": peak_rss_mb(),
        "case_results": case_results,
    }
    del runtime
    Path(args.worker_out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def comparison_status(results: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    baseline = results["pytorch"]
    reasons: list[str] = []
    for backend in ("onnx-fp32", "onnx-int8"):
        candidate = results[backend]
        if candidate["false_positive_count"] > baseline["false_positive_count"]:
            reasons.append(f"{backend}:false_positive_regression")
        if candidate["false_negative_count"] > baseline["false_negative_count"]:
            reasons.append(f"{backend}:false_negative_regression")
        if any(
            item["actual_has_profanity"] != baseline_item["actual_has_profanity"]
            or item["labels"] != baseline_item["labels"]
            for item, baseline_item in zip(candidate["case_results"], baseline["case_results"])
        ):
            reasons.append(f"{backend}:prediction_mismatch")
    return ("passed" if not reasons else "failed"), reasons


def percent_change(candidate: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return round((candidate - baseline) / baseline * 100, 2)


def candidate_assessments(results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    baseline = results["pytorch"]
    assessments: dict[str, dict[str, Any]] = {}
    for backend in ("onnx-fp32", "onnx-int8"):
        candidate = results[backend]
        prediction_match = all(
            item["actual_has_profanity"] == baseline_item["actual_has_profanity"]
            and item["labels"] == baseline_item["labels"]
            for item, baseline_item in zip(candidate["case_results"], baseline["case_results"])
        )
        assessments[backend] = {
            "prediction_match": prediction_match,
            "false_positive_delta": candidate["false_positive_count"] - baseline["false_positive_count"],
            "false_negative_delta": candidate["false_negative_count"] - baseline["false_negative_count"],
            "cold_start_change_percent": percent_change(
                candidate["cold_start_ms"], baseline["cold_start_ms"]
            ),
            "p95_latency_change_percent": percent_change(
                candidate["latency_ms"]["p95"], baseline["latency_ms"]["p95"]
            ),
            "peak_rss_change_percent": percent_change(
                candidate["peak_rss_mb"], baseline["peak_rss_mb"]
            ),
            "local_quality_gate_passed": (
                prediction_match
                and candidate["false_positive_count"] <= baseline["false_positive_count"]
                and candidate["false_negative_count"] <= baseline["false_negative_count"]
            ),
            "promotion_eligible": False,
            "remaining_gate": "arm64_docker_not_validated",
        }
    return assessments


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Text Filter ONNX Shadow Report",
        "",
        f"Status: `{report['status']}`",
        "",
        "| Backend | Pass rate | FP | FN | Cold start ms | p95 ms | Peak RSS MB | Prediction match |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for backend in BACKENDS:
        result = report["backends"][backend]
        assessment = report["candidate_assessments"].get(backend, {})
        prediction_match = "-" if backend == "pytorch" else str(assessment["prediction_match"]).lower()
        lines.append(
            f"| {backend} | {result['pass_rate']} | {result['false_positive_count']} | "
            f"{result['false_negative_count']} | {result['cold_start_ms']} | "
            f"{result['latency_ms']['p95']} | {result['peak_rss_mb']} | {prediction_match} |"
        )
    lines.extend(["", "## Change from PyTorch", ""])
    for backend, assessment in report["candidate_assessments"].items():
        lines.append(
            f"- `{backend}`: cold start {assessment['cold_start_change_percent']}%, "
            f"p95 {assessment['p95_latency_change_percent']}%, "
            f"peak RSS {assessment['peak_rss_change_percent']}%, "
            f"local quality gate `{str(assessment['local_quality_gate_passed']).lower()}`"
        )
    lines.extend([
        "",
        "## Promotion",
        "",
        "- Runtime backend was not changed.",
        "- ARM64 Docker validation is required before adding a feature flag.",
    ])
    return "\n".join(lines) + "\n"


def run_parent(args: argparse.Namespace) -> int:
    script_path = Path(__file__).resolve()
    results: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="text-filter-onnx-shadow-") as temp_dir:
        for backend in BACKENDS:
            worker_out = Path(temp_dir) / f"{backend}.json"
            command = [
                sys.executable,
                str(script_path),
                "--worker",
                "--backend",
                backend,
                "--cases",
                args.cases,
                "--model-dir",
                args.model_dir,
                "--onnx-dir",
                args.onnx_dir,
                "--repeat",
                str(args.repeat),
                "--warmup",
                str(args.warmup),
                "--worker-out",
                str(worker_out),
            ]
            worker_env = os.environ.copy()
            if backend != "pytorch":
                worker_env.update({"USE_TORCH": "0", "USE_TF": "0", "USE_FLAX": "0"})
            completed = subprocess.run(
                command,
                check=False,
                text=True,
                capture_output=True,
                env=worker_env,
            )
            if completed.returncode != 0 or not worker_out.exists():
                raise RuntimeError(
                    f"{backend} benchmark failed with exit {completed.returncode}: "
                    f"{completed.stderr.strip() or completed.stdout.strip()}"
                )
            results[backend] = json.loads(worker_out.read_text(encoding="utf-8"))

    status, reasons = comparison_status(results)
    assessments = candidate_assessments(results)
    report = {
        "schema_version": 1,
        "suite": "text_filter_onnx_shadow",
        "service": "text_filtering",
        "status": status,
        "report_only": True,
        "production_backend_changed": False,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "cases_path": str(Path(args.cases).resolve().relative_to(ROOT_DIR)),
        "repeat": args.repeat,
        "warmup": args.warmup,
        "promotion_recommendation": {
            "eligible": False,
            "reasons": reasons + ["arm64_docker_not_validated"],
            "notes": "Feature flag remains out of scope until quality, latency, RSS, and ARM64 Docker are validated.",
        },
        "candidate_assessments": assessments,
        "backends": results,
    }
    out_path = Path(args.out)
    markdown_path = Path(args.markdown_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print("[OK] text-filter ONNX shadow comparison" if status == "passed" else "[WARN] ONNX prediction mismatch")
    print(json.dumps({
        "status": status,
        "promotion_recommendation": report["promotion_recommendation"],
        "backends": {
            name: {
                key: value for key, value in result.items()
                if key in {
                    "pass_rate", "false_positive_count", "false_negative_count",
                    "cold_start_ms", "latency_ms", "peak_rss_mb",
                }
            }
            for name, result in results.items()
        },
    }, ensure_ascii=False))
    print(f"report={out_path}")
    print(f"markdown_report={markdown_path}")
    return 0 if status == "passed" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark text-filter ONNX shadow backends")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--onnx-dir", default=str(DEFAULT_ONNX_DIR))
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_PATH))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0:
        parser.error("--repeat must be >= 1 and --warmup must be >= 0")
    if args.worker and (not args.backend or not args.worker_out):
        parser.error("--worker requires --backend and --worker-out")
    return args


def main() -> int:
    args = parse_args()
    return run_worker(args) if args.worker else run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
