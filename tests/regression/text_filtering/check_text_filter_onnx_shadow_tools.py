#!/usr/bin/env python3
"""Lightweight regression checks for the report-only ONNX shadow harness."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT_DIR / "tests" / "regression" / "text_filtering" / "benchmark_text_filter_onnx_shadow.py"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def load_module():
    spec = importlib.util.spec_from_file_location("benchmark_text_filter_onnx_shadow", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def backend_result(predictions: list[bool], labels: list[list[int]], fp: int = 0, fn: int = 0):
    return {
        "false_positive_count": fp,
        "false_negative_count": fn,
        "case_results": [
            {"actual_has_profanity": prediction, "labels": case_labels}
            for prediction, case_labels in zip(predictions, labels)
        ],
    }


def main() -> int:
    module = load_module()
    from text_filtering.service import split_sentences as runtime_split_sentences

    samples = [
        "오늘 수업 자료를 확인했습니다. 과제를 제출했습니다.",
        "댓글에 ㅅㅂ 같은 표현을 남기면 안 됩니다.",
        "Please check the notice before tomorrow.",
    ]
    for sample in samples:
        assert module.split_sentences(sample) == runtime_split_sentences(sample), sample

    baseline = backend_result([False, True], [[0], [1]])
    matching = backend_result([False, True], [[0], [1]])
    mismatch = backend_result([False, False], [[0], [0]], fn=1)

    status, reasons = module.comparison_status({
        "pytorch": baseline,
        "onnx-fp32": matching,
        "onnx-int8": matching,
    })
    assert status == "passed", reasons

    status, reasons = module.comparison_status({
        "pytorch": baseline,
        "onnx-fp32": matching,
        "onnx-int8": mismatch,
    })
    assert status == "failed", reasons
    assert "onnx-int8:false_negative_regression" in reasons
    assert "onnx-int8:prediction_mismatch" in reasons

    assessments = module.candidate_assessments({
        "pytorch": {
            **baseline,
            "cold_start_ms": 100.0,
            "latency_ms": {"p95": 20.0},
            "peak_rss_mb": 1000.0,
        },
        "onnx-fp32": {
            **matching,
            "cold_start_ms": 50.0,
            "latency_ms": {"p95": 10.0},
            "peak_rss_mb": 800.0,
        },
        "onnx-int8": {
            **mismatch,
            "cold_start_ms": 25.0,
            "latency_ms": {"p95": 8.0},
            "peak_rss_mb": 400.0,
        },
    })
    assert assessments["onnx-fp32"]["local_quality_gate_passed"] is True
    assert assessments["onnx-fp32"]["p95_latency_change_percent"] == -50.0
    assert assessments["onnx-int8"]["local_quality_gate_passed"] is False
    print("[OK] text-filter ONNX shadow tools")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
