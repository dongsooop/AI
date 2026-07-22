#!/usr/bin/env python3
"""Regression checks for OCI load generation and sanitized summaries."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    load_tool = load_module(
        "run_ai_service_load",
        ROOT_DIR / "tests" / "regression" / "operations" / "run_ai_service_load.py",
    )
    summary_tool = load_module(
        "summarize_oci_ai_baseline",
        ROOT_DIR / "scripts" / "summarize_oci_ai_baseline.py",
    )
    base_url = "http://private-host.invalid"
    secret = "regression-token-must-not-leak"
    os.environ["AI_BASELINE_BEARER_TOKEN"] = secret

    def fake_send_request(spec, timeout):
        return {
            "service": spec.service,
            "status_code": 200,
            "latency_ms": 5.0,
            "success": True,
            "error_type": None,
        }

    def fake_fetch_readiness(url, timeout):
        return {
            "ok": True,
            "status_code": 200,
            "queue_size": 0,
            "queue_max_size": 20,
            "queue_utilization": 0.0,
        }

    original_send_request = load_tool.send_request
    original_fetch_readiness = load_tool.fetch_readiness
    load_tool.send_request = fake_send_request
    load_tool.fetch_readiness = fake_fetch_readiness
    try:
        args = argparse.Namespace(
            scenario="mixed",
            profile="oci-a1-flex-shared",
            concurrency=2,
            duration=None,
            requests_per_service=4,
            timeout=3.0,
            main_base_url=base_url,
            chatbot_base_url=base_url,
            token_file=None,
            ocr_image=None,
            allow_ocr_api_side_effects=False,
            chatbot_query="private-query-must-not-leak",
            text_filter_text="private-text-must-not-leak",
            cache_state="warm",
            readiness_interval=0.01,
            max_consecutive_failures=3,
            max_error_rate=0.25,
            min_error_rate_samples=4,
            max_queue_utilization=0.9,
            max_host_memory_utilization=1.0,
        )
        # Mixed defaults include OCR. Restrict this mock check to chatbot and text-filter
        # by exercising their specs and worker machinery separately.
        original_services = load_tool.SERVICES
        load_tool.SERVICES = ("chatbot", "text-filter")
        report = asyncio.run(load_tool.run_load(args))
        load_tool.SERVICES = original_services
    finally:
        load_tool.send_request = original_send_request
        load_tool.fetch_readiness = original_fetch_readiness
        os.environ.pop("AI_BASELINE_BEARER_TOKEN", None)

    assert report["status"] == "completed", report
    assert report["summary"]["chatbot"]["request_count"] == 4, report
    assert report["summary"]["text-filter"]["request_count"] == 4, report
    serialized = json.dumps(report, ensure_ascii=False)
    assert secret not in serialized
    assert base_url not in serialized
    assert "private-query-must-not-leak" not in serialized
    assert "private-text-must-not-leak" not in serialized

    ocr_args = argparse.Namespace(
        allow_ocr_api_side_effects=False,
        ocr_image=str(ROOT_DIR / "docs" / "OCI_AI_PERFORMANCE_BASELINE.md"),
        concurrency=2,
        duration=None,
        requests_per_service=2,
        main_base_url=base_url,
    )
    try:
        load_tool.build_specs(ocr_args, ["ocr"], ["token-a", "token-b"])
    except ValueError as exc:
        assert "side-effects" in str(exc)
    else:
        raise AssertionError("OCR side-effect guard did not reject the request")
    ocr_args.allow_ocr_api_side_effects = True
    ocr_args.duration = 10
    try:
        load_tool.build_specs(ocr_args, ["ocr"], ["token-a", "token-b"])
    except ValueError as exc:
        assert "--requests-per-service" in str(exc)
    else:
        raise AssertionError("OCR duration guard did not reject the request")

    invalid_rag = {
        "summary": {"status": "failed", "top1_url_accuracy": 0.0, "p95_retrieval_latency_ms": 1.0},
        "results": [{"errors": {"import": "dependency unavailable"}}],
    }
    rows = summary_tool.baseline_rows(None, invalid_rag, None)
    assert rows[0]["status"] == "environment_invalid"
    assert rows[0]["p95_ms"] is None
    sanitized = summary_tool.sanitized_runtime({
        "host": {"architecture": "arm64"},
        "processes": {"ollama": {"pid": 123, "peak_rss_mb": 100}},
        "logs": {"chatbot": {"path": "/private/log", "runtime_events": {"event": {"failed_rate": 0}}}},
    })
    serialized_sanitized = json.dumps(sanitized)
    assert "123" not in serialized_sanitized
    assert "/private/log" not in serialized_sanitized
    print("[OK] OCI AI baseline tools")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
