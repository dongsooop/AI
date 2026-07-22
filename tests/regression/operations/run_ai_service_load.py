#!/usr/bin/env python3
"""Run bounded, sanitized HTTP load scenarios against Dongsooop AI services."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_PATH = ROOT_DIR / "tests" / "reports" / "operations" / "ai_service_load_report.json"
DEFAULT_CHATBOT_QUERY = "컴퓨터소프트웨어공학과 소개를 알려줘"
DEFAULT_TEXT_FILTER_TEXT = "오늘 수업 자료를 확인하고 과제를 제출했습니다."
SERVICES = ("ocr", "chatbot", "text-filter")


@dataclass(frozen=True)
class RequestSpec:
    service: str
    url: str
    headers: dict[str, str]
    body: bytes
    expected_statuses: frozenset[int]


def percentile(values: list[float], value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((value / 100) * len(ordered)) - 1))
    return round(ordered[index], 2)


def latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": round(max(values), 2) if values else None,
    }


def load_tokens(token_file: Path | None) -> list[str]:
    tokens: list[str] = []
    env_token = os.getenv("AI_BASELINE_BEARER_TOKEN", "").strip()
    if env_token:
        tokens.append(env_token)
    if token_file:
        tokens.extend(
            line.strip()
            for line in token_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return list(dict.fromkeys(tokens))


def json_spec(service: str, url: str, payload: dict[str, Any], token: str | None) -> RequestSpec:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return RequestSpec(
        service=service,
        url=url,
        headers=headers,
        body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        expected_statuses=frozenset({200}),
    )


def multipart_spec(url: str, image_path: Path, token: str) -> RequestSpec:
    boundary = f"----dongsooop-baseline-{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'.encode(),
        f"Content-Type: {content_type}\r\n\r\n".encode(),
        image_path.read_bytes(),
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    return RequestSpec(
        service="ocr",
        url=url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        body=body,
        expected_statuses=frozenset({200, 202}),
    )


def send_request(spec: RequestSpec, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    headers = {**spec.headers, "X-Request-ID": str(uuid.uuid4())}
    request = urllib.request.Request(spec.url, data=spec.body, headers=headers, method="POST")
    status_code: int | None = None
    error_type: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = response.status
            response.read()
    except urllib.error.HTTPError as exc:
        status_code = exc.code
        error_type = "http_error"
    except TimeoutError:
        error_type = "timeout"
    except Exception as exc:
        error_type = type(exc).__name__
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "service": spec.service,
        "status_code": status_code,
        "latency_ms": latency_ms,
        "success": status_code in spec.expected_statuses,
        "error_type": error_type,
    }


def fetch_readiness(url: str, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            timetable = payload.get("components", {}).get("timetable", {})
            queue_size = timetable.get("queue_size")
            queue_max = timetable.get("queue_max_size")
            queue_utilization = None
            if isinstance(queue_size, (int, float)) and isinstance(queue_max, (int, float)) and queue_max > 0:
                queue_utilization = float(queue_size) / float(queue_max)
            return {
                "ok": response.status == 200 and payload.get("status") in {"ready", "ok"},
                "status_code": response.status,
                "queue_size": queue_size,
                "queue_max_size": queue_max,
                "queue_utilization": queue_utilization,
            }
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status_code": exc.code, "error_type": "http_error"}
    except Exception as exc:
        return {"ok": False, "status_code": None, "error_type": type(exc).__name__}


def host_memory_utilization() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    values: dict[str, int] = {}
    for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        try:
            values[key] = int(raw.strip().split()[0])
        except (ValueError, IndexError):
            continue
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if not total or available is None:
        return None
    return round((total - available) / total, 4)


def build_specs(args: argparse.Namespace, services: list[str], tokens: list[str]) -> dict[str, list[RequestSpec]]:
    specs: dict[str, list[RequestSpec]] = {}
    token = tokens[0] if tokens else None
    if "chatbot" in services:
        if not token:
            raise ValueError("chatbot load requires AI_BASELINE_BEARER_TOKEN or --token-file")
        specs["chatbot"] = [json_spec(
            "chatbot",
            f"{args.chatbot_base_url.rstrip('/')}/chatbot",
            {"text": args.chatbot_query},
            token,
        )]
    if "text-filter" in services:
        specs["text-filter"] = [json_spec(
            "text-filter",
            f"{args.main_base_url.rstrip('/')}/text_filter_single",
            {"text": args.text_filter_text},
            None,
        )]
    if "ocr" in services:
        if not args.allow_ocr_api_side_effects:
            raise ValueError("OCR API can post schedules to Spring; pass --allow-ocr-api-side-effects after confirming the target")
        if not args.ocr_image:
            raise ValueError("OCR load requires --ocr-image")
        image_path = Path(args.ocr_image)
        if not image_path.exists():
            raise ValueError(f"OCR image not found: {image_path}")
        if args.duration is not None:
            raise ValueError(
                "OCR jobs remain active after HTTP 202 and have no status endpoint; "
                "use --requests-per-service with one distinct JWT subject per request"
            )
        if len(tokens) < args.requests_per_service:
            raise ValueError(
                "OCR load requires one distinct JWT subject per queued request; "
                f"need {args.requests_per_service}, got {len(tokens)}"
            )
        specs["ocr"] = [
            multipart_spec(f"{args.main_base_url.rstrip('/')}/timetable_analysis", image_path, worker_token)
            for worker_token in tokens[:args.requests_per_service]
        ]
    return specs


async def run_load(args: argparse.Namespace) -> dict[str, Any]:
    services = list(SERVICES) if args.scenario == "mixed" else [args.scenario]
    tokens = load_tokens(Path(args.token_file) if args.token_file else None)
    specs = build_specs(args, services, tokens)
    results: list[dict[str, Any]] = []
    readiness_samples: list[dict[str, Any]] = []
    memory_samples: list[float] = []
    stop_event = asyncio.Event()
    counters: Counter[str] = Counter()
    lock = asyncio.Lock()
    started = time.monotonic()
    stop_reason: str | None = None

    async def readiness_monitor() -> None:
        nonlocal stop_reason
        consecutive_failures: Counter[str] = Counter()
        consecutive_queue_pressure: Counter[str] = Counter()
        consecutive_memory_pressure = 0
        urls = []
        if any(service in services for service in ("ocr", "text-filter")):
            urls.append(("main-api", f"{args.main_base_url.rstrip('/')}/ready"))
        if "chatbot" in services:
            urls.append(("chatbot-api", f"{args.chatbot_base_url.rstrip('/')}/ready"))
        while not stop_event.is_set():
            utilization = host_memory_utilization()
            if utilization is not None:
                memory_samples.append(utilization)
                if utilization >= args.max_host_memory_utilization:
                    consecutive_memory_pressure += 1
                else:
                    consecutive_memory_pressure = 0
                if consecutive_memory_pressure >= args.max_consecutive_failures:
                    stop_reason = "host_memory_pressure_limit"
                    stop_event.set()
                    return
            for name, url in urls:
                sample = await asyncio.to_thread(fetch_readiness, url, min(args.timeout, 3.0))
                sample["service"] = name
                readiness_samples.append(sample)
                consecutive_failures[name] = 0 if sample.get("ok") else consecutive_failures[name] + 1
                utilization = sample.get("queue_utilization")
                if isinstance(utilization, float) and utilization >= args.max_queue_utilization:
                    consecutive_queue_pressure[name] += 1
                else:
                    consecutive_queue_pressure[name] = 0
                if consecutive_failures[name] >= args.max_consecutive_failures:
                    stop_reason = f"readiness_failure_limit:{name}"
                    stop_event.set()
                    return
                if consecutive_queue_pressure[name] >= args.max_consecutive_failures:
                    stop_reason = f"queue_pressure_limit:{name}"
                    stop_event.set()
                    return
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=args.readiness_interval)
            except asyncio.TimeoutError:
                pass

    async def worker(service: str, worker_index: int) -> None:
        nonlocal stop_reason
        service_specs = specs[service]
        while not stop_event.is_set():
            async with lock:
                if args.duration is None and counters[service] >= args.requests_per_service:
                    return
                counters[service] += 1
                request_index = counters[service]
            if args.duration is not None and time.monotonic() - started >= args.duration:
                return
            spec_index = request_index - 1 if service == "ocr" else worker_index
            spec = service_specs[spec_index % len(service_specs)]
            result = await asyncio.to_thread(send_request, spec, args.timeout)
            result["request_index"] = request_index
            results.append(result)
            service_results = [item for item in results if item["service"] == service]
            recent = service_results[-args.min_error_rate_samples:]
            if len(recent) >= args.min_error_rate_samples:
                error_rate = sum(not item["success"] for item in recent) / len(recent)
                if error_rate > args.max_error_rate:
                    stop_reason = f"error_rate_limit:{service}"
                    stop_event.set()
                    return

    monitor = asyncio.create_task(readiness_monitor())
    workers = [
        asyncio.create_task(worker(service, worker_index))
        for service in services
        for worker_index in range(args.concurrency)
    ]
    await asyncio.gather(*workers)
    stop_event.set()
    await monitor

    elapsed = round(time.monotonic() - started, 3)
    service_summaries: dict[str, Any] = {}
    for service in services:
        items = [item for item in results if item["service"] == service]
        successful = [item for item in items if item["success"]]
        latencies = [float(item["latency_ms"]) for item in successful]
        errors = [item for item in items if not item["success"]]
        service_summaries[service] = {
            "request_count": len(items),
            "success_count": len(successful),
            "error_count": len(errors),
            "error_rate": round(len(errors) / len(items), 4) if items else None,
            "timeout_count": sum(item.get("error_type") == "timeout" for item in errors),
            "latency_ms": latency_summary(latencies),
            "throughput_rps": round(len(items) / elapsed, 3) if elapsed else None,
            "status_codes": dict(Counter(str(item.get("status_code")) for item in items)),
            "error_types": dict(Counter(str(item.get("error_type")) for item in errors)),
        }

    queue_utilizations = [
        float(sample["queue_utilization"])
        for sample in readiness_samples
        if isinstance(sample.get("queue_utilization"), (int, float))
    ]
    return {
        "schema_version": 1,
        "suite": "ai_service_load",
        "service": "all" if len(services) > 1 else services[0],
        "status": "stopped" if stop_reason else "completed",
        "scenario": args.scenario,
        "profile": args.profile,
        "concurrency_per_service": args.concurrency,
        "cache_state": args.cache_state,
        "duration_target_seconds": args.duration,
        "requests_per_service_target": None if args.duration is not None else args.requests_per_service,
        "elapsed_seconds": elapsed,
        "stop_reason": stop_reason,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "summary": service_summaries,
        "readiness": {
            "sample_count": len(readiness_samples),
            "failure_count": sum(not sample.get("ok") for sample in readiness_samples),
            "max_queue_utilization": round(max(queue_utilizations), 4) if queue_utilizations else None,
            "max_host_memory_utilization": round(max(memory_samples), 4) if memory_samples else None,
        },
        "privacy": {
            "request_bodies_recorded": False,
            "tokens_recorded": False,
            "base_urls_recorded": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded Dongsooop AI service load scenarios")
    parser.add_argument("--scenario", choices=(*SERVICES, "mixed"), required=True)
    parser.add_argument("--profile", default="oci-a1-flex-shared")
    parser.add_argument("--concurrency", type=int, default=1, help="workers per selected service")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--duration", type=float, default=None)
    target.add_argument("--requests-per-service", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--main-base-url", default=os.getenv("MAIN_API_BASE_URL", "http://127.0.0.1:8888"))
    parser.add_argument("--chatbot-base-url", default=os.getenv("CHATBOT_API_BASE_URL", "http://127.0.0.1:8010"))
    parser.add_argument("--token-file", help="local file with one bearer token per line; never written to reports")
    parser.add_argument("--ocr-image")
    parser.add_argument("--allow-ocr-api-side-effects", action="store_true")
    parser.add_argument("--chatbot-query", default=DEFAULT_CHATBOT_QUERY)
    parser.add_argument("--text-filter-text", default=DEFAULT_TEXT_FILTER_TEXT)
    parser.add_argument("--cache-state", choices=("cold", "warm", "mixed", "unknown"), default="unknown")
    parser.add_argument("--readiness-interval", type=float, default=1)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--max-error-rate", type=float, default=0.25)
    parser.add_argument("--min-error-rate-samples", type=int, default=10)
    parser.add_argument("--max-queue-utilization", type=float, default=0.9)
    parser.add_argument("--max-host-memory-utilization", type=float, default=0.95)
    parser.add_argument("--out", default=str(DEFAULT_REPORT_PATH))
    args = parser.parse_args()
    if args.concurrency < 1 or args.requests_per_service < 1:
        parser.error("--concurrency and --requests-per-service must be at least 1")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.timeout <= 0 or args.readiness_interval <= 0:
        parser.error("--timeout and --readiness-interval must be positive")
    if not all(0 <= value <= 1 for value in (
        args.max_error_rate,
        args.max_queue_utilization,
        args.max_host_memory_utilization,
    )):
        parser.error("rate, queue, and memory utilization limits must be between 0 and 1")
    return args


def main() -> int:
    args = parse_args()
    report = asyncio.run(run_load(args))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] AI service load report" if report["status"] == "completed" else "[STOP] AI service load report")
    print(json.dumps({
        "status": report["status"],
        "scenario": report["scenario"],
        "stop_reason": report["stop_reason"],
        "summary": report["summary"],
    }, ensure_ascii=False))
    print(f"report={out_path}")
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
