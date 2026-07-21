#!/usr/bin/env python3
"""Collect OCI host, process, readiness/queue, and runtime-log baseline metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = ROOT_DIR / "tests" / "reports" / "operations" / "oci_ai_runtime_baseline.json"
KEY_VALUE_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)=([^\s]+)")


def parse_mapping(values: list[str], option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} must use NAME=VALUE: {value}")
        name, mapped = value.split("=", 1)
        if not name.strip() or not mapped.strip():
            raise ValueError(f"{option} must use non-empty NAME=VALUE: {value}")
        result[name.strip()] = mapped.strip()
    return result


def percentile(values: list[float], value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((value / 100) * len(ordered)) - 1))
    return round(ordered[index], 2)


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": round(max(values), 2) if values else None,
    }


def cpu_model() -> str:
    candidates = [platform.processor()]
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith(("model name", "hardware")) and ":" in line:
                candidates.insert(0, line.split(":", 1)[1].strip())
                break
    return next((candidate for candidate in candidates if candidate), "unknown")


def total_ram_mb() -> int | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 * 1024))
    except (ValueError, OSError, AttributeError):
        return None


def sample_process(pid: int) -> dict[str, float] | None:
    try:
        completed = subprocess.run(
            ["ps", "-o", "%cpu=,rss=", "-p", str(pid)],
            check=True,
            capture_output=True,
            text=True,
        )
        fields = completed.stdout.strip().split()
        if len(fields) < 2:
            return None
        return {"cpu_percent": float(fields[0]), "rss_mb": round(float(fields[1]) / 1024, 2)}
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def fetch_readiness(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return {"ok": response.status == 200, "status_code": response.status, "payload": payload}
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            payload = {}
        return {"ok": False, "status_code": exc.code, "payload": payload}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}:{exc}", "payload": {}}


def parse_runtime_log(path: Path) -> dict[str, Any]:
    events: dict[str, list[dict[str, str]]] = defaultdict(list)
    request_latencies: dict[str, list[float]] = defaultdict(list)
    request_errors: Counter[str] = Counter()
    request_counts: Counter[str] = Counter()
    active_requests: set[str] = set()
    max_concurrent_requests = 0

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [part.strip() for part in line.split("|")]
        request_id = parts[3] if len(parts) >= 4 else ""
        fields = dict(KEY_VALUE_RE.findall(line))
        if "request_started" in line:
            active_requests.add(request_id or f"line-{len(active_requests)}")
            max_concurrent_requests = max(max_concurrent_requests, len(active_requests))
        if "request_completed" in line:
            path_name = fields.get("path", "unknown")
            request_counts[path_name] += 1
            try:
                request_latencies[path_name].append(float(fields.get("duration_ms", "0")))
            except ValueError:
                pass
            try:
                if int(fields.get("status_code", "0")) >= 500:
                    request_errors[path_name] += 1
            except ValueError:
                pass
            active_requests.discard(request_id)
        elif "request_failed" in line:
            path_name = fields.get("path", "unknown")
            request_counts[path_name] += 1
            request_errors[path_name] += 1
            active_requests.discard(request_id)
        if fields.get("event"):
            events[fields["event"]].append(fields)

    runtime_summary: dict[str, Any] = {}
    for event, items in events.items():
        durations: list[float] = []
        for item in items:
            try:
                durations.append(float(item.get("duration_ms", "0")))
            except ValueError:
                continue
        failed = sum(item.get("status") == "failed" for item in items)
        fallback = sum(item.get("fallback") == "true" or item.get("status") == "fallback" for item in items)
        runtime_summary[event] = {
            "latency_ms": summarize_values(durations),
            "failed_count": failed,
            "failed_rate": round(failed / len(items), 4) if items else 0.0,
            "fallback_count": fallback,
            "fallback_rate": round(fallback / len(items), 4) if items else 0.0,
            "fallback_reasons": dict(Counter(item.get("fallback_reason", "none") for item in items)),
            "bm25_fallback_tiers": dict(Counter(
                item["bm25_fallback_tier"] for item in items if "bm25_fallback_tier" in item
            )),
        }

    request_summary = {}
    for path_name in sorted(request_counts):
        latencies = request_latencies[path_name]
        error_count = request_errors[path_name]
        request_summary[path_name] = {
            "latency_ms": summarize_values(latencies),
            "request_count": request_counts[path_name],
            "error_count": error_count,
            "error_rate": round(error_count / request_counts[path_name], 4) if request_counts[path_name] else None,
        }
    return {
        "path": str(path),
        "request_metrics": request_summary,
        "runtime_events": runtime_summary,
        "max_concurrent_requests": max_concurrent_requests,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect an OCI AI runtime baseline during a bounded load window")
    parser.add_argument("--duration", type=float, default=60, help="sampling window in seconds")
    parser.add_argument("--interval", type=float, default=1, help="process/readiness sampling interval")
    parser.add_argument("--process", action="append", default=[], metavar="NAME=PID", help="process to sample")
    parser.add_argument("--readiness", action="append", default=[], metavar="NAME=URL", help="readiness URL to sample")
    parser.add_argument("--runtime-log", action="append", default=[], metavar="NAME=PATH", help="measurement-window log")
    parser.add_argument("--ocpus", type=float, default=None, help="OCI shape OCPU count from the console/API")
    parser.add_argument("--main-chatbot-topology", choices=("same", "separate", "unknown"), default="unknown")
    parser.add_argument("--ollama-topology", choices=("same", "separate", "unknown"), default="unknown")
    parser.add_argument("--out", default=str(DEFAULT_REPORT_PATH), help="output report path")
    args = parser.parse_args()
    if args.duration <= 0 or args.interval <= 0:
        raise ValueError("--duration and --interval must be positive")

    processes = {name: int(value) for name, value in parse_mapping(args.process, "--process").items()}
    readiness_urls = parse_mapping(args.readiness, "--readiness")
    runtime_logs = {name: Path(value) for name, value in parse_mapping(args.runtime_log, "--runtime-log").items()}
    process_samples: dict[str, list[dict[str, float]]] = defaultdict(list)
    readiness_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    started = time.monotonic()
    while time.monotonic() - started < args.duration:
        for name, pid in processes.items():
            sample = sample_process(pid)
            if sample:
                process_samples[name].append(sample)
        for name, url in readiness_urls.items():
            readiness_samples[name].append(fetch_readiness(url))
        remaining = args.duration - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(min(args.interval, remaining))

    process_metrics = {
        name: {
            "pid": processes[name],
            "sample_count": len(process_samples[name]),
            "cpu_percent": summarize_values([sample["cpu_percent"] for sample in process_samples[name]]),
            "peak_rss_mb": max((sample["rss_mb"] for sample in process_samples[name]), default=None),
        }
        for name in processes
    }
    readiness_metrics: dict[str, Any] = {}
    for name, samples in readiness_samples.items():
        queue_sizes: list[float] = []
        queue_ratios: list[float] = []
        for sample in samples:
            timetable = sample.get("payload", {}).get("components", {}).get("timetable", {})
            size = timetable.get("queue_size")
            maximum = timetable.get("queue_max_size")
            if isinstance(size, (int, float)):
                queue_sizes.append(float(size))
            if isinstance(size, (int, float)) and isinstance(maximum, (int, float)) and maximum > 0:
                queue_ratios.append(float(size) / float(maximum))
        readiness_metrics[name] = {
            "url": readiness_urls[name],
            "sample_count": len(samples),
            "success_rate": round(sum(bool(sample.get("ok")) for sample in samples) / len(samples), 4) if samples else None,
            "queue_size": summarize_values(queue_sizes),
            "max_queue_utilization": round(max(queue_ratios), 4) if queue_ratios else None,
            "errors": dict(Counter(sample.get("error", "none") for sample in samples if not sample.get("ok"))),
        }

    log_metrics = {}
    for name, path in runtime_logs.items():
        log_metrics[name] = parse_runtime_log(path) if path.exists() else {"path": str(path), "error": "not_found"}

    report = {
        "schema_version": 1,
        "suite": "oci_ai_runtime_baseline",
        "service": "all",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "measurement_window_seconds": round(time.monotonic() - started, 2),
        "host": {
            "architecture": platform.machine(),
            "cpu_model": cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "oci_ocpu_count": args.ocpus,
            "ram_mb": total_ram_mb(),
            "platform": platform.platform(),
        },
        "topology": {
            "main_api_and_chatbot": args.main_chatbot_topology,
            "ollama_and_chatbot": args.ollama_topology,
        },
        "processes": process_metrics,
        "readiness": readiness_metrics,
        "logs": log_metrics,
        "notes": [
            "Use a log file limited to the same measurement window; the collector does not infer a timestamp cutoff.",
            "OCI OCPU count is explicit because logical CPU count does not reliably identify the purchased shape.",
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] OCI AI runtime baseline")
    print(json.dumps({"host": report["host"], "topology": report["topology"]}, ensure_ascii=False))
    print(f"report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
