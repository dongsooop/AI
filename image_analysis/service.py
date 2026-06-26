import asyncio
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import multiprocessing as mp

import cv2
import httpx
import numpy as np
from fastapi import Request, UploadFile
from fastapi.responses import JSONResponse

from core.exceptions import (
    BadRequestError,
    ConfigurationError,
    GatewayTimeoutError,
    ServiceUnavailableError,
    UnauthorizedError,
)
from core.logging import (
    RuntimeComponent,
    RuntimeOperation,
    RuntimeStatus,
    get_logger,
    runtime_log_message,
)
from core.settings import get_settings
from image_analysis.ocr_engine import configure_ocr_worker_logging, extract_schedule_runtime_report


logger = get_logger(__name__)
settings = get_settings()
SPRING_TIMETABLE_URL = settings.spring_timetable_url
JOB_TTL = timedelta(minutes=10)
_WORKER_CONCURRENCY = 2
_PROCESS_CONTEXT = mp.get_context("spawn")
_PROCESS_EXECUTOR = ProcessPoolExecutor(
    max_workers=_WORKER_CONCURRENCY,
    mp_context=_PROCESS_CONTEXT,
    initializer=configure_ocr_worker_logging,
)
_WORKER_TASKS: set[asyncio.Task] = set()


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


_job_store: Dict[str, Dict[str, Any]] = {}
_active_users: set[str] = set()
_job_queue: asyncio.Queue = asyncio.Queue(maxsize=20)


def _require_spring_url() -> str:
    if not SPRING_TIMETABLE_URL:
        raise ConfigurationError("SPRING_TIMETABLE_URL is required")
    return SPRING_TIMETABLE_URL


def _cleanup_old_jobs() -> None:
    now = datetime.now(timezone.utc)
    expired = [
        job_id
        for job_id, job in _job_store.items()
        if now - job.get("created_at", now) > JOB_TTL
    ]
    for job_id in expired:
        _job_store.pop(job_id, None)


async def _post_to_spring(schedules: List[dict], token: str, appcheck_token: str = "") -> None:
    spring_url = _require_spring_url()
    now = datetime.now(timezone.utc)
    semester = "FIRST" if 3 <= now.month <= 8 else "SECOND"
    payload = [
        {
            "name": item["name"],
            "professor": item.get("professor", ""),
            "location": item.get("location", ""),
            "week": item["week"],
            "startAt": item["startAt"],
            "endAt": item["endAt"],
            "year": now.year,
            "semester": semester,
        }
        for item in schedules
    ]

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Firebase-AppCheck": appcheck_token,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(spring_url, json=payload, headers=headers)
        if response.is_error:
            response_body = response.text.replace("\n", " ")[:500]
            logger.error(
                "spring_timetable_post_failed status_code=%d url=%s response_body=%s",
                response.status_code,
                spring_url,
                response_body,
            )
        response.raise_for_status()


async def _queue_worker() -> None:
    loop = asyncio.get_running_loop()
    while True:
        job_id, user_id, img, token, appcheck_token = await _job_queue.get()
        start = time.monotonic()
        job = _job_store.get(job_id)
        if job is None:
            _active_users.discard(user_id)
            _job_queue.task_done()
            continue

        job["status"] = JobStatus.RUNNING
        try:
            report = await loop.run_in_executor(_PROCESS_EXECUTOR, extract_schedule_runtime_report, img)
            result = report.get("schedules", [])
            diagnostics = report.get("diagnostics", {})
            _log_timetable_runtime(job_id, start, diagnostics, len(result))
            if result:
                await _post_to_spring(result, token, appcheck_token)
                job["status"] = JobStatus.DONE
                logger.info("timetable_job_done job_id=%s result_count=%d", job_id, len(result))
            else:
                job["status"] = JobStatus.DONE
                logger.info("timetable_job_done job_id=%s result_count=0 spring_post=skipped", job_id)
        except Exception as exc:
            job["status"] = JobStatus.ERROR
            job["error"] = str(exc)
            logger.error(
                runtime_log_message(
                    "timetable_job_runtime",
                    component=RuntimeComponent.OCR,
                    operation=RuntimeOperation.REQUEST,
                    status=RuntimeStatus.FAILED,
                    duration_ms=int((time.monotonic() - start) * 1000),
                    result_count=0,
                    fallback=True,
                    fallback_reason="worker_exception",
                    error_code=type(exc).__name__,
                    job_id=job_id,
                )
            )
            logger.exception("timetable_job_failed job_id=%s", job_id, exc_info=exc)
        finally:
            _active_users.discard(user_id)
            _job_store.pop(job_id, None)
            _job_queue.task_done()
            _cleanup_old_jobs()


def _log_timetable_runtime(job_id: str, start: float, diagnostics: dict[str, Any], result_count: int) -> None:
    ocr = diagnostics.get("ocr", {})
    runtime = diagnostics.get("runtime", {})
    failure_reason = diagnostics.get("failure_reason")
    fallback_cell_count = int(ocr.get("fallback_cells", 0) or 0)
    logger.info(
        runtime_log_message(
            "timetable_job_runtime",
            component=RuntimeComponent.OCR,
            operation=RuntimeOperation.REQUEST,
            status=RuntimeStatus.SUCCESS if not failure_reason else RuntimeStatus.FALLBACK,
            duration_ms=int((time.monotonic() - start) * 1000),
            result_count=result_count,
            fallback=bool(failure_reason) or fallback_cell_count > 0,
            fallback_reason=failure_reason,
            error_code=None,
            job_id=job_id,
            grid_detection_duration_ms=runtime.get("grid_detection_duration_ms", 0),
            ocr_duration_ms=runtime.get("ocr_duration_ms", 0),
            engine_total_duration_ms=runtime.get("total_duration_ms", 0),
            extracted_cell_count=ocr.get("accepted_cells", 0),
            total_cell_count=ocr.get("total_cells", 0),
            text_cell_count=ocr.get("text_cells", 0),
            ocr_fallback_cell_count=fallback_cell_count,
        )
    )


async def start_queue_workers() -> None:
    _require_spring_url()
    for _ in range(_WORKER_CONCURRENCY):
        task = asyncio.create_task(_queue_worker())
        _WORKER_TASKS.add(task)
        task.add_done_callback(_WORKER_TASKS.discard)
    logger.info("timetable_workers_started concurrency=%d", _WORKER_CONCURRENCY)


def get_timetable_readiness() -> dict[str, object]:
    running_workers = sum(1 for task in _WORKER_TASKS if not task.done())
    ready = bool(SPRING_TIMETABLE_URL) and running_workers >= _WORKER_CONCURRENCY
    return {
        "status": "ready" if ready else "not_ready",
        "required": True,
        "spring_url_configured": bool(SPRING_TIMETABLE_URL),
        "workers_expected": _WORKER_CONCURRENCY,
        "workers_running": running_workers,
        "queue_size": _job_queue.qsize(),
        "queue_max_size": _job_queue.maxsize,
    }


async def enqueue_timetable_analysis(request: Request, file: UploadFile, user_id: str):
    if user_id in _active_users:
        raise ServiceUnavailableError("Already processing", code="timetable_already_processing")
    _active_users.add(user_id)

    try:
        file_bytes = await file.read()
        npimg = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img is None:
            raise BadRequestError("Invalid image format", code="invalid_image_format")

        auth_header = request.headers.get("Authorization", "")
        parts = auth_header.split(" ", 1)
        if len(parts) != 2 or parts[0] != "Bearer" or not parts[1]:
            raise UnauthorizedError("Invalid Authorization header", code="invalid_authorization_header")
        token = parts[1]
        appcheck_token = request.headers.get("X-Firebase-AppCheck", "")
        job_id = str(uuid.uuid4())
        _job_store[job_id] = {
            "status": JobStatus.PENDING,
            "created_at": datetime.now(timezone.utc),
        }

        try:
            await asyncio.wait_for(_job_queue.put((job_id, user_id, img, token, appcheck_token)), timeout=10.0)
        except asyncio.TimeoutError as exc:
            _job_store.pop(job_id, None)
            raise GatewayTimeoutError("Server is busy", code="timetable_queue_timeout") from exc

        logger.info("timetable_job_queued job_id=%s user_id=%s", job_id, user_id)
        return JSONResponse(status_code=202, content={"job_id": job_id})
    except Exception:
        jid = locals().get("job_id")
        if jid:
            _job_store.pop(jid, None)
        _active_users.discard(user_id)
        raise
