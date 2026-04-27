import asyncio
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
from core.logging import get_logger
from core.settings import get_settings
from image_analysis.ocr_engine import extract_schedule_fixed_scaled


logger = get_logger(__name__)
settings = get_settings()
SPRING_TIMETABLE_URL = settings.spring_timetable_url
JOB_TTL = timedelta(minutes=10)
_WORKER_CONCURRENCY = 2
_PROCESS_CONTEXT = mp.get_context("spawn")
_PROCESS_EXECUTOR = ProcessPoolExecutor(
    max_workers=_WORKER_CONCURRENCY,
    mp_context=_PROCESS_CONTEXT,
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
        response.raise_for_status()


async def _queue_worker() -> None:
    loop = asyncio.get_running_loop()
    while True:
        job_id, user_id, img, token, appcheck_token = await _job_queue.get()
        job = _job_store.get(job_id)
        if job is None:
            _active_users.discard(user_id)
            _job_queue.task_done()
            continue

        job["status"] = JobStatus.RUNNING
        try:
            result = await loop.run_in_executor(_PROCESS_EXECUTOR, extract_schedule_fixed_scaled, img)
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
            logger.exception("timetable_job_failed job_id=%s", job_id, exc_info=exc)
        finally:
            _active_users.discard(user_id)
            _job_store.pop(job_id, None)
            _job_queue.task_done()
            _cleanup_old_jobs()


async def start_queue_workers() -> None:
    _require_spring_url()
    for _ in range(_WORKER_CONCURRENCY):
        task = asyncio.create_task(_queue_worker())
        _WORKER_TASKS.add(task)
        task.add_done_callback(_WORKER_TASKS.discard)
    logger.info("timetable_workers_started concurrency=%d", _WORKER_CONCURRENCY)


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
