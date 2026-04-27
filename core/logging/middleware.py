from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request

from core.logging.config import get_logger
from core.logging.context import clear_request_id, set_request_id


logger = get_logger(__name__)


def register_request_logging(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        set_request_id(request_id)
        request.state.request_id = request_id
        start = time.perf_counter()

        logger.info("request_started method=%s path=%s", request.method, request.url.path)
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = int((time.perf_counter() - start) * 1000)
            logger.exception(
                "request_failed method=%s path=%s duration_ms=%d",
                request.method,
                request.url.path,
                duration_ms,
            )
            raise
        else:
            duration_ms = int((time.perf_counter() - start) * 1000)
            response.headers["X-Request-ID"] = request_id
            logger.info(
                "request_completed method=%s path=%s status_code=%d duration_ms=%d",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
            )
            return response
        finally:
            clear_request_id()
