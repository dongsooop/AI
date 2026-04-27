from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from core.exceptions.base import AppError
from core.logging.config import get_logger
from core.logging.context import set_request_id


logger = get_logger(__name__)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


def _error_response(
        request: Request, 
        status_code: int, 
        code: str, 
        message: str,
        extra_headers: dict[str, str] | None = None
        ) -> JSONResponse:
    request_id = _request_id(request)
    headers = {"X-Request-ID": request_id}
    if extra_headers:
        headers.update(extra_headers)
    return JSONResponse(
        status_code=status_code,
        content={
            "error": message,
            "code": code,
            "request_id": request_id,
        },
        headers = headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        set_request_id(_request_id(request))
        log_method = logger.error if exc.status_code >= 500 else logger.warning
        log_method(
            "handled_app_error status_code=%d code=%s path=%s message=%s",
            exc.status_code,
            exc.code,
            request.url.path,
            exc.message,
        )
        return _error_response(request, exc.status_code, exc.code, exc.message)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        set_request_id(_request_id(request))
        message = str(exc.detail)
        log_method = logger.error if exc.status_code >= 500 else logger.warning
        log_method(
            "handled_http_exception status_code=%d path=%s message=%s",
            exc.status_code,
            request.url.path,
            message,
        )
        return _error_response(request, exc.status_code, "http_error", message, extra_headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        set_request_id(_request_id(request))
        logger.warning(
            "request_validation_failed path=%s errors=%s",
            request.url.path,
            exc.errors(),
        )
        return _error_response(
            request,
            422,
            "request_validation_error",
            "요청 형식이 올바르지 않습니다.",
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        set_request_id(_request_id(request))
        logger.exception("unhandled_exception path=%s", request.url.path, exc_info=exc)
        return _error_response(
            request,
            500,
            "internal_server_error",
            "서버 내부 오류가 발생했습니다.",
        )
