import time

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.auth import verify_jwt_token
from core.exceptions import BadRequestError
from core.logging import (
    RuntimeComponent,
    RuntimeOperation,
    RuntimeStatus,
    get_logger,
    runtime_log_message,
)
from text_filtering.service import analyze_fields

router = APIRouter()
logger = get_logger(__name__)


class TextRequest(BaseModel):
    text: str


def _log_rule_filter_runtime(
    endpoint: str,
    start: float,
    *,
    status: RuntimeStatus,
    field_count: int,
    sentence_count: int,
    result_count: int,
    has_profanity: bool,
    fallback: bool = False,
    fallback_reason: str | None = None,
    error_code: str | None = None,
) -> None:
    runtime_status = RuntimeStatus.FALLBACK if fallback and status == RuntimeStatus.SUCCESS else status
    logger.info(
        runtime_log_message(
            "text_filter_rule_runtime",
            component=RuntimeComponent.TEXT_FILTERING,
            operation=RuntimeOperation.RULE_FILTER,
            status=runtime_status,
            duration_ms=int((time.monotonic() - start) * 1000),
            result_count=result_count,
            fallback=fallback,
            fallback_reason=fallback_reason,
            error_code=error_code,
            endpoint=endpoint,
            field_count=field_count,
            sentence_count=sentence_count,
            has_profanity=has_profanity,
        )
    )


def _count_result_sentences(*results: dict) -> int:
    return sum(len(result.get("results", [])) for result in results)


@router.post("/text_filter_rule")
async def rule_filter_api(
    payload: TextRequest,
    username: str = Depends(verify_jwt_token)
):
    full_text = payload.text.strip()

    try:
        title, tags, content = [x.strip() for x in full_text.split('|', 2)]
    except ValueError as exc:
        raise BadRequestError(
            "text 형식은 '제목 | 태그 | 본문' 이어야 합니다.",
            code="invalid_text_filter_rule_format",
        ) from exc

    start = time.monotonic()
    try:
        analyzed = analyze_fields([
            ("제목", title, True),
            ("태그", tags, False),
            ("본문", content, True),
        ])
    except Exception as exc:
        _log_rule_filter_runtime(
            "/text_filter_rule",
            start,
            status=RuntimeStatus.FAILED,
            field_count=3,
            sentence_count=0,
            result_count=0,
            has_profanity=False,
            fallback=True,
            fallback_reason="rule_filter_failed",
            error_code=type(exc).__name__,
        )
        raise
    title_result = analyzed["제목"]
    tags_result = analyzed["태그"]
    content_result = analyzed["본문"]

    response = {
        "username": username,
        "제목": title_result,
        "태그": tags_result,
        "본문": content_result
    }

    has_profanity = title_result["has_profanity"] or tags_result["has_profanity"] or content_result["has_profanity"]
    result_count = _count_result_sentences(title_result, tags_result, content_result)
    _log_rule_filter_runtime(
        "/text_filter_rule",
        start,
        status=RuntimeStatus.SUCCESS,
        field_count=3,
        sentence_count=result_count,
        result_count=result_count,
        has_profanity=has_profanity,
    )

    if has_profanity:
        return JSONResponse(content=response, status_code=400)
    return JSONResponse(content=response, status_code=200)


@router.post("/text_filter_content")
async def text_filter_content_api(
    payload: TextRequest):
    text = payload.text.strip()
    start = time.monotonic()
    try:
        analyzed = analyze_fields([
            ("content", text, True),
        ])
    except Exception as exc:
        _log_rule_filter_runtime(
            "/text_filter_content",
            start,
            status=RuntimeStatus.FAILED,
            field_count=1,
            sentence_count=0,
            result_count=0,
            has_profanity=False,
            fallback=True,
            fallback_reason="content_filter_failed",
            error_code=type(exc).__name__,
        )
        raise
    content_result = analyzed["content"]

    response = {
        "content": content_result
    }

    result_count = _count_result_sentences(content_result)
    _log_rule_filter_runtime(
        "/text_filter_content",
        start,
        status=RuntimeStatus.SUCCESS,
        field_count=1,
        sentence_count=result_count,
        result_count=result_count,
        has_profanity=content_result["has_profanity"],
    )

    return JSONResponse(
        status_code=400 if content_result["has_profanity"] else 200,
        content=response
    )
