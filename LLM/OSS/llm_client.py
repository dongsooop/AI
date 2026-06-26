import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional

from openai import OpenAI

from core.exceptions import ConfigurationError
from core.logging import (
    RuntimeComponent,
    RuntimeOperation,
    RuntimeStatus,
    get_logger,
    runtime_log_message,
)
from core.settings import get_settings


settings = get_settings()
logger = get_logger(__name__)

_oss_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chatbot_oss")
_client: Optional[OpenAI] = None
_client_lock = threading.Lock()


def _get_oss_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    with _client_lock:
        if _client is None:
            if not settings.oss_api_key or not settings.oss_model:
                raise ConfigurationError("OSS_API_KEY and OSS_MODEL are required")
            client_kwargs = {"api_key": settings.oss_api_key}
            if settings.oss_base_url:
                client_kwargs["base_url"] = settings.oss_base_url
            _client = OpenAI(**client_kwargs)
            logger.info("oss_client_initialized base_url=%s", bool(settings.oss_base_url))
    return _client


def call_oss(messages: list[dict[str, str]], **kwargs) -> str:
    if not any(message.get("role") == "system" for message in messages):
        messages = [{
            "role": "system",
            "content": (
                "Reasoning: low\n한국어로 단 한 문장으로만 답하라.\n"
                "연락처/전화/번호/문의 요청이 없는 한 전화번호나 이메일, URL을 임의로 만들지 말고 포함하지 마라.\n"
                "모르면 모른다고 답하라."
            ),
        }] + messages

    try:
        client = _get_oss_client()
    except ConfigurationError:
        raise

    start = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=settings.oss_model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.3),
            max_tokens=kwargs.get("max_tokens", 64),
            timeout=kwargs.get("timeout", 45),
        )
        if not response.choices:
            logger.warning(
                runtime_log_message(
                    "chatbot_llm_runtime",
                    component=RuntimeComponent.CHATBOT,
                    operation=RuntimeOperation.LLM,
                    status=RuntimeStatus.FALLBACK,
                    duration_ms=int((time.monotonic() - start) * 1000),
                    result_count=0,
                    fallback=True,
                    fallback_reason="empty_choices",
                    error_code=None,
                    model=settings.oss_model,
                    empty_response=True,
                )
            )
            logger.warning("oss_call_empty_choices")
            return ""
        output = (response.choices[0].message.content or "").strip()
        logger.info(
            runtime_log_message(
                "chatbot_llm_runtime",
                component=RuntimeComponent.CHATBOT,
                operation=RuntimeOperation.LLM,
                status=RuntimeStatus.SUCCESS if output else RuntimeStatus.FALLBACK,
                duration_ms=int((time.monotonic() - start) * 1000),
                result_count=1 if output else 0,
                fallback=not bool(output),
                fallback_reason=None if output else "empty_response",
                error_code=None,
                model=settings.oss_model,
                empty_response=not bool(output),
            )
        )
        return output
    except Exception:
        logger.warning(
            runtime_log_message(
                "chatbot_llm_runtime",
                component=RuntimeComponent.CHATBOT,
                operation=RuntimeOperation.LLM,
                status=RuntimeStatus.FAILED,
                duration_ms=int((time.monotonic() - start) * 1000),
                result_count=0,
                fallback=True,
                fallback_reason="llm_exception",
                error_code="oss_call_failed",
                model=settings.oss_model,
                empty_response=True,
            )
        )
        logger.warning("oss_call_failed", exc_info=True)
        return ""


async def call_oss_async(messages: list[dict[str, str]], **kwargs) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_oss_executor, partial(call_oss, messages, **kwargs))
