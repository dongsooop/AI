import datetime as dt
import hashlib
import threading
import time
from typing import Optional

import httpx
from cachetools import TTLCache
from pydantic import BaseModel

from core.logging import (
    RuntimeComponent,
    RuntimeOperation,
    RuntimeStatus,
    get_logger,
    runtime_log_message,
)
from core.settings import get_settings
from LLM.OSS.chat_log_store import log_chatbot
from LLM.OSS.formatter import (
    scrub_non_contact,
)
from LLM.OSS.llm_client import call_oss_async
from LLM.OSS.modes import (
    COUNCIL_KWS,
    GOVERNANCE_REMOVE_RE,
    GOVERNANCE_TARGET_RE,
    GREETING_RE,
    RELATIVE_DATE_KEYWORDS,
    decide_mode,
    looks_like_schedule,
    looks_like_topic,
)
from LLM.OSS.tools import (
    ToolResult,
    run_empty_oss_fallback_tools,
    run_final_fallback_tools,
    run_mode_tools,
    run_oss_fast_path_tools,
)
from LLM.rule_book.graph import run_rule_book


settings = get_settings()
logger = get_logger(__name__)

_CACHE_RULE_BOOK: TTLCache = TTLCache(maxsize=200, ttl=86400)
_CACHE_GENERAL: TTLCache = TTLCache(maxsize=500, ttl=3600)
_CACHE_SKIP = frozenset({"oss", "greet", "whoami", "relation", "guard"})
_cache_lock = threading.Lock()

PROFANITY_GUARD_TEXT = "부적절한 표현이 포함되어 답변하기 어려워요. 질문을 순화해서 다시 입력해 주세요."


class ChatReq(BaseModel):
    text: Optional[str] = None
    messages: Optional[list[dict[str, str]]] = None
    engine: Optional[str] = None


def _query_hash(user_text: str) -> str:
    return hashlib.sha256((user_text or "").encode("utf-8")).hexdigest()[:12]


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _is_direct_answer_route(result: ToolResult | None) -> bool:
    return bool(result and result.name in {"metadata_direct_answer", "confident_search_answer"})


def _log_tool_route(user_text: str, mode: str, stage: str, result: ToolResult) -> None:
    logger.info(
        "chatbot_tool_route query_hash=%s mode=%s stage=%s tool=%s engine=%s confidence=%.2f "
        "llm_required=%s has_text=%s reason=%s",
        _query_hash(user_text),
        mode,
        stage,
        result.name,
        result.engine,
        result.confidence,
        result.llm_required,
        bool(result.text.strip()),
        result.reason,
    )


def _log_chatbot_summary(
    user_text: str,
    start: float,
    mode: str,
    response: dict,
    *,
    cache_hit: bool = False,
    direct_answer_route: bool = False,
    fallback: bool = False,
    fallback_reason: str | None = None,
    status: RuntimeStatus = RuntimeStatus.SUCCESS,
    error_code: str | None = None,
) -> None:
    text = str(response.get("text", "") or "")
    logger.info(
        runtime_log_message(
            "chatbot_request_summary",
            component=RuntimeComponent.CHATBOT,
            operation=RuntimeOperation.REQUEST,
            status=status,
            duration_ms=_elapsed_ms(start),
            result_count=1 if text else 0,
            fallback=fallback,
            fallback_reason=fallback_reason,
            error_code=error_code,
            query_hash=_query_hash(user_text),
            mode=mode,
            engine=response.get("engine", "none"),
            cache_hit=cache_hit,
            direct_answer_route=direct_answer_route,
        )
    )


async def should_block_profanity(user_text: str) -> bool:
    if not settings.chatbot_profanity_filter_enabled:
        return False
    if not (user_text or "").strip():
        return False
    if not settings.text_filter_api_url:
        logger.warning("chatbot_profanity_filter_missing_url fail_closed=true")
        return True

    try:
        async with httpx.AsyncClient(timeout=settings.text_filter_api_timeout) as client:
            response = await client.post(settings.text_filter_api_url, json={"text": user_text})
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        logger.warning("chatbot_profanity_filter_failed fail_closed=true error=%s", exc)
        return True

    if not isinstance(payload, dict):
        logger.warning("chatbot_profanity_filter_invalid_response fail_closed=true")
        return True

    results = payload.get("results", [])
    if not isinstance(results, list):
        logger.warning("chatbot_profanity_filter_invalid_response fail_closed=true")
        return True
    return any(str(label).strip() == "비속어" for label in results)


def extract_user_text(req: ChatReq) -> str:
    if req.text:
        return req.text.strip()
    return next(
        (
            (message.get("content") or "").strip()
            for message in reversed(req.messages or [])
            if message.get("role") == "user" and (message.get("content") or "").strip()
        ),
        "",
    )


def ensure_messages(req: ChatReq, user_text: str) -> list[dict[str, str]]:
    if req.messages and isinstance(req.messages, list) and req.messages:
        return req.messages
    return [{"role": "user", "content": user_text}]


async def chat_with_oss(req: ChatReq) -> dict:
    start = time.monotonic()
    user_text = extract_user_text(req)
    messages_for_oss = ensure_messages(req, user_text)
    compact_user_text = "".join(user_text.split())

    if await should_block_profanity(user_text):
        latency = int((time.monotonic() - start) * 1000)
        log_chatbot(user_text, "guard", PROFANITY_GUARD_TEXT, None, False, latency)
        response = {"engine": "guard", "text": PROFANITY_GUARD_TEXT}
        _log_chatbot_summary(user_text, start, "guard", response)
        return response

    if GOVERNANCE_REMOVE_RE.search(user_text) and GOVERNANCE_TARGET_RE.search(user_text):
        response = {
            "engine": "guard",
            "text": "해당 요청은 도움을 드리기 어려워요; 공식 절차나 문의는 학생자치기구 페이지의 연락처를 이용해 주세요.",
        }
        _log_chatbot_summary(user_text, start, "guard", response)
        return response

    mode = req.engine or decide_mode(user_text)
    if mode == "oss" and len(compact_user_text) <= 2:
        response = {"engine": "greet", "text": "네, 무엇을 도와드릴까요?"}
        _log_chatbot_summary(user_text, start, mode, response)
        return response
    normalized = " ".join(user_text.strip().split())
    relative_date_scope = ""
    if looks_like_schedule(user_text) and any(keyword in user_text for keyword in RELATIVE_DATE_KEYWORDS):
        relative_date_scope = f":{dt.date.today().isoformat()}"
    cache_key = f"{mode}{relative_date_scope}:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"

    if mode not in _CACHE_SKIP:
        cache = _CACHE_RULE_BOOK if mode == "rule_book" else _CACHE_GENERAL
        with _cache_lock:
            cached = cache.get(cache_key)
        if cached is not None:
            latency = int((time.monotonic() - start) * 1000)
            response = dict(cached)
            log_chatbot(user_text, mode, response.get("text", ""), response.get("url"), True, latency)
            _log_chatbot_summary(user_text, start, mode, response, cache_hit=True)
            return response

    def cache_and_return(
        response: dict,
        *,
        direct_answer_route: bool = False,
        fallback: bool = False,
        fallback_reason: str | None = None,
    ) -> dict:
        if mode not in _CACHE_SKIP:
            cache = _CACHE_RULE_BOOK if mode == "rule_book" else _CACHE_GENERAL
            with _cache_lock:
                cache[cache_key] = response
        latency = int((time.monotonic() - start) * 1000)
        log_chatbot(user_text, mode, response.get("text", ""), response.get("url"), False, latency)
        _log_chatbot_summary(
            user_text,
            start,
            mode,
            response,
            direct_answer_route=direct_answer_route,
            fallback=fallback,
            fallback_reason=fallback_reason,
        )
        return response

    if mode == "rule_book":
        answer = await run_rule_book(user_text)
        return cache_and_return({"engine": "rule_book", "text": answer})
    if mode == "greet":
        response = {"engine": "greet", "text": "안녕하세요, 무엇을 도와드릴까요?"}
        _log_chatbot_summary(user_text, start, mode, response)
        return response
    if mode == "whoami":
        response = {"engine": "whoami", "text": f"{settings.service_name}의 {settings.bot_name}입니다."}
        _log_chatbot_summary(user_text, start, mode, response)
        return response
    if mode == "relation":
        response = {
            "engine": "relation",
            "text": f"우리는 {settings.org_name} 정보를 함께 해결하는 대화 파트너이고, 저는 {settings.service_name}의 {settings.bot_name}입니다.",
        }
        _log_chatbot_summary(user_text, start, mode, response)
        return response

    tool_result = run_mode_tools(mode, user_text)
    _log_tool_route(user_text, mode, "mode_tools", tool_result)
    if tool_result.resolved:
        return cache_and_return(tool_result.to_response(), direct_answer_route=_is_direct_answer_route(tool_result))

    grounded_fallback: ToolResult | None = None
    if mode == "oss":
        fast_path = run_oss_fast_path_tools(user_text)
        _log_tool_route(user_text, mode, "oss_fast_path", fast_path)
        if fast_path.resolved:
            return cache_and_return(fast_path.to_response(), direct_answer_route=_is_direct_answer_route(fast_path))

        grounded_fallback = run_final_fallback_tools(mode, user_text)
        _log_tool_route(user_text, mode, "oss_grounded_fallback", grounded_fallback)
        if grounded_fallback.resolved:
            return cache_and_return(
                grounded_fallback.to_response(),
                direct_answer_route=_is_direct_answer_route(grounded_fallback),
                fallback=True,
                fallback_reason=grounded_fallback.reason,
            )

        if grounded_fallback.llm_required and grounded_fallback.text.strip():
            output = await call_oss_async(
                [
                    {
                        "role": "system",
                        "content": "Reasoning: low\n다음 <context>의 사실만 사용해 한국어로 한 문장으로만 답하라. 불릿/개행 금지. 임의의 전화번호/URL을 생성하지 말라.",
                    },
                    {"role": "user", "content": user_text + "\n\n<context>\n" + grounded_fallback.text + "\n</context>"},
                ],
                max_tokens=96,
                temperature=0.2,
                timeout=45,
            )
            if not any(keyword in user_text for keyword in ("연락처", "전화", "번호", "문의")) and not any(keyword in user_text for keyword in COUNCIL_KWS):
                output = scrub_non_contact(output)
            if output:
                latency = int((time.monotonic() - start) * 1000)
                log_chatbot(user_text, "oss", output, None, False, latency)
                response = {"engine": "oss", "text": output}
                _log_chatbot_summary(
                    user_text,
                    start,
                    mode,
                    response,
                    fallback=True,
                    fallback_reason=grounded_fallback.reason,
                )
                return response

            fallback = run_empty_oss_fallback_tools(user_text)
            _log_tool_route(user_text, mode, "oss_grounded_empty_fallback", fallback)
            if fallback.handled:
                response = fallback.to_response()
                latency = int((time.monotonic() - start) * 1000)
                log_chatbot(user_text, response["engine"], response["text"], response.get("url"), False, latency)
                _log_chatbot_summary(
                    user_text,
                    start,
                    mode,
                    response,
                    direct_answer_route=_is_direct_answer_route(fallback),
                    fallback=True,
                    fallback_reason=fallback.reason,
                )
                return response

            if len(user_text) <= 2 or GREETING_RE.search(user_text):
                latency = int((time.monotonic() - start) * 1000)
                output = "안녕하세요, 무엇을 도와드릴까요?"
                log_chatbot(user_text, "greet", output, None, False, latency)
                response = {"engine": "greet", "text": output}
                _log_chatbot_summary(
                    user_text,
                    start,
                    mode,
                    response,
                    fallback=True,
                    fallback_reason="empty_llm_greeting",
                )
                return response

            return cache_and_return({
                "engine": "oss",
                "text": "관련 근거를 충분히 확인하지 못했어요. 질문을 조금 더 구체적으로 다시 입력해 주세요.",
            }, fallback=True, fallback_reason="grounded_llm_empty")

        output = await call_oss_async(messages_for_oss)
        if not any(keyword in user_text for keyword in ("연락처", "전화", "번호", "문의")) and not any(keyword in user_text for keyword in COUNCIL_KWS):
            output = scrub_non_contact(output)

        if not output:
            fallback = run_empty_oss_fallback_tools(user_text)
            _log_tool_route(user_text, mode, "oss_empty_fallback", fallback)
            if fallback.handled:
                response = fallback.to_response()
                latency = int((time.monotonic() - start) * 1000)
                log_chatbot(user_text, response["engine"], response["text"], response.get("url"), False, latency)
                _log_chatbot_summary(
                    user_text,
                    start,
                    mode,
                    response,
                    direct_answer_route=_is_direct_answer_route(fallback),
                    fallback=True,
                    fallback_reason=fallback.reason,
                )
                return response

            if len(user_text) <= 2 or GREETING_RE.search(user_text):
                output = "안녕하세요, 무엇을 도와드릴까요?"

        if output:
            latency = int((time.monotonic() - start) * 1000)
            log_chatbot(user_text, "oss", output, None, False, latency)
            response = {"engine": "oss", "text": output}
            _log_chatbot_summary(
                user_text,
                start,
                mode,
                response,
                fallback=output == "안녕하세요, 무엇을 도와드릴까요?",
                fallback_reason="empty_llm_greeting" if output == "안녕하세요, 무엇을 도와드릴까요?" else None,
            )
            return response
        
        if grounded_fallback is not None and not grounded_fallback.text.strip():
            return cache_and_return({
                "engine": "oss",
                "text": "관련 근거를 충분히 확인하지 못했어요. 질문을 조금 더 구체적으로 다시 입력해주세요."
            }, fallback=True, fallback_reason="empty_rag_context")

    fallback = grounded_fallback or run_final_fallback_tools(mode, user_text)
    _log_tool_route(user_text, mode, "final_fallback", fallback)
    if fallback.resolved:
        return cache_and_return(
            fallback.to_response(),
            direct_answer_route=_is_direct_answer_route(fallback),
            fallback=True,
            fallback_reason=fallback.reason,
        )

    fused = await call_oss_async(
        [
            {
                "role": "system",
                "content": "Reasoning: low\n다음 <context>의 사실만 사용해 한국어로 한 문장으로만 답하라. 불릿/개행 금지. 임의의 전화번호/URL을 생성하지 말라.",
            },
            {"role": "user", "content": user_text + "\n\n<context>\n" + (fallback.text or "") + "\n</context>"},
        ],
        max_tokens=96,
        temperature=0.2,
        timeout=45,
    )
    if not fused:
        text = run_mode_tools("topic", user_text).text
        fused = text if looks_like_topic(user_text) else "좋아요, 무엇을 이야기해 볼까요?"

    latency = int((time.monotonic() - start) * 1000)
    log_chatbot(user_text, mode, fused, None, False, latency)
    response = {"engine": mode, "text": fused}
    _log_chatbot_summary(
        user_text,
        start,
        mode,
        response,
        fallback=True,
        fallback_reason=fallback.reason if fallback else None,
    )
    return response
