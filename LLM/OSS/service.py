import datetime as dt
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import httpx
from cachetools import TTLCache
from psycopg2 import pool as pg_pool
from pydantic import BaseModel
from sshtunnel import SSHTunnelForwarder

from core.exceptions import ConfigurationError
from core.logging import get_logger
from core.settings import get_settings
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

_ssh_tunnel: Optional[SSHTunnelForwarder] = None
_db_pool: Optional[pg_pool.ThreadedConnectionPool] = None
_log_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="chatbot_log")

PROFANITY_GUARD_TEXT = "부적절한 표현이 포함되어 답변하기 어려워요. 질문을 순화해서 다시 입력해 주세요."


class ChatReq(BaseModel):
    text: Optional[str] = None
    messages: Optional[list[dict[str, str]]] = None
    engine: Optional[str] = None


def init_db_pool() -> None:
    global _ssh_tunnel, _db_pool
    ssh_host = settings.ssh_host
    db_kwargs = dict(
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        connect_timeout=3,
        options="-c statement_timeout=5000",
    )
    connection_errors: list[Exception] = []
    if ssh_host:
        try:
            if not settings.ssh_user:
                raise ConfigurationError("SSH_USER is required when SSH_HOST is set")
            if not settings.ssh_key_path:
                raise ConfigurationError("SSH_KEY_PATH is required when SSH_HOST is set")
            ssh_key_path = Path(settings.ssh_key_path).expanduser()
            if not ssh_key_path.exists():
                raise ConfigurationError(f"SSH key file does not exist: {ssh_key_path}")
            _ssh_tunnel = SSHTunnelForwarder(
                (ssh_host, 22),
                ssh_username=settings.ssh_user,
                ssh_pkey=str(ssh_key_path),
                remote_bind_address=(settings.ssh_db_host, settings.ssh_db_port),
            )
            _ssh_tunnel.start()
            tunnel_db_kwargs = dict(db_kwargs, host="localhost", port=_ssh_tunnel.local_bind_port)
            _db_pool = pg_pool.ThreadedConnectionPool(minconn=1, maxconn=5, **tunnel_db_kwargs)
            logger.info(
                "chatbot_db_pool_initialized ssh_tunnel=%s db_host=%s db_port=%s",
                True,
                tunnel_db_kwargs["host"],
                tunnel_db_kwargs["port"],
            )
            return
        except ConfigurationError as exc:
            connection_errors.append(exc)
            if settings.db_host is None:
                raise
            logger.warning("chatbot_ssh_db_config_invalid fallback_to_direct=true error=%s", exc)
        except Exception as exc:
            connection_errors.append(exc)
            if _ssh_tunnel:
                _ssh_tunnel.stop()
                _ssh_tunnel = None
            if settings.db_host is None:
                raise ConfigurationError(f"SSH database connection failed: {exc}") from exc
            logger.warning("chatbot_ssh_db_connect_failed fallback_to_direct=true error=%s", exc)

    hosts = (settings.db_host, ) if settings.db_host else ("localhost",)
    for host in hosts:
        if not host:
            continue
        direct_db_kwargs = dict(db_kwargs, host=host, port=settings.db_port)
        try:
            _db_pool = pg_pool.ThreadedConnectionPool(minconn=1, maxconn=5, **direct_db_kwargs)
            logger.info(
                "chatbot_db_pool_initialized ssh_tunnel=%s db_host=%s db_port=%s",
                False,
                direct_db_kwargs["host"],
                direct_db_kwargs["port"],
            )
            return
        except Exception as exc:
            connection_errors.append(exc)
            logger.warning(
                "chatbot_direct_db_connect_failed db_host=%s db_port=%s error=%s",
                host,
                settings.db_port,
                exc,
            )

    raise ConfigurationError(f"Database connection failed: {connection_errors[-1]}")


def shutdown_db_pool() -> None:
    if _db_pool:
        _db_pool.closeall()
    if _ssh_tunnel:
        _ssh_tunnel.stop()
    logger.info("chatbot_db_pool_shutdown")


def _log_chatbot(query: str, mode: str, response: str, url: Optional[str], cache_hit: bool, latency_ms: int) -> None:
    if _db_pool is None:
        return
    conn = None
    try:
        conn = _db_pool.getconn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chatbot_logs (query, mode, response, url, cache_hit, latency_ms)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (query, mode, response, url, cache_hit, latency_ms),
                )
    except Exception as exc:
        logger.warning("chatbot_log_write_failed: %s", exc, exc_info=True)
    finally:
        if conn and _db_pool:
            _db_pool.putconn(conn)


def _log_tool_route(user_text: str, mode: str, stage: str, result: ToolResult) -> None:
    query_hash = hashlib.sha256((user_text or "").encode("utf-8")).hexdigest()[:12]
    logger.info(
        "chatbot_tool_route query_hash=%s mode=%s stage=%s tool=%s engine=%s confidence=%.2f "
        "llm_required=%s has_text=%s reason=%s",
        query_hash,
        mode,
        stage,
        result.name,
        result.engine,
        result.confidence,
        result.llm_required,
        bool(result.text.strip()),
        result.reason,
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
        _log_executor.submit(_log_chatbot, user_text, "guard", PROFANITY_GUARD_TEXT, None, False, latency)
        return {"engine": "guard", "text": PROFANITY_GUARD_TEXT}

    if GOVERNANCE_REMOVE_RE.search(user_text) and GOVERNANCE_TARGET_RE.search(user_text):
        return {
            "engine": "guard",
            "text": "해당 요청은 도움을 드리기 어려워요; 공식 절차나 문의는 학생자치기구 페이지의 연락처를 이용해 주세요.",
        }

    mode = req.engine or decide_mode(user_text)
    if mode == "oss" and len(compact_user_text) <= 2:
        return {"engine": "greet", "text": "네, 무엇을 도와드릴까요?"}
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
            _log_executor.submit(_log_chatbot, user_text, mode, response.get("text", ""), response.get("url"), True, latency)
            return response

    def cache_and_return(response: dict) -> dict:
        if mode not in _CACHE_SKIP:
            cache = _CACHE_RULE_BOOK if mode == "rule_book" else _CACHE_GENERAL
            with _cache_lock:
                cache[cache_key] = response
        latency = int((time.monotonic() - start) * 1000)
        _log_executor.submit(_log_chatbot, user_text, mode, response.get("text", ""), response.get("url"), False, latency)
        return response

    if mode == "rule_book":
        answer = await run_rule_book(user_text)
        return cache_and_return({"engine": "rule_book", "text": answer})
    if mode == "greet":
        return {"engine": "greet", "text": "안녕하세요, 무엇을 도와드릴까요?"}
    if mode == "whoami":
        return {"engine": "whoami", "text": f"{settings.service_name}의 {settings.bot_name}입니다."}
    if mode == "relation":
        return {
            "engine": "relation",
            "text": f"우리는 {settings.org_name} 정보를 함께 해결하는 대화 파트너이고, 저는 {settings.service_name}의 {settings.bot_name}입니다.",
        }

    tool_result = run_mode_tools(mode, user_text)
    _log_tool_route(user_text, mode, "mode_tools", tool_result)
    if tool_result.resolved:
        return cache_and_return(tool_result.to_response())

    grounded_fallback: ToolResult | None = None
    if mode == "oss":
        fast_path = run_oss_fast_path_tools(user_text)
        _log_tool_route(user_text, mode, "oss_fast_path", fast_path)
        if fast_path.resolved:
            return cache_and_return(fast_path.to_response())

        grounded_fallback = run_final_fallback_tools(mode, user_text)
        _log_tool_route(user_text, mode, "oss_grounded_fallback", grounded_fallback)
        if grounded_fallback.resolved:
            return cache_and_return(grounded_fallback.to_response())

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
                _log_executor.submit(_log_chatbot, user_text, "oss", output, None, False, latency)
                return {"engine": "oss", "text": output}

            fallback = run_empty_oss_fallback_tools(user_text)
            _log_tool_route(user_text, mode, "oss_grounded_empty_fallback", fallback)
            if fallback.handled:
                response = fallback.to_response()
                latency = int((time.monotonic() - start) * 1000)
                _log_executor.submit(_log_chatbot, user_text, response["engine"], response["text"], response.get("url"), False, latency)
                return response

            if len(user_text) <= 2 or GREETING_RE.search(user_text):
                latency = int((time.monotonic() - start) * 1000)
                output = "안녕하세요, 무엇을 도와드릴까요?"
                _log_executor.submit(_log_chatbot, user_text, "greet", output, None, False, latency)
                return {"engine": "greet", "text": output}

            return cache_and_return({
                "engine": "oss",
                "text": "관련 근거를 충분히 확인하지 못했어요. 질문을 조금 더 구체적으로 다시 입력해 주세요.",
            })

        output = await call_oss_async(messages_for_oss)
        if not any(keyword in user_text for keyword in ("연락처", "전화", "번호", "문의")) and not any(keyword in user_text for keyword in COUNCIL_KWS):
            output = scrub_non_contact(output)

        if not output:
            fallback = run_empty_oss_fallback_tools(user_text)
            _log_tool_route(user_text, mode, "oss_empty_fallback", fallback)
            if fallback.handled:
                response = fallback.to_response()
                latency = int((time.monotonic() - start) * 1000)
                _log_executor.submit(_log_chatbot, user_text, response["engine"], response["text"], response.get("url"), False, latency)
                return response

            if len(user_text) <= 2 or GREETING_RE.search(user_text):
                output = "안녕하세요, 무엇을 도와드릴까요?"

        if output:
            latency = int((time.monotonic() - start) * 1000)
            _log_executor.submit(_log_chatbot, user_text, "oss", output, None, False, latency)
            return {"engine": "oss", "text": output}
        
        if grounded_fallback is not None and not grounded_fallback.text.strip():
            return cache_and_return({
                "engine": "oss",
                "text": "관련 근거를 충분히 확인하지 못했어요. 질문을 조금 더 구체적으로 다시 입력해주세요."
            })

    fallback = grounded_fallback or run_final_fallback_tools(mode, user_text)
    _log_tool_route(user_text, mode, "final_fallback", fallback)
    if fallback.resolved:
        return cache_and_return(fallback.to_response())

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
    _log_executor.submit(_log_chatbot, user_text, mode, fused, None, False, latency)
    return {"engine": mode, "text": fused}
