import datetime as dt
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from cachetools import TTLCache
from openai import OpenAI
from psycopg2 import pool as pg_pool
from pydantic import BaseModel
from sshtunnel import SSHTunnelForwarder

from core.settings import get_settings
from LLM.OSS.formatter import (
    dept_clarification_message,
    render_chatty_schedule,
    scrub_non_contact,
)
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
from LLM.OSS.postprocess import run_postprocess
from LLM.rule_book.graph import run_rule_book
from LLM.sub_model.query_index import build_answer
from LLM.sub_model.schedule_index import schedule_search


settings = get_settings()

_CACHE_RULE_BOOK: TTLCache = TTLCache(maxsize=200, ttl=86400)
_CACHE_GENERAL: TTLCache = TTLCache(maxsize=500, ttl=3600)
_CACHE_SKIP = frozenset({"oss", "greet", "whoami", "relation", "guard"})
_cache_lock = threading.Lock()

_ssh_tunnel: Optional[SSHTunnelForwarder] = None
_db_pool: Optional[pg_pool.ThreadedConnectionPool] = None
_log_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="chatbot_log")

_client: Optional[OpenAI] = None
_client_lock = threading.Lock()


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
    if ssh_host:
        _ssh_tunnel = SSHTunnelForwarder(
            (ssh_host, 22),
            ssh_username=settings.ssh_user,
            ssh_pkey=settings.ssh_key_path,
            remote_bind_address=("localhost", 5433),
        )
        _ssh_tunnel.start()
        db_kwargs.update(host="localhost", port=_ssh_tunnel.local_bind_port)
    else:
        db_kwargs["host"] = "localhost"
    _db_pool = pg_pool.ThreadedConnectionPool(minconn=1, maxconn=5, **db_kwargs)


def shutdown_db_pool() -> None:
    if _db_pool:
        _db_pool.closeall()
    if _ssh_tunnel:
        _ssh_tunnel.stop()


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
        print(f"[chatbot_log ERROR] {exc}")
    finally:
        if conn and _db_pool:
            _db_pool.putconn(conn)


def _get_oss_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    with _client_lock:
        if _client is None:
            if not settings.oss_api_key or not settings.oss_model:
                raise RuntimeError("OSS_API_KEY and OSS_MODEL are required")
            client_kwargs = {"api_key": settings.oss_api_key}
            if settings.oss_base_url:
                client_kwargs["base_url"] = settings.oss_base_url
            _client = OpenAI(**client_kwargs)
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
        response = _get_oss_client().chat.completions.create(
            model=settings.oss_model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.3),
            max_tokens=kwargs.get("max_tokens", 64),
        )
        return (response.choices[0].message.content or "").strip()
    except Exception:
        return ""


def call_submodel(user_text: str) -> str:
    base = ""
    try:
        result = build_answer(user_text, top_k=12)
        base = (result or {}).get("answer", "").strip()
    except Exception:
        base = ""

    if settings.json_only_mode:
        return base

    try:
        schedule = schedule_search(user_text, top_k=8)
    except Exception:
        schedule = ""

    if schedule and base:
        return schedule + "\n" + base
    return schedule or base


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

    if GOVERNANCE_REMOVE_RE.search(user_text) and GOVERNANCE_TARGET_RE.search(user_text):
        return {
            "engine": "guard",
            "text": "해당 요청은 도움을 드리기 어려워요; 공식 절차나 문의는 학생자치기구 페이지의 연락처를 이용해 주세요.",
        }

    mode = req.engine or decide_mode(user_text)
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

    if mode == "fast":
        schedule_only = schedule_search(user_text, top_k=8)
        if schedule_only:
            return cache_and_return({"engine": "fast", "text": render_chatty_schedule(schedule_only, user_text)})

        clarification = dept_clarification_message(user_text)
        if clarification:
            return cache_and_return({"engine": "fast", "text": clarification})

        sub_answer = call_submodel(user_text)
        text, url = run_postprocess("fast", user_text, sub_answer)
        response = {"engine": "fast", "text": text}
        if url:
            response["url"] = url
        return cache_and_return(response)

    if mode == "policy":
        sub_answer = call_submodel(user_text)
        text, url = run_postprocess("policy", user_text, sub_answer)
        response = {"engine": "policy", "text": text}
        if url:
            response["url"] = url
        return cache_and_return(response)

    if mode == "dorm":
        sub_answer = call_submodel(user_text)
        text, url = run_postprocess("dorm", user_text, sub_answer)
        response = {"engine": "dorm", "text": text}
        if url:
            response["url"] = url
        return cache_and_return(response)

    if mode == "grad":
        sub_answer = call_submodel(user_text)
        text, url = run_postprocess("grad", user_text, sub_answer)
        response = {"engine": "grad", "text": text}
        if url:
            response["url"] = url
        return cache_and_return(response)

    if mode == "topic":
        sub_answer = call_submodel(user_text)
        text, url = run_postprocess("topic", user_text, sub_answer)
        response = {"engine": "topic", "text": text}
        if url:
            response["url"] = url
        return cache_and_return(response)

    if mode == "oss":
        output = call_oss(messages_for_oss)
        if not any(keyword in user_text for keyword in ("연락처", "전화", "번호", "문의")) and not any(keyword in user_text for keyword in COUNCIL_KWS):
            output = scrub_non_contact(output)

        if not output:
            if looks_like_schedule(user_text):
                schedule_only = schedule_search(user_text, top_k=8)
                if schedule_only:
                    response = {"engine": "fast", "text": render_chatty_schedule(schedule_only, user_text)}
                    latency = int((time.monotonic() - start) * 1000)
                    _log_executor.submit(_log_chatbot, user_text, "fast", response["text"], None, False, latency)
                    return response

            if len(user_text) <= 2 or GREETING_RE.search(user_text):
                output = "안녕하세요, 무엇을 도와드릴까요?"
            else:
                sub_answer = call_submodel(user_text)
                if looks_like_schedule(user_text) and sub_answer:
                    response = {"engine": "fast", "text": render_chatty_schedule(sub_answer, user_text)}
                    latency = int((time.monotonic() - start) * 1000)
                    _log_executor.submit(_log_chatbot, user_text, "fast", response["text"], None, False, latency)
                    return response
                if sub_answer:
                    if looks_like_topic(user_text):
                        text, _ = run_postprocess("topic", user_text, sub_answer)
                    else:
                        text, _ = run_postprocess("fast", user_text, sub_answer)
                    output = text
                else:
                    output = "잘 이해하지 못했어요. 다시 질문해주세요."

        latency = int((time.monotonic() - start) * 1000)
        _log_executor.submit(_log_chatbot, user_text, "oss", output, None, False, latency)
        return {"engine": "oss", "text": output}

    sub_answer = call_submodel(user_text)
    fused = call_oss(
        [
            {
                "role": "system",
                "content": "Reasoning: low\n다음 <context>의 사실만 사용해 한국어로 한 문장으로만 답하라. 불릿/개행 금지. 임의의 전화번호/URL을 생성하지 말라.",
            },
            {"role": "user", "content": user_text + "\n\n<context>\n" + (sub_answer or "") + "\n</context>"},
        ],
        max_tokens=64,
        temperature=0.2,
    )
    if not fused:
        text, _ = run_postprocess("topic", user_text, sub_answer)
        fused = text if looks_like_topic(user_text) else "좋아요, 무엇을 이야기해 볼까요?"

    latency = int((time.monotonic() - start) * 1000)
    _log_executor.submit(_log_chatbot, user_text, mode, fused, None, False, latency)
    return {"engine": mode, "text": fused}
