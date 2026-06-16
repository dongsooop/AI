import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from core.exceptions import ConfigurationError
from core.logging import get_logger
from core.settings import get_settings
from LLM.OSS.chat_log_store import init_db_pool, is_db_pool_initialized, shutdown_db_pool
from LLM.rule_book.index import build_index, get_index


settings = get_settings()
logger = get_logger(__name__)
_startup_complete = False
_startup_error_code: str | None = None
_log_db_error_code: str | None = None


async def startup_chatbot_runtime() -> None:
    global _startup_complete, _startup_error_code, _log_db_error_code
    _startup_complete = False
    _startup_error_code = None
    _log_db_error_code = None
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, build_index)
    except Exception:
        _startup_error_code = "rule_book_unavailable"
        logger.exception("chatbot_rule_book_startup_failed")
        raise
    try:
        await loop.run_in_executor(None, init_db_pool)
    except ConfigurationError:
        _log_db_error_code = "chat_log_db_unavailable"
        if settings.chatbot_log_db_required:
            _startup_error_code = "chat_log_db_unavailable"
            raise
        logger.warning("chatbot_log_db_unavailable startup_continues=true", exc_info=True)
    _startup_complete = True


def shutdown_chatbot_runtime() -> None:
    global _startup_complete, _startup_error_code, _log_db_error_code
    shutdown_db_pool()
    _startup_complete = False
    _startup_error_code = None
    _log_db_error_code = None


def get_chatbot_runtime_readiness() -> dict[str, object]:
    components: dict[str, dict[str, object]] = {}

    rule_index = get_index()
    rule_book_chunks = getattr(rule_index, "chunks", None) or []
    rule_book_ready = bool(getattr(rule_index, "_built", False)) and bool(rule_book_chunks)
    components["rule_book"] = {
        "status": "ready" if rule_book_ready else "not_ready",
        "required": True,
        "chunks": len(rule_book_chunks),
    }

    try:
        # Keep this lazy import here: LLM.OSS.tools imports query_index during normal
        # request routing, while readiness must tolerate import/runtime failures.
        from LLM.sub_model import query_index

        search_rows = len(query_index.search_df)
        embedding_rows = int(query_index.embeddings.shape[0])
        query_index_ready = search_rows > 0 and embedding_rows == search_rows
        components["query_index"] = {
            "status": "ready" if query_index_ready else "not_ready",
            "required": True,
            "documents": search_rows,
            "embeddings": embedding_rows,
            "contact_keywords": len(query_index.CONTACT_KWS),
        }
    except Exception:
        query_index_ready = False
        logger.warning("chatbot_query_index_readiness_failed", exc_info=True)
        components["query_index"] = {
            "status": "not_ready",
            "required": True,
            "error_code": "query_index_unavailable",
        }

    log_db_ready = is_db_pool_initialized()
    log_db_required = settings.chatbot_log_db_required
    components["chat_log_db"] = {
        "status": "ready" if log_db_ready else ("not_ready" if log_db_required else "degraded"),
        "required": log_db_required,
        "error_code": _log_db_error_code,
    }

    ready = (
        _startup_complete
        and _startup_error_code is None
        and rule_book_ready
        and query_index_ready
        and (log_db_ready or not log_db_required)
    )
    return {
        "status": "ready" if ready else "not_ready",
        "startup_complete": _startup_complete,
        "startup_error_code": _startup_error_code,
        "components": components,
    }


@asynccontextmanager
async def chatbot_lifespan(app: FastAPI):
    await startup_chatbot_runtime()
    yield
    shutdown_chatbot_runtime()
