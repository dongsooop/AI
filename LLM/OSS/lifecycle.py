import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from core.exceptions import ConfigurationError
from core.logging import get_logger
from core.settings import get_settings
from LLM.OSS.chat_log_store import init_db_pool, shutdown_db_pool
from LLM.rule_book.index import build_index


settings = get_settings()
logger = get_logger(__name__)


async def startup_chatbot_runtime() -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, build_index)
    try:
        await loop.run_in_executor(None, init_db_pool)
    except ConfigurationError:
        if settings.chatbot_log_db_required:
            raise
        logger.warning("chatbot_log_db_unavailable startup_continues=true", exc_info=True)


def shutdown_chatbot_runtime() -> None:
    shutdown_db_pool()


@asynccontextmanager
async def chatbot_lifespan(app: FastAPI):
    await startup_chatbot_runtime()
    yield
    shutdown_chatbot_runtime()
