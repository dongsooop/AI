from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI

from core.exceptions import register_exception_handlers
from core.logging import configure_logging, register_request_logging

configure_logging("chatbot-api")

from LLM.OSS.Open_AI_OSS import router as Open_AI_OSS, init_db_pool, shutdown_db_pool
from LLM.rule_book.index import build_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.get_event_loop().run_in_executor(None, build_index)
    await asyncio.get_event_loop().run_in_executor(None, init_db_pool)
    yield
    shutdown_db_pool()


app = FastAPI(lifespan=lifespan)
register_request_logging(app)
register_exception_handlers(app)

@app.get("/health")
def health():
    return {"status":"ok"}

app.include_router(Open_AI_OSS)
