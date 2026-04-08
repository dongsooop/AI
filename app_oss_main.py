from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI
from LLM.OSS.Open_AI_OSS import router as Open_AI_OSS
from LLM.rule_book.index import build_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.get_event_loop().run_in_executor(None, build_index)
    yield


app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status":"ok"}

app.include_router(Open_AI_OSS)