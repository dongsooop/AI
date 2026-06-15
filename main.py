from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse

from core.exceptions import register_exception_handlers
from core.logging import configure_logging, register_request_logging

configure_logging("main-api")

from text_filtering.text_filtering import router as text_filter_router
from text_filtering.text_filtering_rule import router as text_filter_rule_router
from image_analysis.timetable_analysis import router as timetable
from image_analysis.service import get_timetable_readiness, start_queue_workers
from text_filtering.service import get_text_filter_readiness


@asynccontextmanager
async def lifespan(app: FastAPI):
    await start_queue_workers()
    yield


app = FastAPI(lifespan=lifespan)

MAX_FILE_SIZE = 3 * 1024 * 1024

@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    return await call_next(request)


register_request_logging(app)
register_exception_handlers(app)

# 헬스체크
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    components = {
        "text_filter": get_text_filter_readiness(),
        "timetable": get_timetable_readiness(),
    }
    is_ready = all(component["status"] == "ready" for component in components.values())
    payload = {
        "status": "ready" if is_ready else "not_ready",
        "components": components,
    }
    status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=status_code, content=payload)

# 라우터 등록
app.include_router(text_filter_router)
app.include_router(text_filter_rule_router)
app.include_router(timetable)
