from fastapi import FastAPI, Request, HTTPException
from text_filtering.text_filtering import router as text_filter_router
from text_filtering.text_filtering_rule import router as text_filter_rule_router
from image_analysis.timetable_analysis import router as timetable

app = FastAPI()

MAX_FILE_SIZE = 3 * 1024 * 1024

@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    return await call_next(request)

# 라우터 등록
app.include_router(text_filter_router)
app.include_router(text_filter_rule_router)
app.include_router(timetable)