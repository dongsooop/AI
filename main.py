from fastapi import FastAPI
from text_filtering.text_filtering import router as text_filter_router
from text_filtering.text_filtering_rule import router as text_filter_rule_router
from image_analysis.timetable_analysis import router as timetable

app = FastAPI()

# 라우터 등록
app.include_router(text_filter_router)
app.include_router(text_filter_rule_router)
app.include_router(timetable)