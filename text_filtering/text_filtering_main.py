from fastapi import FastAPI
from text_filtering import router as text_filter_router
from text_filtering_rule import router as text_filter_rule_router

app = FastAPI()

# 두 개의 라우터 등록
app.include_router(text_filter_router)
app.include_router(text_filter_rule_router)