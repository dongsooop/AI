from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.auth import verify_jwt_token
from text_filtering.service import analyze_fields, analyze_text_labels

router = APIRouter()


class TextRequest(BaseModel):
    text: str


@router.post("/text_filter_board")
async def text_filter_board_api(payload: TextRequest, username: str = Depends(verify_jwt_token)):
    try:
        full_text = payload.text.strip()
        try:
            intro, motive = [x.strip() for x in full_text.split('|', 1)]
        except ValueError:
            return JSONResponse(
                content={"error": "입력 형식은 '자기소개 | 지원동기' 이어야 합니다."},
                status_code=422
            )

        analyzed = analyze_fields([
            ("자기소개", intro, True),
            ("지원동기", motive, True),
        ])
        intro_result = analyzed["자기소개"]
        motive_result = analyzed["지원동기"]

        response = {
            "username": username,
            "자기소개": intro_result,
            "지원동기": motive_result
        }

        if intro_result["has_profanity"] or motive_result["has_profanity"]:
            return JSONResponse(status_code=400, content=response)
        else:
            return JSONResponse(status_code=200, content=response)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/text_filter_market")
async def text_filter_market_api(payload: TextRequest, username: str = Depends(verify_jwt_token)):
    try:
        full_text = payload.text.strip()
        try:
            title, content = [x.strip() for x in full_text.split('|', 1)]
        except ValueError:
            return JSONResponse(
                content={"error": "입력 형식은 '제목 | 내용' 이어야 합니다."},
                status_code=422
            )

        analyzed = analyze_fields([
            ("제목", title, True),
            ("내용", content, True),
        ])
        title_result = analyzed["제목"]
        content_result = analyzed["내용"]

        response = {
            "username": username,
            "제목": title_result,
            "내용": content_result
        }

        if title_result["has_profanity"] or content_result["has_profanity"]:
            return JSONResponse(status_code=400, content=response)
        else:
            return JSONResponse(status_code=200, content=response)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/text_filter_single")
async def text_filter_single_api(payload: TextRequest):
    try:
        text = payload.text.strip()
        results = analyze_text_labels(text)

        return JSONResponse(
            status_code=200,
            content={"text" : text, "results" : results}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
