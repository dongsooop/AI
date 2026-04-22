from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.auth import verify_jwt_token
from services.text_filtering import analyze_fields

router = APIRouter()


class TextRequest(BaseModel):
    text: str


@router.post("/text_filter_rule")
async def rule_filter_api(
    payload: TextRequest,
    username: str = Depends(verify_jwt_token)
):
    try:
        full_text = payload.text.strip()

        try:
            title, tags, content = [x.strip() for x in full_text.split('|', 2)]
        except ValueError:
            return JSONResponse(
                content={"error": "text 형식은 '제목 | 태그 | 본문' 이어야 합니다."},
                status_code=422
            )

        analyzed = analyze_fields([
            ("제목", title, True),
            ("태그", tags, False),
            ("본문", content, True),
        ])
        title_result = analyzed["제목"]
        tags_result = analyzed["태그"]
        content_result = analyzed["본문"]

        response = {
            "username": username,
            "제목": title_result,
            "태그": tags_result,
            "본문": content_result
        }

        if title_result["has_profanity"] or tags_result["has_profanity"] or content_result["has_profanity"]:
            return JSONResponse(content=response, status_code=400)
        else:
            return JSONResponse(content=response, status_code=200)

    except Exception as e:
        return JSONResponse(
            status_code=500,content={"error": str(e)}
        )


@router.post("/text_filter_content")
async def text_filter_content_api(
    payload: TextRequest):
    try:
        text = payload.text.strip()
        analyzed = analyze_fields([
            ("content", text, True),
        ])
        content_result = analyzed["content"]

        response = {
            "content": content_result
        }

        return JSONResponse(
            status_code=400 if content_result["has_profanity"] else 200,
            content=response
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
