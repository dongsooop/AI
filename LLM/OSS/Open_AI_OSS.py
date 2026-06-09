from fastapi import APIRouter, Depends

from core.auth import verify_jwt_token
from LLM.OSS.service import ChatReq, chat_with_oss


router = APIRouter()


@router.post("/chatbot")
async def chat(req: ChatReq, username: str = Depends(verify_jwt_token)):
    _ = username
    return await chat_with_oss(req)
