from fastapi import APIRouter, File, Request, UploadFile

from core.auth import verify_jwt_token
from image_analysis.service import enqueue_timetable_analysis, start_queue_workers


router = APIRouter()


@router.post("/timetable_analysis")
async def upload_timetable(request: Request, file: UploadFile = File(...)):
    user_id = verify_jwt_token(request)
    return await enqueue_timetable_analysis(request, file, user_id)
