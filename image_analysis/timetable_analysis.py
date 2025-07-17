from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import os
from jose import JWTError, jwt
from dotenv import load_dotenv
import base64
from jose.exceptions import ExpiredSignatureError

router = APIRouter()
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

def verify_jwt_token(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or malformed")

    token = auth_header.split(" ")[1]
    try:
        padded_key = SECRET_KEY + '=' * (-len(SECRET_KEY) % 4)
        sc = base64.urlsafe_b64decode(padded_key)
        payload = jwt.decode(token, sc, algorithms=[ALGORITHM])

        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token: no subject")
        return username

    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


tesseract_config = r"--psm 6 --oem 1 -l kor+eng"

TARGET_WIDTH = 1170
TARGET_HEIGHT = 2532
SCALE = 2

def extract_schedule_fixed_scaled(img):
    resized_fixed = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    resized = cv2.resize(resized_fixed, (TARGET_WIDTH * SCALE, TARGET_HEIGHT * SCALE), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)

    top_left_x = 318 * SCALE
    top_left_y = 670 * SCALE
    cell_w = 160 * SCALE
    cell_h = 66 * SCALE

    def load_list_from_txt(path: str) -> list:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    WEEKDAYS_FILE = "data/weekdays.txt"
    TIME_SLOTS_FILE = "data/time_slots.txt"

    weekdays = load_list_from_txt(WEEKDAYS_FILE)
    time_slots = load_list_from_txt(TIME_SLOTS_FILE)

    results = []
    rows, cols = len(time_slots), len(weekdays)

    for r in range(rows):
        for c in range(cols):
            x1 = top_left_x + c * cell_w
            y1 = top_left_y + r * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            roi = sharpened[int(y1):int(y2), int(x1):int(x2)]
            pil_image = Image.fromarray(roi)
            text = pytesseract.image_to_string(pil_image, config=tesseract_config).strip()
            lines = [re.sub(r"[|]", "", line.strip().replace(" ", "")) for line in text.split("\n") if line.strip()]

            if not lines or len(lines[0]) < 2:
                continue

            course = lines[0]
            professor = lines[1] if len(lines) > 1 else ""
            room = lines[2] if len(lines) > 2 else ""

            if len(course) < 2 or course in ["|", "-", "_"]:
                continue

            if course and professor:
                results.append({
                    "day": weekdays[c],
                    "period": r + 1,
                    "course": course,
                    "professor": professor,
                    "time": time_slots[r],
                    "room": room
                })

    return results

@router.post("/timetable")
async def upload_timetable(request: Request, file: UploadFile = File(...)):
    username = verify_jwt_token(request)

    try:
        file_bytes = await file.read()
        npimg = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        schedule = extract_schedule_fixed_scaled(img)

        return JSONResponse(content={"status": "success", "schedule": schedule})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})