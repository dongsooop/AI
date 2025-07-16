from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from jose import JWTError, jwt
import torch
import os
from typing import Tuple, List, Dict
import re
from dotenv import load_dotenv
import base64


load_dotenv()
router = APIRouter()

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
    except JWTError as e:
        raise HTTPException(status_code=401, detail="Invalid token")


class TextRequest(BaseModel):
    text: str


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "my_electra_finetuned")
LOG_PATH = os.path.join(BASE_DIR, "..", "data", "bad_text_sample.txt")


tokenizer = ElectraTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = ElectraForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def split_sentences(text: str) -> List[str]:
    text = re.sub(r'([\.!?])', r'\1\n', text)
    endings = ['다', '요', '죠', '네', '습니다', '습니까', '해요', '했어요', '하였습니다', '하네요', '해봐요']
    for end in endings:
        text = re.sub(rf'({end})(?=\s)', r'\1\n', text)
    return [s.strip() for s in text.split('\n') if s.strip()]


def predict(text: str) -> Tuple[int, str]:
    encoded = tokenizer.encode_plus(
        text, add_special_tokens=True, max_length=64,
        padding="max_length", truncation=True, return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred_label = torch.argmax(logits, dim=-1).item()
    return pred_label, "비속어" if pred_label == 1 else "정상"


def analyze_field(field_name: str, text: str, log_file=None) -> Dict:
    sentences = split_sentences(text)
    results = []
    has_bad = False

    for sent in sentences:
        label_num, label_text = predict(sent)
        results.append({"sentence": sent, "label": label_text})
        
        if field_name in ['제목', '본문'] and log_file:
            log_file.write(f"{sent}|{label_num}\n")
        
        if label_text == "비속어":
            has_bad = True

    return {
        "field": field_name,
        "has_profanity": has_bad,
        "results": results
    }


@router.post("/text_filter_rule")
async def rule_filter_api(
    request: Request,
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

        with open(LOG_PATH, "a", encoding="utf-8") as f:
            title_result = analyze_field("제목", title, f)
            content_result = analyze_field("본문", content, f)

        tags_result = analyze_field("태그", tags)  # 태그는 로그 미포함

        response = {
            "username": username,
            "제목": title_result,
            "태그": tags_result,
            "본문": content_result
        }

        if title_result["has_profanity"]:
            return JSONResponse(content=response, status_code=400)
        elif tags_result["has_profanity"]:
            return JSONResponse(content=response, status_code=400)
        elif content_result["has_profanity"]:
            return JSONResponse(content=response, status_code=400)
        else:
            return JSONResponse(content=response, status_code=200)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "서버 내부 오류가 발생했습니다.", "detail": str(e)}
        )