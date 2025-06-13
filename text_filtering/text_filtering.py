from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import ElectraTokenizer, ElectraForSequenceClassification
import torch
import os
import re
from typing import Tuple, List, Dict

router = APIRouter()

class TextRequest(BaseModel):
    text: str

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "my_electra_finetuned")
LOG_PATH = os.path.join(BASE_DIR, "..", "data", "bad_text_sample.txt")

# 모델 및 토크나이저 로드
tokenizer = ElectraTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = ElectraForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 문장 분리 함수
def split_sentences(text: str) -> List[str]:
    text = re.sub(r'([.!?])', r'\1\n', text)
    endings = ['다', '요', '죠', '네', '습니다', '습니까', '해요', '했어요', '하였습니다', '하네요', '해봐요']
    for end in endings:
        text = re.sub(rf'({end})(?=\s)', r'\1\n', text)
    return [s.strip() for s in text.split('\n') if s.strip()]

# 예측 함수
def predict(text: str) -> Tuple[int, str]:
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred_label = torch.argmax(logits, dim=-1).item()

    return pred_label, "비속어" if pred_label == 1 else "정상"

# 분석 함수
def analyze_field(field_name: str, text: str, log_file) -> Dict:
    sentences = split_sentences(text)
    results = []
    has_profanity = False

    for sent in sentences:
        label_num, label_text = predict(sent)
        results.append({
            "sentence": sent,
            "label": label_text
        })
        log_file.write(f"{sent}|{label_num}\n")
        if label_text == "비속어":
            has_profanity = True

    return {
        "field": field_name,
        "has_profanity": has_profanity,
        "results": results
    }

# ✅ API 엔드포인트
@router.post("/text_filter")
async def text_filter_api(request: TextRequest):
    try:
        full_text = request.text.strip()
        try:
            intro, motive = [x.strip() for x in full_text.split('|', 1)]
        except ValueError:
            return JSONResponse(
                content={"error": "입력 형식은 '자기소개 | 지원동기' 이어야 합니다."},
                status_code=422
            )

        with open(LOG_PATH, "a", encoding="utf-8") as f:
            intro_result = analyze_field("자기소개", intro, f)
            motive_result = analyze_field("지원동기", motive, f)

        response = {
            "자기소개": intro_result,
            "지원동기": motive_result
        }

        # 상태코드 조건 분기
        if intro_result["has_profanity"]:
            return JSONResponse(status_code=400, content=response)
        elif motive_result["has_profanity"]:
            return JSONResponse(status_code=401, content=response)
        else:
            return JSONResponse(status_code=200, content=response)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})