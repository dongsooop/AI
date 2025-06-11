from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import ElectraTokenizer, ElectraForSequenceClassification
import torch
import os
from typing import Tuple, List, Dict
import re
from fastapi.responses import JSONResponse

app = FastAPI()

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
    text = re.sub(r'([\.!?])', r'\1\n', text)
    endings = ['다', '요', '죠', '네', '습니다', '습니까', '해요', '했어요', '하였습니다', '하네요', '해봐요']
    for end in endings:
        text = re.sub(rf'({end})(?=\s)', r'\1\n', text)
    return [s.strip() for s in text.split('\n') if s.strip()]

# 예측 함수
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

# 분석 함수
def analyze_field(field_name: str, text: str, log_file=None) -> Dict:
    sentences = split_sentences(text)
    results = []
    has_bad = False

    for sent in sentences:
        label_num, label_text = predict(sent)
        results.append({"sentence": sent, "label": label_text})
        
        # 제목/본문 로그 저장
        if field_name in ['제목', '본문'] and log_file:
            log_file.write(f"{sent}|{label_num}\n")
        
        if label_text == "비속어":
            has_bad = True

    return {
        "field": field_name,
        "has_profanity": has_bad,
        "results": results
    }

# API 엔드포인트
@app.post("/predict")
async def classify_text(request: Request):
    data = await request.json()
    full_text = data.get("text", "")
    
    try:
        title, tags, content = [x.strip() for x in full_text.split('|', 2)]
    except ValueError:
        return JSONResponse(
            content={"error": "text 형식은 '제목 | 태그 | 본문' 이어야 합니다."},
            status_code=422
        )

    # 로그 저장
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        title_result = analyze_field("제목", title, f)
        content_result = analyze_field("본문", content, f)

    tags_result = analyze_field("태그", tags)  # 태그는 로그 미포함

    # 전체 응답 데이터
    response = {
        "제목": title_result,
        "태그": tags_result,
        "본문": content_result
    }

    # 우선순위에 따라 HTTP 상태 코드 반환
    if title_result["has_profanity"]:
        return JSONResponse(content=response, status_code=400)
    elif tags_result["has_profanity"]:
        return JSONResponse(content=response, status_code=401)
    elif content_result["has_profanity"]:
        return JSONResponse(content=response, status_code=402)
    else:
        return JSONResponse(content=response, status_code=200)