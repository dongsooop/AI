from fastapi import FastAPI
from pydantic import BaseModel
from transformers import ElectraTokenizer, ElectraForSequenceClassification
import torch
import os
from typing import Tuple, List
import re

# FastAPI 앱 생성
app = FastAPI()

# 요청 바디 정의
class TextRequest(BaseModel):
    text: str

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "my_electra_finetuned")
LOG_PATH = os.path.join(BASE_DIR, "..", "data", "bad_text_sample.txt")
# EC2 환경이라면 아래 경로 사용
# LOG_PATH = "/app/data/bad_text_sample.txt"

# 모델 및 토크나이저 로드
tokenizer = ElectraTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = ElectraForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

# 장비 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ✅ Rule-based 문장 분리 함수
def split_sentences(text: str) -> List[str]:
    # 1. 마침표, 느낌표, 물음표 처리
    text = re.sub(r'([\.!?])', r'\1\n', text)

    # 2. 조사나 종결어미 기반 추가 개행
    endings = ['다', '요', '죠', '네', '습니다', '습니까', '해요', '했어요', '하였습니다', '하네요', '해봐요']
    for end in endings:
        text = re.sub(rf'({end})(?=\s)', r'\1\n', text)

    # 4. 개행 기준 문장 분리 후 정제
    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    return sentences

# ✅ 예측 함수
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

# ✅ API 엔드포인트
@app.post("/predict")
async def classify_text(request: TextRequest):
    full_text = request.text.strip()
    sentences = split_sentences(full_text)

    results = []
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        for sent in sentences:
            label_num, label_text = predict(sent)
            f.write(f"{sent}|{label_num}\n")
            results.append({"sentence": sent, "label": label_text})

    return {"results": results}