from fastapi import FastAPI
from pydantic import BaseModel
from transformers import ElectraTokenizer, ElectraForSequenceClassification
import torch
import os
from typing import Tuple

# FastAPI 앱 생성
app = FastAPI()

# 요청 바디 정의
class TextRequest(BaseModel):
    text: str

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "my_electra_finetuned")
LOG_PATH = os.path.join(BASE_DIR, "..", "data", "bad_text_sample.txt")
# AWS전용 PATH수정
# LOG_PATH = "/app/data/bad_text_sample.txt"

# 모델 및 토크나이저 로드 (로컬에서)
tokenizer = ElectraTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = ElectraForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

# 장비 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 예측 함수 (label 숫자와 텍스트 반환)
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

# API 엔드포인트
@app.post("/predict")
async def classify_text(request: TextRequest):
    text = request.text.strip()
    label_num, label_text = predict(text)

    # 🔥 저장: 텍스트|0 또는 텍스트|1
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{text}|{label_num}\n")

    return {"label": label_text}