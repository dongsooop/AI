from fastapi import FastAPI, HTTPException
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

# 모델 및 토크나이저 로드
tokenizer = ElectraTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = ElectraForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

# 장비 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ✅ Rule-based 문장 분리 함수
def split_sentences(text: str) -> List[str]:
    text = re.sub(r'([\.!?])', r'\1\n', text)
    endings = ['다', '요', '죠', '네', '습니다', '습니까', '해요', '했어요', '하였습니다', '하네요', '해봐요']
    for end in endings:
        text = re.sub(rf'({end})(?=\s)', r'\1\n', text)
    return [s.strip() for s in text.split('\n') if s.strip()]

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
    has_bad = False

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        for sent in sentences:
            label_num, label_text = predict(sent)
            f.write(f"{sent}|{label_num}\n")
            results.append({"sentence": sent, "label": label_text})
            if label_text == "비속어":
                has_bad = True

    if has_bad:
        raise HTTPException(status_code=400, detail={"results": results})

    return {"results": results}