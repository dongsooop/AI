from transformers import ElectraTokenizer, ElectraForSequenceClassification
import torch
# 저장된 모델과 토크나이저 로드
model = ElectraForSequenceClassification.from_pretrained("model/my_electra_finetuned")
tokenizer = ElectraTokenizer.from_pretrained("model/my_electra_finetuned")

file_path = "data/test_sentence.txt"  # 실제 경로
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

model.eval()
for line in lines:
    text = line.strip()
    if not text:
        continue

    # (1) 전처리(예: 형태소 분석 등) - 필요하다면 동일 과정을 적용해야
    #     만약 학습 시 Okt 등으로 morphs() 했다면, 이 단계에서 동일하게 처리해야 합니다.
    #     아래는 간단 예시(정규식 제거만)
    # text = re.sub(..., "", text)

    # (2) 토크나이저로 인코딩
    encoded = tokenizer.encode_plus(
        text,  # 전처리 후 텍스트
        add_special_tokens=True,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # (3) 모델 추론
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred_label = torch.argmax(logits, dim=-1).item()

    # (4) 결과 표시
    # 라벨 1이면 비속어, 0이면 정상이라고 가정
    if pred_label == 1:
        print(f"[비속어] {text}")
    else:
        print(f"[정상] {text}")