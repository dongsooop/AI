import re
from konlpy.tag import Okt
from transformers import ElectraTokenizer, ElectraForSequenceClassification
import torch
import torch.nn.functional as F


sentence = "친구야 정말 죽고싶은거야?"

# 문자 정규화, 공백제거
sentence = re.sub(r"[^가-힣0-9A-Za-z\s]", "", sentence).strip()

# 형태소 분석
okt = Okt()

# 문장 형태소 단위로 분리
tokens = okt.morphs(sentence)

# 불용어 사전 로드
with open("data/bad_text.txt", "r", encoding="utf-8") as file:
    bad_text_list = file.read().strip().split(",")

# 토큰 순회 후 bad_text_list에 포함된 단어가 있는지 확인
tagged_tokens = []
for t in tokens:
    if t in bad_text_list:
        tagged_tokens.append("[BAD]")
    else: 
        tagged_tokens.append(t)

# 마스킹 토큰 하나의 문자열로 합침
masked_sentence = " ".join(tagged_tokens)

# KoBERT or KoELECTRA 모델을 사용
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

# tagged_tokens 합치기 및 encode에 전달
tagged_sentence = " ".join(tagged_tokens)
encoded = tokenizer.encode_plus(
    tagged_sentence,
    add_special_tokens=True,
    max_length=64,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)

# 모델 인퍼런스(Fine-tuned) -> 비속어 확률
model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator")

# 추론
model.eval()

with torch.no_grad():
    outputs = model(
        input_ids = encoded["input_ids"],
        attention_mask = encoded["attention_mask"]
    )
    logits = outputs.logits

# 임계값 설정 & 비속어 판정
probs = F.softmax(logits, dim=1)
bad_prob = probs[0][1].item()

# 임계값 설정 후 판단
threshold = 0.45

print("==== 디버깅 정보 ====")
print("원 문장         :", sentence)
print("형태소 토큰     :", tokens)
print("비속어 확률     :", bad_prob)

# 13) 결과 출력
if bad_prob >= threshold:
    print("비속어 문장 (확률: {:.2f}%)".format(bad_prob*100))
else:
    print("정상 문장 (비속어 확률: {:.2f}%)".format(bad_prob*100))