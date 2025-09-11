import re
from konlpy.tag import Okt
from transformers import ElectraTokenizer, ElectraForSequenceClassification
import torch
import torch.nn.functional as F

sentence = "친구야 정말 죽고싶은거야?"

sentence = re.sub(r"[^가-힣0-9A-Za-z\s]", "", sentence).strip()
okt = Okt()
tokens = okt.morphs(sentence)

with open("data/bad_text.txt", "r", encoding="utf-8") as file:
    bad_text_list = file.read().strip().split(",")

tagged_tokens = []
for t in tokens:
    if t in bad_text_list:
        tagged_tokens.append("[BAD]")
    else: 
        tagged_tokens.append(t)

masked_sentence = " ".join(tagged_tokens)
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
tagged_sentence = " ".join(tagged_tokens)
encoded = tokenizer.encode_plus(
    tagged_sentence,
    add_special_tokens=True,
    max_length=64,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)


model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator")

model.eval()

with torch.no_grad():
    outputs = model(
        input_ids = encoded["input_ids"],
        attention_mask = encoded["attention_mask"]
    )
    logits = outputs.logits

probs = F.softmax(logits, dim=1)
bad_prob = probs[0][1].item()
threshold = 0.45

print("==== 디버깅 정보 ====")
print("원 문장         :", sentence)
print("형태소 토큰     :", tokens)
print("비속어 확률     :", bad_prob)

if bad_prob >= threshold:
    print("비속어 문장 (확률: {:.2f}%)".format(bad_prob*100))
else:
    print("정상 문장 (비속어 확률: {:.2f}%)".format(bad_prob*100))