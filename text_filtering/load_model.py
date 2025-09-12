from transformers import ElectraTokenizer, ElectraForSequenceClassification
import torch


model = ElectraForSequenceClassification.from_pretrained("model/my_electra_finetuned")
tokenizer = ElectraTokenizer.from_pretrained("model/my_electra_finetuned")
file_path = "data/test_sentence.txt"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

model.eval()
for line in lines:
    text = line.strip()
    
    if not text:
        continue
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred_label = torch.argmax(logits, dim=-1).item()

    if pred_label == 1:
        print(f"[비속어] {text}")
    else:
        print(f"[정상] {text}")