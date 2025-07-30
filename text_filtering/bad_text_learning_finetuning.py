import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraTokenizer, ElectraForSequenceClassification, AdamW
import re
from konlpy.tag import Okt


okt = Okt()
def preprocess_text(text):
    text = re.sub(r"[^가-힣0-9A-Za-z\s\.?!]", "", text).strip()
    tokens = okt.morphs(text)
    return " ".join(tokens)


class CommentDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_len=64):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        proc_text = preprocess_text(text)
        encoded = self.tokenizer.encode_plus(
            proc_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

model_path = "model/my_electra_finetuned"
tokenizer = ElectraTokenizer.from_pretrained(model_path)
model = ElectraForSequenceClassification.from_pretrained(model_path)

device = torch.device("mps" if torch.cuda.is_available() else "cpu")
model.to(device)

new_data = []
with open("data/new_bad_text_sample.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        text, label = line.rsplit("|", 1)
        new_data.append((text.strip(), int(label.strip())))

dataset = CommentDataset(new_data, tokenizer)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
optimizer = AdamW(model.parameters(), lr=5e-6)

model.train()
for epoch in range(3):
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    print(f"[추가 학습 Epoch {epoch+1}] Loss: {avg_loss:.4f}")


model.save_pretrained("model/my_electra_finetuned")
tokenizer.save_pretrained("model/my_electra_finetuned")