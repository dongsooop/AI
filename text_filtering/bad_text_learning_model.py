import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from torch.utils.data import Dataset, DataLoader, random_split
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
        
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long)
        }


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def eval_one_epoch(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total


if __name__ == "__main__":
    train_file = "data/bad_text_sample.txt"
    data_list = []
    
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit("|", 1)
            text = parts[0].strip()
            label = int(parts[1].strip())
            data_list.append((text, label))
    
    print(f"[INFO] 데이터 개수: {len(data_list)}")
    
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    dataset = CommentDataset(data_list, tokenizer, max_len=64)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"[INFO] 학습 세트: {len(train_dataset)} | 검증 세트: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    model = ElectraForSequenceClassification.from_pretrained(
        "monologg/koelectra-base-v3-discriminator",
        num_labels=2
    )
    
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    epochs = 10
    
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = eval_one_epoch(model, val_loader, device)
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
    

    model.save_pretrained(
    "model/my_electra_finetuned",
    safe_serialization=False
    )
    tokenizer.save_pretrained("model/my_electra_finetuned")
    test_text = "시바"
    model.eval()
    
    proc_text = preprocess_text(test_text)
    encoded = tokenizer.encode_plus(
        proc_text,
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
    
    if pred_label == 1:
        print(f"[결과] '{test_text}' => 비속어 판정")
    else:
        print(f"[결과] '{test_text}' => 정상 문장")