import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from transformers import ElectraTokenizer, ElectraForSequenceClassification
from torch.utils.data import Dataset, DataLoader, random_split
from konlpy.tag import Okt


##########################
# 1) 전처리 함수
##########################
okt = Okt()

def preprocess_text(text):
    # 예시: 불필요한 특수문자 제거
    text = re.sub(r"[^가-힣0-9A-Za-z\s\.?!]", "", text).strip()
    # 형태소 분석 -> 토큰 리스트
    tokens = okt.morphs(text)
    # 토큰들을 공백으로 연결 (또는 그대로 리스트로 두어도 무방)
    return " ".join(tokens)


##########################
# 2) Dataset 정의
##########################
class CommentDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_len=64):
        """
        data_list: [(문장, 라벨), (문장, 라벨), ...]
        tokenizer: KoELECTRA 전처리용 tokenizer
        max_len: 최대 토큰 길이
        """
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        # 전처리 (형태소 분석 등)
        proc_text = preprocess_text(text)
        
        # KoELECTRA 토크나이저
        encoded = self.tokenizer.encode_plus(
            proc_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 텐서 형식으로 반환
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long)
        }


##########################
# 3) 학습/검증 루프 정의
##########################
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
            logits = outputs.logits  # shape: (batch_size, 2)
            preds = torch.argmax(logits, dim=-1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total


##########################
# 4) 메인 실행부
##########################
if __name__ == "__main__":
    # ----- 4.1) 데이터 불러오기 (문장|라벨)
    train_file = "data/bad_text_sample.txt"  # 실제 경로에 맞춰 수정
    data_list = []
    
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # "문장|라벨" 형태이므로, 뒷부분만 분리
            parts = line.rsplit("|", 1)
            text = parts[0].strip()
            label = int(parts[1].strip())
            data_list.append((text, label))
    
    print(f"[INFO] 데이터 개수: {len(data_list)}")
    
    # ----- 4.2) Tokenizer & Dataset
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    dataset = CommentDataset(data_list, tokenizer, max_len=64)
    
    # ----- 4.3) Train/Validation 분할
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"[INFO] 학습 세트: {len(train_dataset)} | 검증 세트: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # ----- 4.4) 모델 초기화
    model = ElectraForSequenceClassification.from_pretrained(
        "monologg/koelectra-base-v3-discriminator",
        num_labels=2  # 비속어(1), 정상(0) 이진 분류
    )
    
    # GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 옵티마이저, 에폭 설정
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    epochs = 10
    
    # ----- 4.5) 학습
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = eval_one_epoch(model, val_loader, device)
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    # ----- 4.6) 학습된 모델 저장
    model.save_pretrained(
    "model/my_electra_finetuned",
    safe_serialization=False
    )
    tokenizer.save_pretrained("model/my_electra_finetuned")
    
    # ----- 4.7) 예시로 추론 테스트
    test_text = "애새끼가 초딩도 아니고 ㅋㅋㅋㅋ"
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