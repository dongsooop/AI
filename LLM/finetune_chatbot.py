import os, re, unicodedata, json, argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sentencepiece as spm
from torch.cuda.amp import autocast, GradScaler


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=False)

    def forward(self, x):
        embeded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embeded)
        return outputs, (hidden, cell)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)
        seq_len = encoder_outputs.size(1)
        hidden = hidden.repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        attn_weight = F.softmax(attention, dim=1)
        context = torch.bmm(attn_weight.unsqueeze(1), encoder_outputs)
        return context, attn_weight


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size,
                            num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)
        self.attention = Attention(hidden_size)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)
        embedded = self.embedding(input_token)
        context, attn_weights = self.attention(hidden[-1], encoder_outputs)
        lstm_input = torch.cat((embedded, context), dim=2)
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        concat = torch.cat((outputs, context), dim=2)
        logits = self.fc_out(concat).squeeze(1)
        return logits, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.embedding.num_embeddings
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)

        encoder_outputs, (hidden, cell) = self.encoder(src)
        input_token = trg[:, 0]  # <bos>

        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input_token = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1
        return outputs



class SentencePieceTokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        ok = self.sp.load(model_path)
        if not ok:
            raise FileNotFoundError(f"SentencePiece model not found: {model_path}")
        self.vocab_size = self.sp.get_piece_size()
    def encode(self, text): return self.sp.encode(text, out_type=int)
    def decode(self, ids): return self.sp.decode(ids)
    def pad_id(self): return self.sp.pad_id()
    def bos_id(self): return self.sp.bos_id()
    def eos_id(self): return self.sp.eos_id()



def clean_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = re.sub(r'#@\S+', '', s)
    s = re.sub(r'[^\S\r\n]+', ' ', s)
    s = s.strip()
    return unicodedata.normalize('NFC', s)

def looks_bad(s):
    if not s: return True
    s = unicodedata.normalize("NFC", s)
    if len(s) < 2: return True
    if re.search(r'(.)\1{3,}', s): return True
    if "싸우자" in s: return True
    if len(re.sub(r'[\w가-힣]', '', s)) > len(s) * 0.5: return True
    return False


def clean_pairs(path_in, path_out):
    data = json.load(open(path_in, encoding="utf-8"))
    out = []
    for ex in data:
        src = clean_text(ex["input"]); tgt = clean_text(ex["target"])
        if not src or not tgt: continue
        if looks_bad(tgt): continue
        out.append({"input": src, "target": tgt})
    json.dump(out, open(path_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"✅ {len(out)} examples -> {path_out}")


class ChatDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_len=64):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tok = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        it = clean_text(self.data[idx]["input"])
        tt = clean_text(self.data[idx]["target"])
        inp = [self.tok.bos_id()] + self.tok.encode(it) + [self.tok.eos_id()]
        tgt = [self.tok.bos_id()] + self.tok.encode(tt) + [self.tok.eos_id()]
        inp = inp[:self.max_len] + [self.tok.pad_id()] * max(0, self.max_len - len(inp))
        tgt = tgt[:self.max_len] + [self.tok.pad_id()] * max(0, self.max_len - len(tgt))
        return torch.tensor(inp), torch.tensor(tgt)


def _resize_linear_out(linear: nn.Linear, new_out: int):
    old_out, in_dim = linear.weight.shape
    new_linear = nn.Linear(in_dim, new_out, bias=True)
    with torch.no_grad():
        copy = min(old_out, new_out)
        new_linear.weight[:copy] = linear.weight[:copy]
        if linear.bias is not None:
            new_linear.bias[:copy] = linear.bias[:copy]
    return new_linear


def resize_embeddings_if_needed(model, tokenizer):
    new_vocab = tokenizer.vocab_size
    old_vocab = model.decoder.embedding.num_embeddings

    if new_vocab == old_vocab:
        return model, False
    
    def _resize_embedding(emb: nn.Embedding, new_num):
        new_emb = nn.Embedding(new_num, emb.embedding_dim, padding_idx=emb.padding_idx)
        with torch.no_grad():
            copy = min(emb.num_embeddings, new_num)
            new_emb.weight[:copy] = emb.weight[:copy]
            if emb.padding_idx is not None and emb.padding_idx < new_num:
                new_emb.weight[emb.padding_idx].zero_()
        return new_emb
    
    model.encoder.embedding = _resize_embedding(model.encoder.embedding, new_vocab)
    model.decoder.embedding = _resize_embedding(model.decoder.embedding, new_vocab)
    model.decoder.fc_out   = _resize_linear_out(model.decoder.fc_out, new_out=new_vocab)

    return model, True


def load_checkpoint(model, optimizer=None, ckpt_path=None, map_location=None):
    if not ckpt_path or not os.path.exists(ckpt_path):
        print("🆕 체크포인트 없음 → 새로 시작")
        return 0, None
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if optimizer and 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception as e:
                print(f"⚠️ 옵티마이저 로드 무시: {e}")
        start_epoch = ckpt.get('epoch', -1) + 1
        print(f"✅ 체크포인트 로드 (epoch={start_epoch})")
        return start_epoch, ckpt
    else:
        model.load_state_dict(ckpt, strict=False)
        print("✅ state_dict 로드")
        return 0, ckpt


def _make_criterion(pad_id, smoothing=0.1):
    try:
        return nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=smoothing)
    except TypeError:
        return nn.CrossEntropyLoss(ignore_index=pad_id)


def train_one_epoch(model, loader, tokenizer, optimizer, device, tfr=0.6, clip=1.0, smoothing=0.1):
    model.train()
    pad_id = tokenizer.pad_id()
    crit = _make_criterion(pad_id, smoothing=smoothing)
    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    total = 0.0
    for src, trg in tqdm(loader, desc="train"):
        src = src.to(device, non_blocking=True)
        trg = trg.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            out = model(src, trg, teacher_forcing_ratio=tfr)
            V = out.size(-1)
            loss = crit(out[:, 1:].reshape(-1, V), trg[:, 1:].reshape(-1))

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        total += loss.item()

    return total / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, smoothing=0.1):
    model.eval()
    pad_id = tokenizer.pad_id()
    crit = _make_criterion(pad_id, smoothing=smoothing)

    use_amp = (device.type == "cuda")
    total = 0.0

    for src, trg in tqdm(loader, desc="valid"):
        src = src.to(device, non_blocking=True)
        trg = trg.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            out = model(src, trg, teacher_forcing_ratio=0.0)
            V = out.size(-1)
            loss = crit(out[:, 1:].reshape(-1, V), trg[:, 1:].reshape(-1))

        total += loss.item()

    return total / max(1, len(loader))


def finetune(model, tokenizer, train_loader, valid_loader, device,
            ckpt_in=None, ckpt_out_dir="./ft_ckpts",
            lr=5e-4, epochs_p1=2, epochs_p2=4, freeze_encoder_first=True):
    os.makedirs(ckpt_out_dir, exist_ok=True)
    model, resized = resize_embeddings_if_needed(model, tokenizer)
    if resized: print("🔧 임베딩/출력층 리사이즈 → 토크나이저와 동기화")
    model.to(device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    enc_params = list(model.encoder.parameters())
    dec_params = list(model.decoder.parameters())

    if freeze_encoder_first:
        for p in enc_params: p.requires_grad = False
        optimizer = optim.Adam([p for p in dec_params if p.requires_grad], lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    sched = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True, min_lr=1e-6
    )

    start_epoch, _ = load_checkpoint(model, optimizer, ckpt_in, map_location=device)
    epoch_idx = start_epoch

    for _ in range(epochs_p1):
        tr = train_one_epoch(model, train_loader, tokenizer, optimizer, device, tfr=0.7, smoothing=0.1)
        va = evaluate(model, valid_loader, tokenizer, device, smoothing=0.1)
        sched.step(va)
        print(f"[P1][{epoch_idx+1}] train={tr:.4f} | valid={va:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")
        torch.save({
            "epoch": epoch_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "valid_loss": va
        }, os.path.join(ckpt_out_dir, f"ft_epoch{epoch_idx+1}.pt"))
        epoch_idx += 1

    for p in enc_params: p.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=lr*0.5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True, min_lr=1e-6
    )

    for e in range(epochs_p2):
        tfr = max(0.3, 0.7 - 0.1*e)
        tr = train_one_epoch(model, train_loader, tokenizer, optimizer, device, tfr=tfr, smoothing=0.1)
        va = evaluate(model, valid_loader, tokenizer, device, smoothing=0.1)
        sched.step(va)
        print(f"[P2][{epoch_idx+1}] train={tr:.4f} | valid={va:.4f} | tfr={tfr:.2f} | lr={optimizer.param_groups[0]['lr']:.2e}")
        torch.save({
            "epoch": epoch_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "valid_loss": va
        }, os.path.join(ckpt_out_dir, f"ft_epoch{epoch_idx+1}.pt"))
        epoch_idx += 1

    print("✅ 파인튜닝 완료")
    return model


@torch.no_grad()
def generate(model, tokenizer, text, device, max_len=64,
            temperature=0.9, top_k=50, top_p=0.9,
            repetition_penalty=1.2, min_len=3,
            no_repeat_ngram_size=3,
            bad_words=("싸우자",),):
    model.eval()

    def ban_bad_words(probs):
        for bw in bad_words:
            ids = tokenizer.encode(bw)
            if len(ids) == 1:
                probs[ids[0]] = 0.0
        return probs

    def ban_no_repeat_ngram(probs, generated, n):
        if n <= 0 or len(generated) < n-1:
            return probs
        tail = generated[-(n-1):]
        for i in range(len(generated)-(n-1)):
            if generated[i:i+n-1] == tail:
                nxt = generated[i+n-1]
                probs[nxt] = 0.0
        return probs

    src_ids = [tokenizer.bos_id()] + tokenizer.encode(clean_text(text)) + [tokenizer.eos_id()]
    src_ids = src_ids[:max_len]
    pad = tokenizer.pad_id()
    src = torch.tensor(src_ids + [pad]*(max_len-len(src_ids)), dtype=torch.long, device=device).unsqueeze(0)

    enc_out, (h, c) = model.encoder(src)
    inp = torch.tensor([[tokenizer.bos_id()]], device=device)
    out_ids, seen = [], {}

    for t in range(max_len):
        logits, h, c, _ = model.decoder(inp, h, c, enc_out)
        logits = logits.squeeze(0)

        for tok, cnt in seen.items():
            if cnt > 0:
                logits[tok] /= repetition_penalty

        logits = logits / max(1e-8, temperature)
        probs = F.softmax(logits, dim=-1)

        probs = ban_bad_words(probs)
        probs = ban_no_repeat_ngram(probs, out_ids, no_repeat_ngram_size)

        if top_k and top_k > 0:
            topk = torch.topk(probs, k=min(top_k, probs.numel()))
            mask = torch.full_like(probs, 0.0)
            mask.scatter_(0, topk.indices, topk.values)
            probs = mask

        if top_p and 0 < top_p < 1:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cumsum > top_p)
            cutoff_idx = torch.argmax(cutoff.int()).item() if cutoff.any() else (len(sorted_probs)-1)
            keep = sorted_idx[:cutoff_idx+1]
            mask = torch.zeros_like(probs)
            mask[keep] = probs[keep]
            probs = mask

        if probs.sum() <= 0:
            probs = F.softmax(logits, dim=-1)

        next_id = torch.multinomial(probs, num_samples=1).item()

        if t < min_len and next_id == tokenizer.eos_id():
            next_id = torch.topk(probs, k=2).indices[1].item()

        if next_id == tokenizer.eos_id():
            break
        out_ids.append(next_id)
        seen[next_id] = seen.get(next_id, 0) + 1
        inp = torch.tensor([[next_id]], device=device)

    return clean_text(tokenizer.decode(out_ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spm_model", default="model/llm_model/chatbot_spm.model")
    ap.add_argument("--train_json", default="data/converted_dataset/train_chatbot_data.json")
    ap.add_argument("--valid_json", default="data/converted_dataset/valid_chatbot_data.json")
    ap.add_argument("--embed_size", type=int, default=256)
    ap.add_argument("--hidden_size", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs_p1", type=int, default=2)
    ap.add_argument("--epochs_p2", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--ckpt_in", default=None)
    ap.add_argument("--ckpt_out", default="./ft_ckpts")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                        else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print("🔌 device:", device)

    tok = SentencePieceTokenizer(args.spm_model)

    train_ds = ChatDataset(args.train_json, tok, max_len=args.max_len)
    valid_ds = ChatDataset(args.valid_json, tok, max_len=args.max_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=0, pin_memory=(device.type=="cuda"))
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        num_workers=0, pin_memory=(device.type=="cuda"))

    vocab = tok.vocab_size
    enc = Encoder(vocab, args.embed_size, args.hidden_size, num_layers=args.layers, dropout=args.dropout)
    dec = Decoder(vocab, args.embed_size, args.hidden_size, num_layers=args.layers, dropout=args.dropout)
    model = Seq2Seq(enc, dec, device)

    model = finetune(
        model, tok, train_dl, valid_dl, device,
        ckpt_in=args.ckpt_in, ckpt_out_dir=args.ckpt_out,
        lr=args.lr, epochs_p1=args.epochs_p1, epochs_p2=args.epochs_p2,
        freeze_encoder_first=True
    )

    for q in ["안녕 친구야?", "오늘 뭐해?", "배고파.."]:
        print("👤:", q)
        print("🤖:", generate(model, tok, q, device))


if __name__ == "__main__":
    main()