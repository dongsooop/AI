# -*- coding: utf-8 -*-
import os, re, json, pickle, numpy as np, pandas as pd
from pathlib import Path
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from index_utils import get_tokenizer
from dotenv import load_dotenv

# ---------- 경로 해석 ----------
load_dotenv()

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent

ART_DIR = Path(os.environ.get("ART_DIR"))

print(f"[PATH] ART_DIR = {ART_DIR}")

# ---------- 로드 ----------
search_df = pd.read_parquet(ART_DIR / "search_df.parquet")
embeddings = np.load(ART_DIR / "embeddings.npy", mmap_mode="r").astype(np.float32)

tokenize_kor = get_tokenizer()
if (ART_DIR / "bm25.pkl").exists():
    with open(ART_DIR / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
else:
    from index_utils import load_json_gz
    toks = load_json_gz(ART_DIR / "tokenized.json.gz")
    bm25 = BM25Okapi(toks)

model_name = "intfloat/multilingual-e5-base"
model = SentenceTransformer(model_name)

@lru_cache(maxsize=512)
def embed_query(q: str):
    return model.encode([f"query: {q}"], normalize_embeddings=True)[0].astype(np.float32)

def _minmax(x):
    x = np.asarray(x, float)
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo + 1e-9)

NOTICE_KWS = ["공지","안내","모집","변경","연장","일정","신청","발표","채용","장학","기간","마감","등록","수강","정정"]
CONTACT_KWS = ["연락처","전화","전화번호","이메일","메일","담당자","문의","상담","교직원","직원","contact","email","phone"]

def recency_score(ts, half_life_days=30):
    if pd.isna(ts) or ts is None: return 0.0
    ts = pd.to_datetime(ts)
    days = max((pd.Timestamp.now(tz="Asia/Seoul") - ts).days, 0)
    return float(np.exp(-np.log(2) * days / half_life_days))

def hybrid_search(query, top_k=8, alpha=0.6, recency_weight=0.45, notice_boost=0.20, contact_boost=0.18):
    qv = embed_query(query)
    dense_raw = embeddings @ qv
    bm25_raw  = bm25.get_scores(tokenize_kor(query))
    base = alpha * _minmax(dense_raw) + (1 - alpha) * _minmax(bm25_raw)

    rec = _minmax(np.array([recency_score(ts) for ts in search_df["updated_at"]]))
    bonus = recency_weight * rec

    if any(k in query for k in NOTICE_KWS):
        bonus = bonus + notice_boost * search_df["notice_flag"].to_numpy()

    if any(k in query for k in CONTACT_KWS):
        is_contact = (search_df["doc_type"] == "contact").astype(float).to_numpy()
        has_phone = (search_df["phone"].astype(str) != "").astype(float).to_numpy()
        has_email = (search_df["email"].astype(str) != "").astype(float).to_numpy()
        bonus = bonus + contact_boost * (is_contact + 0.5*has_phone + 0.5*has_email)

    scores = base + bonus
    out = search_df.copy()
    out["score"] = scores
    rep_idx = out.groupby(["url","doc_type"])["score"].idxmax()
    out = out.loc[rep_idx]
    out = out.sort_values(["score","updated_at"], ascending=[False, False], na_position="last").head(top_k)
    return out.reset_index(drop=True)

def _snippet(text, q, max_chars=220):
    if not isinstance(text, str): return ""
    t = " ".join(text.split())
    if len(t) <= max_chars: return t
    toks = [x for x in re.findall(r"[가-힣A-Za-z0-9]+", q) if len(x) >= 2]
    pos = min([t.lower().find(x.lower()) for x in toks if t.lower().find(x.lower()) >= 0] or [0])
    start = max(pos - 60, 0); end = min(start + max_chars, len(t))
    return (("…" if start>0 else "") + t[start:end] + ("…" if end<len(t) else ""))

def build_answer(query, top_k=6):
    hits = hybrid_search(query, top_k=top_k)
    contact_like = any(k in query for k in CONTACT_KWS)
    top_contacts = hits[hits["doc_type"]=="contact"]
    if contact_like and not top_contacts.empty:
        lines = []
        for _, r in top_contacts.head(3).iterrows():
            p = r.get("phone", ""); e = r.get("email", ""); u = r.get("unit", "")
            parts = []
            if isinstance(p, str) and p and p != "nan": parts.append(f"전화 {p}")
            if isinstance(e, str) and e and e != "nan": parts.append(f"이메일 {e}")
            joined = " / ".join(parts) if parts else "연락처 정보가 텍스트에 포함됨"
            lines.append(f"- {u}: {joined}  (출처: {r['title']})")
        others = hits[hits["doc_type"]!="contact"].head(2)
        srcs = [{"title": r["title"], "url": r["url"]} for _, r in others.iterrows()]
        srcs = [{"title": r["title"], "url": r["url"]} for _, r in top_contacts.head(3).iterrows()] + srcs
        return {"answer": "\n".join(lines), "sources": srcs}
    else:
        lines = [f"- {r['title']}: {_snippet(r['text'], query)}" for _, r in hits.head(3).iterrows()]
        srcs = [{"title": r["title"], "url": r["url"]} for _, r in hits.head(3).iterrows()]
        return {"answer": "\n".join(lines), "sources": srcs}

if __name__ == "__main__":
    q1 = "학생성공지원팀 담당자 전화번호 알려줘"
    q2 = "대학소개와 비전 전략체계를 요약해줘"
    for q in [q1, q2]:
        res = build_answer(q, top_k=8)
        print("Q:", q)
        print("A:\n", res["answer"])
        print("SOURCES:", json.dumps(res["sources"], ensure_ascii=False, indent=2))
        print("="*80)