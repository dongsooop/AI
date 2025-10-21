import os, json, re, hashlib, pickle, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

from LLM.sub_model.index_utils import (
    normalize_text, chunk_text, get_tokenizer,
    extract_units_and_contacts, clean_name, compose_contact_passage,
    dump_json_gz
)
# from index_utils import (
#     normalize_text, chunk_text, get_tokenizer,
#     extract_units_and_contacts, clean_name, compose_contact_passage,
#     dump_json_gz
# )


load_dotenv()

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent.parent

def env_path(name: str) -> Path:
    val = os.getenv(name, "").strip()
    if not val:
        raise FileNotFoundError(f"{name} environment empty.")
    s = os.path.expandvars(val)
    p = Path(s)
    if not p.is_absolute():
        p = (ROOT_DIR / p)
    return p.resolve()


DATA_JSON        = env_path("DATA_JSON")
ART_DIR          = env_path("ART_DIR")
SEARCH_DF_PATH   = env_path("SEARCH_DF_PATH")
EMB_PATH         = env_path("EMB_PATH")
BM25_PATH        = env_path("BM25_PATH")
TOK_PATH         = env_path("TOK_PATH")
CONTACTS_CSV     = env_path("CONTACTS_CSV")
META_PATH        = env_path("META_PATH")
CONTACT_KWS_PATH = env_path("CONTACT_KWS_PATH")

print(f"📁 데이터 경로 확인:")
print(f"   - DATA_JSON: {DATA_JSON}")
print(f"   - ART_DIR: {ART_DIR}")

if not DATA_JSON.exists():
    raise FileNotFoundError(f"메인 데이터 파일을 찾을 수 없습니다: {DATA_JSON}")
    
# 메인 데이터만 로드 (공지사항 제외)
print("📖 학교 기본 정보 로딩 중...")
with open(DATA_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data).dropna(subset=["content"])
df["title"]  = df["title"].fillna("")
df["url"]    = df["url"].fillna("")
df["source"] = "main"
print(f"   - 학교 기본 정보: {len(df)}개 문서")


# 텍스트 정규화 및 중복 제거
print("🔧 텍스트 정규화 및 중복 제거 중...")
df["title"]       = df["title"].apply(normalize_text)
df["content"]     = df["content"].apply(normalize_text)
df["fulltext"]    = (df["title"] + " " + df["content"]).apply(normalize_text)

# 중복 제거 (URL 기준만)
df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
print(f"   - 중복 제거 후: {len(df)}개 문서")


# 문서를 청크로 분할
print("✂️ 문서 청킹 중...")
rows = []
for i, r in df.iterrows():
    parts = chunk_text(r["fulltext"], max_tokens=400, overlap=0.15)
    for j, ch in enumerate(parts):
        rows.append({
            "doc_id": i,
            "chunk_id": f"{i}-{j}",
            "title": r["title"],
            "url": r["url"],
            "source": r["source"],
            "text": ch,
            "doc_type": "page",
            "unit": "", "phone": "", "email": ""
        })

page_chunks = pd.DataFrame(rows).reset_index(drop=True)
print(f"   - 총 청크 수: {len(page_chunks)}개")


# 연락처 추출 (학교 기본 정보에서만)
print("📞 연락처 추출 중...")
DASH_CHARS = r"\-\u2010\u2011\u2012\u2013\u2014\u2212\uFE58\uFE63\uFF0D"
def _preclean_contact_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\u00A0", " ").replace("\u200B", "")
    s = re.sub(f"[{DASH_CHARS}]", "-", s)
    return s

contact_cands = []

# 메인 데이터에서만 연락처 추출
for _, r in df.iterrows():
    row_like = pd.Series({
        "title":   r["title"],
        "content": _preclean_contact_text(r["content"]),
        "url":     r["url"],
        "source":  "main",
    })
    contact_cands.extend(extract_units_and_contacts(row_like))

contacts_raw = pd.DataFrame(contact_cands)
print(f"   - 연락처 후보: {len(contacts_raw)}개")


if contacts_raw.empty:
    print("⚠️ 연락처 후보가 비어 있습니다. 추출 로직/입력 텍스트를 확인하세요.")
    contact_docs = pd.DataFrame()
else:
    print("   - 연락처 후보 상세:")
    try:
        print(contacts_raw.head(10)[["unit","phone","email","title","url"]].to_string(index=False))
    except Exception:
        print(contacts_raw.head(10).to_string(index=False))

    # 연락처 클러스터링 (중복 제거)
    contacts_raw["unit_norm"] = contacts_raw["unit"].apply(clean_name)
    CLUSTER_SIM = 0.92
    cluster_map, reps = {}, []
    for name in contacts_raw["unit_norm"].dropna().unique():
        assigned = False
        for cid, rep in enumerate(reps):
            if SequenceMatcher(None, name, rep).ratio() >= CLUSTER_SIM:
                cluster_map[name] = cid
                assigned = True
                break
        if not assigned:
            cid = len(reps)
            reps.append(name)
            cluster_map[name] = cid

    contacts_raw["cluster_id"] = contacts_raw["unit_norm"].map(cluster_map.get)

    rep_map = {}
    for cid in set(x for x in contacts_raw["cluster_id"] if pd.notna(x)):
        members = contacts_raw[contacts_raw["cluster_id"] == cid]
        rep_map[cid] = members["unit"].value_counts().idxmax()

    contacts_raw["unit_canonical"] = contacts_raw["cluster_id"].map(rep_map.get)
    contacts_df = (
        contacts_raw
        .sort_values(["cluster_id", "url"])
        .drop_duplicates(subset=["cluster_id", "phone", "email"])
        .reset_index(drop=True)
    )
    
    # 연락처 문서 생성
    contact_rows = []
    for idx, r in contacts_df.iterrows():
        txt = compose_contact_passage(
            r.get("unit_canonical", r.get("unit", "")),
            r.get("phone", ""),
            r.get("email", ""),
            r.get("title", ""),
            r.get("url", "")
        )
        contact_rows.append({
            "doc_id": -1,
            "chunk_id": f"c-{idx}",
            "title": f"{r.get('unit_canonical', r.get('unit',''))} 연락처",
            "url": r.get("url", ""),
            "source": r.get("source", ""),
            "text": txt,
            "doc_type": "contact",
            "unit": r.get("unit_canonical", r.get("unit","")),
            "phone": r.get("phone", ""),
            "email": r.get("email", "")
        })

    contact_docs = pd.DataFrame(contact_rows)
    print(f"   - 최종 연락처 문서: {len(contact_docs)}개")


# 최종 데이터 통합
print("🔗 최종 데이터 통합 중...")
ALL_COLS = [
    "doc_id","chunk_id","title","url","source",
    "text","doc_type","unit","phone","email"
]

def _ensure_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    for c in ALL_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[ALL_COLS]
    df["doc_id"]      = pd.to_numeric(df["doc_id"], errors="coerce").fillna(-1).astype("int64")
    for c in ["chunk_id","title","url","source","text","doc_type","unit","phone","email"]:
        df[c] = df[c].astype("object")
    return df

page_chunks  = _ensure_columns(page_chunks)
contact_docs = _ensure_columns(contact_docs)
search_df = pd.concat([page_chunks, contact_docs], ignore_index=True).reset_index(drop=True)

print(f"   - 페이지 청크: {len(page_chunks)}개")
print(f"   - 연락처 문서: {len(contact_docs)}개")
print(f"   - 총 검색 문서: {len(search_df)}개")

# BM25 및 임베딩 생성
print("🧠 BM25 및 임베딩 생성 중...")
tokenize_kor = get_tokenizer()
tokenized_corpus = [tokenize_kor(t) for t in search_df["text"].tolist()]
bm25 = BM25Okapi(tokenized_corpus)

model_name = "intfloat/multilingual-e5-base"
model = SentenceTransformer(model_name)

def embed_passages(texts):
    return model.encode(
        [f"passage: {t}" for t in texts],
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype(np.float32)

embeddings = embed_passages(search_df["text"].tolist())

# 아티팩트 저장
print("💾 아티팩트 저장 중...")
ART_DIR.mkdir(parents=True, exist_ok=True)

search_df.to_parquet(SEARCH_DF_PATH, index=False)
np.save(EMB_PATH, embeddings)

try:
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
except Exception:
    dump_json_gz(tokenized_corpus, str(TOK_PATH))

if not contact_docs.empty:
    contact_docs.to_csv(CONTACTS_CSV, index=False, encoding="utf-8-sig")

meta = {
    "built_at": datetime.now().isoformat(),
    "model_name": model_name,
    "n_entries": int(search_df.shape[0]),
    "n_pages": int(page_chunks.shape[0]),
    "n_contacts": int(contact_docs.shape[0]),
    "emb_dim": int(embeddings.shape[1]),
    "art_dir": str(ART_DIR),
    "paths": {
        "search_df": str(SEARCH_DF_PATH),
        "embeddings": str(EMB_PATH),
        "bm25": str(BM25_PATH),
        "tokens_backup": str(TOK_PATH),
        "contacts_csv": str(CONTACTS_CSV),
    }
}

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"✅ 학교 기본 정보 인덱스 구축 완료 → {ART_DIR}")
print(f"   - search_df : {SEARCH_DF_PATH.name}")
print(f"   - embeddings: {EMB_PATH.name}")
print(f"   - bm25      : {BM25_PATH.name}")
if not contact_docs.empty:
    print(f"   - contacts  : {CONTACTS_CSV.name}")