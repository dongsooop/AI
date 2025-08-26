import os, json, re, hashlib, pickle, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

from index_utils import (
    normalize_text, extract_date_from_content, chunk_text, get_tokenizer,
    extract_units_and_contacts, clean_name, compose_contact_passage,
    dump_json_gz
)

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
DEPT_DIR         = env_path("DEPT_DIR")
ART_DIR          = env_path("ART_DIR")
SEARCH_DF_PATH   = env_path("SEARCH_DF_PATH")
EMB_PATH         = env_path("EMB_PATH")
BM25_PATH        = env_path("BM25_PATH")
TOK_PATH         = env_path("TOK_PATH")
CONTACTS_CSV     = env_path("CONTACTS_CSV")
META_PATH        = env_path("META_PATH")
NOTICE_KWS_PATH  = env_path("NOTICE_KWS_PATH")
CONTACT_KWS_PATH = env_path("CONTACT_KWS_PATH")

if not DATA_JSON:
    raise FileNotFoundError("dmu_documents_cleaned.json 경로를 찾을 수 없습니다.")
if not DEPT_DIR:
    print("⚠️ 부서 공지 CSV 디렉토리를 찾지 못했습니다(DEPT_DIR).")
    
with open(DATA_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

df_json = pd.DataFrame(data).dropna(subset=["content"])
df_json["title"]  = df_json["title"].fillna("")
df_json["url"]    = df_json["url"].fillna("")
df_json["source"] = "main"
df_list = []

if DEPT_DIR and Path(DEPT_DIR).exists():
    for file in sorted(os.listdir(DEPT_DIR)):
        if not file.endswith("_notices.csv"):
            continue
        file_path = Path(DEPT_DIR) / file
        try:
            df_dept = pd.read_csv(file_path)
            df_dept["title"] = df_dept["제목"].fillna("")
            df_dept["url"]   = df_dept["링크"].fillna("")
            df_dept["content"] = (
                "부서: "   + df_dept["부서"].fillna("없음") + "\n" +
                "작성자: " + df_dept["작성자"].fillna("없음") + "\n" +
                "작성일: " + df_dept["작성일"].astype(str).fillna("없음")
            )
            df_dept["source"] = file.replace("_notices.csv", "")
            df_list.append(df_dept[["title", "content", "url", "source"]])
        except Exception as e:
            print(f"⚠️ 파일 오류: {file_path} - {e}")

df_csv = (
    pd.concat(df_list, ignore_index=True)
    if df_list else pd.DataFrame(columns=["title","content","url","source"])
)


df = pd.concat(
    [df_json[["title","content","url","source"]], df_csv],
    ignore_index=True
)
df = df.dropna(subset=["title","content"]).reset_index(drop=True)


df["title"]       = df["title"].apply(normalize_text)
df["content"]     = df["content"].apply(normalize_text)
df["fulltext"]    = (df["title"] + " " + df["content"]).apply(normalize_text)
df["content_hash"] = df["fulltext"].apply(lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest())
df = df.drop_duplicates(subset=["url"]).drop_duplicates(subset=["content_hash"]).reset_index(drop=True)


df["notice_flag"] = (df["source"] != "main").astype(int)
df["updated_at"] = df.apply(
    lambda r: extract_date_from_content(r["content"]) if r["notice_flag"]==1 else None, axis=1
)


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
            "notice_flag": r["notice_flag"],
            "updated_at": r["updated_at"],
            "text": ch,
            "doc_type": "page",
            "unit": "", "phone": "", "email": ""
        })
page_chunks = pd.DataFrame(rows).reset_index(drop=True)


DASH_CHARS = r"\-\u2010\u2011\u2012\u2013\u2014\u2212\uFE58\uFE63\uFF0D"
def _preclean_contact_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\u00A0", " ").replace("\u200B", "")
    s = re.sub(f"[{DASH_CHARS}]", "-", s)
    return s

contact_cands = []


for _, r in df_json.iterrows():
    row_like = pd.Series({
        "title":   r["title"],
        "content": _preclean_contact_text(r["content"]),
        "url":     r["url"],
        "source":  "main",
    })
    contact_cands.extend(extract_units_and_contacts(row_like))


if not df_csv.empty:
    for _, r in df_csv.iterrows():
        row_like = pd.Series({
            "title":   r["title"],
            "content": _preclean_contact_text(r["content"]),
            "url":     r["url"],
            "source":  r.get("source", "dept"),
        })
        contact_cands.extend(extract_units_and_contacts(row_like))

contacts_raw = pd.DataFrame(contact_cands)


if contacts_raw.empty:
    print("⚠️ 연락처 후보가 비어 있습니다. 추출 로직/입력 텍스트를 확인하세요.")
    contact_docs = pd.DataFrame()
else:
    print("contacts_raw rows:", len(contacts_raw))
    try:
        print(contacts_raw.head(10)[["unit","phone","email","title","url"]].to_string(index=False))
    except Exception:
        print(contacts_raw.head(10).to_string(index=False))


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
            "notice_flag": 0,
            "updated_at": r.get("updated_at", None),
            "text": txt,
            "doc_type": "contact",
            "unit": r.get("unit_canonical", r.get("unit","")),
            "phone": r.get("phone", ""),
            "email": r.get("email", "")
        })

    contact_docs = pd.DataFrame(contact_rows)
    try:
        print("contact_docs rows:", len(contact_docs))
        print(contact_docs.head(10)[["unit","phone","email","title"]].to_string(index=False))
    except Exception:
        pass


ALL_COLS = [
    "doc_id","chunk_id","title","url","source",
    "notice_flag","updated_at","text","doc_type",
    "unit","phone","email"
]


def _ensure_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    for c in ALL_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[ALL_COLS]
    df["updated_at"]  = pd.to_datetime(df["updated_at"], errors="coerce")
    df["doc_id"]      = pd.to_numeric(df["doc_id"], errors="coerce").fillna(-1).astype("int64")
    df["notice_flag"] = pd.to_numeric(df["notice_flag"], errors="coerce").fillna(0).astype("int64")
    for c in ["chunk_id","title","url","source","text","doc_type","unit","phone","email"]:
        df[c] = df[c].astype("object")
    return df


page_chunks  = _ensure_columns(page_chunks)
contact_docs = _ensure_columns(contact_docs)
search_df = pd.concat([page_chunks, contact_docs], ignore_index=True).reset_index(drop=True)

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

print(f"✅ Unified index built → {ART_DIR}")
print(f"   - search_df : {SEARCH_DF_PATH.name}")
print(f"   - embeddings: {EMB_PATH.name}")
print(f"   - bm25      : {BM25_PATH.name}")
if not contact_docs.empty:
    print(f"   - contacts  : {CONTACTS_CSV.name}")