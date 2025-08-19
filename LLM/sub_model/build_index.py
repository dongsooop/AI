import os, json, re, hashlib, pickle, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

from index_utils import (
    normalize_text, extract_date_from_content, chunk_text, get_tokenizer,
    extract_units_and_contacts, clean_name, compose_contact_passage
)

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent.parent

load_dotenv()


art_dir_env = os.environ.get("ART_DIR", "").strip()

if not art_dir_env:
    raise FileNotFoundError("ART_DIR environment empty.")

_art = Path(os.path.expandvars(art_dir_env))
ART_DIR = (_art if _art.is_absolute() else (ROOT_DIR / _art)).resolve()


def _resolve_env_path(var_name: str, default_filename: str, base_dir: Path = ART_DIR) -> Path:
    raw = os.environ.get(var_name, "").strip()
    if raw:
        s = os.path.expandvars(raw)
        s = s.replace("ART_DIR/", f"{base_dir}{os.sep}").replace("ART_DIR", str(base_dir))
        p = Path(s)
        if not p.is_absolute():
            p = (ROOT_DIR / p)
        return p.resolve()
    return (base_dir / default_filename).resolve()

SEARCH_DF_PATH = _resolve_env_path("SEARCH_DF_PATH", "search_df.parquet")
EMB_PATH       = _resolve_env_path("EMB_PATH",       "embeddings.npy")
BM25_PATH      = _resolve_env_path("BM25_PATH",      "bm25.pkl")
TOK_PATH       = _resolve_env_path("TOK_PATH",       "tokenized_corpus.json.gz")
CONTACTS_CSV   = _resolve_env_path("CONTACTS_CSV",   "contact_docs.csv")
META_PATH      = _resolve_env_path("META_PATH",      "meta.json")


for p in [SEARCH_DF_PATH, EMB_PATH, BM25_PATH, TOK_PATH, CONTACTS_CSV, META_PATH]:
    p.parent.mkdir(parents=True, exist_ok=True)

def _resolve_input(var_name: str, must_exist=True) -> Path | None:
    raw = os.environ.get(var_name, "").strip()
    if not raw:
        return None
    p = Path(os.path.expandvars(raw))
    if not p.is_absolute():
        p = (ROOT_DIR / p)
    p = p.resolve()
    if must_exist and not p.exists():
        raise FileNotFoundError(f"{var_name} not found: {p}")
    return p

DATA_JSON = _resolve_input("DATA_JSON", must_exist=True)
DEPT_DIR  = _resolve_input("DEPT_DIR",  must_exist=False)

if not DATA_JSON:
    raise FileNotFoundError("dmu_documents_cleaned.json 경로를 찾을 수 없습니다.")
if not DEPT_DIR:
    print("⚠️ 부서 공지 CSV 디렉토리를 찾지 못했습니다(DEPT_DIR).")



with open(DATA_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

df_json = pd.DataFrame(data).dropna(subset=["content"])
df_json["title"] = df_json["title"].fillna("")
df_json["url"] = df_json["url"].fillna("")
df_json["source"] = "main"
df_list = []

if DEPT_DIR and Path(DEPT_DIR).exists():
    for file in sorted(os.listdir(DEPT_DIR)):
        if not file.endswith("_notices.csv"): continue
        file_path = Path(DEPT_DIR) / file
        try:
            df_dept = pd.read_csv(file_path)
            df_dept["title"] = df_dept["제목"].fillna("")
            df_dept["url"] = df_dept["링크"].fillna("")
            df_dept["content"] = (
                "부서: " + df_dept["부서"].fillna("없음") + "\n"
                + "작성자: " + df_dept["작성자"].fillna("없음") + "\n"
                + "작성일: " + df_dept["작성일"].astype(str).fillna("없음")
            )
            df_dept["source"] = file.replace("_notices.csv", "")
            df_list.append(df_dept[["title","content","url","source"]])
        except Exception as e:
            print(f"⚠️ 파일 오류: {file_path} - {e}")


df_csv = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame(columns=["title","content","url","source"])
df = pd.concat([df_json[["title","content","url","source"]], df_csv], ignore_index=True)
df = df.dropna(subset=["title","content"]).reset_index(drop=True)


df["title"]     = df["title"].apply(normalize_text)
df["content"]   = df["content"].apply(normalize_text)
df["fulltext"]  = (df["title"] + " " + df["content"]).apply(normalize_text)
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


contact_cands = []
for _, r in df.iterrows():
    contact_cands.extend(extract_units_and_contacts(r))

contacts_raw = pd.DataFrame(contact_cands)


if contacts_raw.empty:
    contact_docs = pd.DataFrame(columns=list(page_chunks.columns))
else:
    contacts_raw["unit_norm"] = contacts_raw["unit"].apply(clean_name)
    cluster_map, reps = {}, []
    for name in contacts_raw["unit_norm"].unique():
        assigned = False

        for cid, rep in enumerate(reps):
            if SequenceMatcher(None, name, rep).ratio() >= 0.92:
                cluster_map[name] = cid; assigned = True; break
            
        if not assigned:
            cid = len(reps); reps.append(name); cluster_map[name] = cid
    contacts_raw["cluster_id"] = contacts_raw["unit_norm"].map(cluster_map.get)
    rep_map = {}

    for cid in set(cluster_map.values()):
        members = contacts_raw[contacts_raw["cluster_id"] == cid]
        rep_map[cid] = members["unit"].value_counts().idxmax()
    contacts_raw["unit_canonical"] = contacts_raw["cluster_id"].map(rep_map.get)
    contacts_df = (contacts_raw
        .sort_values(["cluster_id","url"])
        .drop_duplicates(subset=["cluster_id","phone","email"])
        .reset_index(drop=True))
    
    contact_rows = []
    for idx, r in contacts_df.iterrows():
        txt = compose_contact_passage(r["unit_canonical"], r["phone"], r["email"], r["title"], r["url"])
        contact_rows.append({
            "doc_id": -1, "chunk_id": f"c-{idx}",
            "title": f"{r['unit_canonical']} 연락처",
            "url": r["url"], "source": r["source"],
            "notice_flag": 0,
            "updated_at": r.get("updated_at", None),
            "text": txt,
            "doc_type": "contact",
            "unit": r["unit_canonical"], "phone": r["phone"], "email": r["email"]
        })
    contact_docs = pd.DataFrame(contact_rows)


search_df = pd.concat([page_chunks, contact_docs], ignore_index=True).reset_index(drop=True)

tokenize_kor = get_tokenizer()
tokenized_corpus = [tokenize_kor(t) for t in search_df["text"].tolist()]
bm25 = BM25Okapi(tokenized_corpus)

model_name = "intfloat/multilingual-e5-base"
model = SentenceTransformer(model_name)


def embed_passages(texts):
    return model.encode([f"passage: {t}" for t in texts],
                        normalize_embeddings=True, show_progress_bar=True).astype(np.float32)

embeddings = embed_passages(search_df["text"].tolist())


search_df.to_parquet(SEARCH_DF_PATH, index=False)
np.save(EMB_PATH, embeddings)


try:
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
except Exception:
    from index_utils import dump_json_gz
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