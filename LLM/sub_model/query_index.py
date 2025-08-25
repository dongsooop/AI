import os, re, json, pickle, numpy as np, pandas as pd
from pathlib import Path
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from index_utils import get_tokenizer, load_json_gz
from dotenv import load_dotenv


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

def load_list_from_txt(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


DATA_JSON        = env_path("DATA_JSON")
DEPT_DIR         = env_path("DEPT_DIR")
ART_DIR          = env_path("ART_DIR")
SEARCH_DF_PATH   = env_path("SEARCH_DF_PATH")
EMB_PATH         = env_path("EMB_PATH")
BM25_PATH        = env_path("BM25_PATH")
TOK_PATH         = env_path("TOK_PATH")
CONTACTS_CSV     = env_path("CONTACTS_CSV")
META_PATH        = env_path("META_PATH")
NOTICE_KWS  = load_list_from_txt(env_path("NOTICE_KWS_PATH"))
CONTACT_KWS = load_list_from_txt(env_path("CONTACT_KWS_PATH"))

embeddings = np.load(EMB_PATH, mmap_mode="r").astype(np.float32)
row_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
row_norms[row_norms == 0] = 1.0
embeddings = embeddings / row_norms

UNIT_TOK_RE = re.compile(r"[가-힣A-Za-z0-9]{2,}")

def _unit_terms(q: str):
    toks = UNIT_TOK_RE.findall(q or "")
    prefer = [t for t in toks if re.search(r"(팀|과|센터|처|단|부|원|본부)$", t)]
    return prefer if prefer else toks[:3]


model_name = "intfloat/multilingual-e5-base"
model = SentenceTransformer(model_name)

search_df = pd.read_parquet(SEARCH_DF_PATH)
for c in ["doc_type", "title", "text", "unit", "phone", "email", "updated_at", "notice_flag", "url"]:
    if c not in search_df.columns:
        search_df[c] = np.nan

search_df["doc_type"]   = search_df["doc_type"].fillna("").astype(str).str.strip().str.lower()
search_df["title"]      = search_df["title"].fillna("").astype(str).str.strip()
search_df["text"]       = search_df["text"].fillna("").astype(str)
search_df["unit"]       = search_df["unit"].fillna("").astype(str)
search_df["phone"]      = search_df["phone"].fillna("").astype(str).str.replace(r"\s+", "", regex=True)
search_df["email"]      = search_df["email"].fillna("").astype(str).str.strip()
search_df["notice_flag"]= pd.to_numeric(search_df["notice_flag"], errors="coerce").fillna(0).astype(int)

tokenize_kor = get_tokenizer()

if BM25_PATH.exists():
    with open(BM25_PATH, 'rb') as f:
        bm25 = pickle.load(f)
else:
    if TOK_PATH.exists():
        tokenized_courpus = load_json_gz(str(TOK_PATH))
    else:
        tokenized_courpus = [tokenize_kor(t) for t in search_df["text"].astype(str).tolist()]
    bm25 = BM25Okapi(tokenized_courpus)


def _parse_dt(ts):
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return pd.NaT
    try:
        return pd.to_datetime(ts)
    except Exception:
        s = str(ts)
        for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d", "%Y%m%d"):
            try:
                return pd.to_datetime(s[:10], format=fmt)
            except Exception:
                pass
        m = re.search(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", s)
        if m:
            y, mo, d = map(int, m.groups())
            return pd.Timestamp(year=y, month=mo, day=d)
        return pd.NaT

search_df["updated_at"] = search_df["updated_at"].apply(_parse_dt)


@lru_cache(maxsize=512)
def embed_query(q: str):
    return model.encode([f"query: {q}"], normalize_embeddings=True)[0].astype(np.float32)


def _minmax(x):
    x = np.asarray(x, float)
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo + 1e-9)


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


def build_answer(query, top_k=6):
    contact_like = any(k in query for k in CONTACT_KWS)

    if contact_like:
        qv = embed_query(query)
        dense_all = embeddings @ qv
        mask = (search_df["doc_type"] == "contact").to_numpy()
        idx = np.where(mask)[0]

        if idx.size > 0:
            scores = dense_all[idx].copy()
            terms = _unit_terms(query)
            if terms:
                pat = re.compile("|".join(map(re.escape, terms)), flags=re.IGNORECASE)
                hit_title = search_df.loc[idx, "title"].str.contains(pat, na=False).astype(float).to_numpy()
                hit_unit  = search_df.loc[idx, "unit"].str.contains(pat,  na=False).astype(float).to_numpy()
                scores += 0.20 * (hit_title + hit_unit)

            order = np.argsort(-scores)
            take = idx[order][:top_k]
            picks = search_df.loc[take, ["title", "unit", "url"]].fillna("")

            lines = []
            for _, r in picks.iterrows():
                label = (r["unit"] or r["title"]).strip() or r["title"]
                lines.append(f"- {label}: {r['url']}")
            return {"answer": "\n".join(lines)}

    hits = hybrid_search(query, top_k=top_k)
    picks = hits[["title", "url"]].fillna("")
    lines = [f"- {r['title']}: {r['url']}" for _, r in picks.iterrows()]
    return {"answer": "\n".join(lines)}


if __name__ == "__main__":
    q1 = "학생성공지원팀 담당자 전화번호 알려줘"
    q2 = "컴퓨터공학부 담당자 전화번호 알려줘"
    q3 = "총학생회"
    for q in [q1, q2, q3]:
        res = build_answer(q, top_k=8)
        print("Q:", q)
        print("A:\n", res["answer"])