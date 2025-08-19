import os, re, json, pickle, numpy as np, pandas as pd
from pathlib import Path
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from index_utils import get_tokenizer, load_json_gz
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta


load_dotenv()
KST = timezone(timedelta(hours=9))

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


DATA_JSON = env_path("DATA_JSON")
DEPT_DIR  = env_path("DEPT_DIR")
ART_DIR = env_path("ART_DIR")
SEARCH_DF_PATH = env_path("SEARCH_DF_PATH")
EMB_PATH       = env_path("EMB_PATH")
BM25_PATH      = env_path("BM25_PATH")
TOK_PATH       = env_path("TOK_PATH")
CONTACTS_CSV   = env_path("CONTACTS_CSV")
META_PATH      = env_path("META_PATH")
NOTICE_KWS_PATH  = env_path("NOTICE_KWS_PATH")
CONTACT_KWS_PATH = env_path("CONTACT_KWS_PATH")


search_df  = pd.read_parquet(SEARCH_DF_PATH)
embeddings = np.load(EMB_PATH, mmap_mode="r").astype(np.float32)


row_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
row_norms[row_norms == 0] = 1.0
embeddings = embeddings / row_norms

if embeddings.shape[0] != len(search_df):
    raise ValueError(f"Row mismatch: embeddings={embeddings.shape[0]} vs search_df={len(search_df)}")

tokenize_kor = get_tokenizer()

if BM25_PATH.exists():
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
else:
    if Path(TOK_PATH).exists():
        toks = load_json_gz(str(TOK_PATH))
    else:
        toks = [tokenize_kor(t) for t in search_df["text"].astype(str).tolist()]
    bm25 = BM25Okapi(toks)

model_name = "intfloat/multilingual-e5-base"
model = SentenceTransformer(model_name)

@lru_cache(maxsize=512)
def embed_query(q: str):
    return model.encode([f"query: {q}"], normalize_embeddings=True)[0].astype(np.float32)

def load_list_from_txt(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

try:
    NOTICE_KWS  = load_list_from_txt(NOTICE_KWS_PATH)
except FileNotFoundError:
    NOTICE_KWS = []
try:
    CONTACT_KWS = load_list_from_txt(CONTACT_KWS_PATH)
except FileNotFoundError:
    CONTACT_KWS = []


def _minmax(x):
    x = np.asarray(x, float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo + 1e-9)


def _parse_dt(ts):
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return None
    if isinstance(ts, pd.Timestamp):
        return ts.to_pydatetime().replace(tzinfo=ts.tzinfo or KST)
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=KST)
    s = str(ts).strip()
    for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(s[:10], fmt).replace(tzinfo=KST)
        except Exception:
            pass
    m = re.search(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", s)
    if m:
        y, mo, d = map(int, m.groups())
        return datetime(y, mo, d, tzinfo=KST)
    return None


def recency_score(ts, half_life_days=30, now=None):
    if now is None:
        now = datetime.now(KST)
    dt = _parse_dt(ts)
    if dt is None:
        return 0.0
    days = max((now - dt).days, 0)
    return float(np.exp(-np.log(2) * days / half_life_days))


def _contains_any(q: str, kws: list) -> bool:
    if not kws:
        return False
    ql = (q or "").lower()
    return any((k in q) or (k.lower() in ql) for k in kws)


def _compile_or(terms):
    if not terms:
        return None
    safe = [re.escape(t) for t in terms if t]
    return re.compile("|".join(safe), flags=re.IGNORECASE)

KW_RULES = [
    {
        "name": "contact",
        "intent_terms": ["전화", "연락처", "번호", "문의", "담당자", "tel", "이메일", "email"],
        "doc_terms":    ["전화", "연락처", "대표번호", "문의", "담당자", "tel", "이메일", "email", "@", "전화번호"],
        "title_w": 1.0,
        "text_w":  0.5,
        "rule_w":  0.60,
    },
    {
        "name": "intro",
        "intent_terms": ["대학소개", "비전", "전략체계", "교육이념", "미션"],
        "doc_terms":    ["대학소개", "비전", "전략체계", "교육이념", "미션", "핵심가치", "전략"],
        "title_w": 1.0,
        "text_w":  0.4,
        "rule_w":  0.45,
    },
]

for _r in KW_RULES:
    _r["_doc_pat"] = _compile_or(_r["doc_terms"])
    _r["_intent_pat"] = _compile_or(_r["intent_terms"])


def hybrid_search(query, top_k=8, alpha=0.6, recency_weight=0.45, notice_boost=0.20, contact_boost=0.18):
    qv = embed_query(query)

    dense_raw = embeddings @ qv
    bm25_raw  = bm25.get_scores(tokenize_kor(query))

    base = alpha * _minmax(np.nan_to_num(dense_raw, nan=0.0, posinf=0.0, neginf=0.0)) \
         + (1 - alpha) * _minmax(np.nan_to_num(bm25_raw,  nan=0.0, posinf=0.0, neginf=0.0))

    upd = search_df["updated_at"] if "updated_at" in search_df.columns else [None]*len(search_df)
    rec = _minmax(np.array([recency_score(ts) for ts in upd]))
    bonus = recency_weight * rec

    if _contains_any(query, NOTICE_KWS):
        nf = search_df["notice_flag"].to_numpy() if "notice_flag" in search_df.columns else np.zeros(len(search_df))
        bonus = bonus + notice_boost * nf

    if _contains_any(query, CONTACT_KWS):
        doc_type  = search_df["doc_type"] if "doc_type" in search_df.columns else pd.Series([""]*len(search_df))
        phone_col = search_df["phone"]    if "phone"    in search_df.columns else pd.Series([""]*len(search_df))
        email_col = search_df["email"]    if "email"    in search_df.columns else pd.Series([""]*len(search_df))

        is_contact = doc_type.eq("contact").astype(float).to_numpy()
        has_phone  = phone_col.fillna("").astype(str).str.len().gt(0).astype(float).to_numpy()
        has_email  = email_col.fillna("").astype(str).str.len().gt(0).astype(float).to_numpy()

        bonus = bonus + contact_boost * (is_contact + 0.5*has_phone + 0.5*has_email)

    try:
        title_col = search_df.get("title", pd.Series([""] * len(search_df))).fillna("").astype(str)
        text_col  = search_df.get("text",  pd.Series([""] * len(search_df))).fillna("").astype(str)
        kw_signal = np.zeros(len(search_df), dtype=float)

        for rule in KW_RULES:
            if rule.get("_intent_pat") and rule["_intent_pat"].search(query or ""):
                has_title = title_col.str.contains(rule["_doc_pat"], na=False, regex=True).astype(float) if rule["_doc_pat"] else 0.0
                has_text  = text_col.str.contains(rule["_doc_pat"],  na=False, regex=True).astype(float) if rule["_doc_pat"] else 0.0
                kw_signal += rule["rule_w"] * (rule["title_w"] * has_title + rule["text_w"] * has_text)

        if np.any(kw_signal):
            bonus += 0.35 * _minmax(kw_signal)
    except Exception:
        pass

    scores = np.nan_to_num(base + bonus, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

    out = search_df.copy()
    out["score"] = np.nan_to_num(pd.to_numeric(scores, errors="coerce"), nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

    if "url" in out.columns:
        group_cols = ["url"] + (["doc_type"] if "doc_type" in out.columns else [])
        rep_idx = out.groupby(group_cols, dropna=False)["score"].idxmax()
        rep_idx = rep_idx.dropna().astype(int)
        out = out.loc[rep_idx]

    sort_cols = ["score"] + (["updated_at"] if "updated_at" in out.columns else [])
    out = out.sort_values(sort_cols, ascending=[False]*len(sort_cols), na_position="last").head(top_k)
    return out.reset_index(drop=True)


def _snippet(text, q, max_chars=220):
    if not isinstance(text, str):
        return ""
    t = " ".join(text.split())
    if len(t) <= max_chars:
        return t
    toks = [x for x in re.findall(r"[가-힣A-Za-z0-9]+", q) if len(x) >= 2]
    pos_candidates = [t.lower().find(x.lower()) for x in toks]
    pos_candidates = [p for p in pos_candidates if p >= 0]
    pos = min(pos_candidates) if pos_candidates else 0
    start = max(pos - 60, 0); end = min(start + max_chars, len(t))
    return (("…" if start>0 else "") + t[start:end] + ("…" if end<len(t) else ""))


def build_answer(query, top_k=6):
    hits = hybrid_search(query, top_k=top_k)
    contact_like = _contains_any(query, CONTACT_KWS)
    doc_type_col = hits["doc_type"] if "doc_type" in hits.columns else pd.Series([""]*len(hits))
    top_contacts = hits[doc_type_col.eq("contact")]

    if contact_like and not top_contacts.empty:
        lines = []
        for _, r in top_contacts.head(3).iterrows():
            unit = r.get("unit") or r.get("title") or ""
            p = (r.get("phone") or "").strip()
            e = (r.get("email") or "").strip()
            parts = []
            if p and p.lower() != "nan": parts.append(f"전화 {p}")
            if e and e.lower() != "nan": parts.append(f"이메일 {e}")
            joined = " / ".join(parts) if parts else "연락처 정보가 텍스트에 포함됨"
            src_title = r.get("title") or ""
            lines.append(f"- {unit}: {joined}  (출처: {src_title})")

        others = hits[~doc_type_col.eq("contact")].head(2)
        srcs = (
            [{"title": rr.get("title",""), "url": rr.get("url","")} for _, rr in top_contacts.head(3).iterrows()] +
            [{"title": rr.get("title",""), "url": rr.get("url","")} for _, rr in others.iterrows()]
        )
        return {"answer": "\n".join(lines), "sources": srcs}

    topn = hits.head(3)
    lines = [f"- {r.get('title','')}: {_snippet(r.get('text',''), query)}" for _, r in topn.iterrows()]
    srcs  = [{"title": r.get("title",""), "url": r.get("url","")} for _, r in topn.iterrows()]
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