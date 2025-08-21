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

def _preclean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\u00A0", " ").replace("\u200B", "")
    s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212\uFE58\uFE63\uFF0D]", "-", s)
    return s


for c in ["doc_type", "title", "text", "unit", "phone", "email"]:
    if c not in search_df.columns:
        search_df[c] = ""

search_df["doc_type"] = search_df["doc_type"].astype(str).str.strip().str.lower()
search_df["title"]    = search_df["title"].fillna("").astype(str).str.strip()
search_df["text"]     = search_df["text"].fillna("").astype(str)
search_df["unit"]     = search_df["unit"].fillna("").astype(str)


search_df["phone"]    = search_df["phone"].fillna("").astype(str).str.replace(r"\s+", "", regex=True)
search_df["email"]    = search_df["email"].fillna("").astype(str).str.strip()


search_df["norm_text"]  = search_df["text"].apply(_preclean_text)
search_df["norm_title"] = search_df["title"].apply(_preclean_text)


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

PHONE_RE = re.compile(r'(0\d{1,2})[^\d]{0,5}(\d{3,4})[^\d]{0,5}(\d{4})')
EMAIL_RE = re.compile(r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})')
CONTACT_HINT_RE = re.compile(r'(연락처|전화|전화번호|이메일|email|tel)', re.IGNORECASE)
UNIT_TOK_RE = re.compile(r'[가-힣A-Za-z0-9]{2,}')

def _unit_pattern_from_query(q: str):
    toks = UNIT_TOK_RE.findall(q or '')
    prefer = [t for t in toks if re.search(r'(팀|과|센터|처|단|부|원|본부)$', t)]
    terms = prefer if prefer else toks[:3]
    if not terms:
        return None
    return re.compile("|".join(map(re.escape, terms)), re.IGNORECASE)


def _extract_from_text(text: str):
    if not isinstance(text, str) or not text:
        return None, None
    t = text.replace("\u00A0"," ").replace("\u200B"," ")
    t = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212\uFE58\uFE63\uFF0D]", "-", t)
    m = PHONE_RE.search(t)
    phone = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None
    m2 = EMAIL_RE.search(t)
    email = m2.group(0) if m2 else None
    return phone, email


def _get_text_field(row):
    for c in ("text","content","body","raw_text","raw_html","clean_text"):
        v = row.get(c)
        if isinstance(v, str) and v.strip():
            return v
    return ""


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


def hybrid_search(query, top_k=8, alpha=0.55, recency_weight=0.45, notice_boost=0.20, contact_boost=0.18):
    qv = embed_query(query)
    dense_raw = embeddings @ qv
    bm25_raw  = bm25.get_scores(tokenize_kor(query))
    base = alpha * _minmax(np.nan_to_num(dense_raw, 0, 0, 0)) \
         + (1 - alpha) * _minmax(np.nan_to_num(bm25_raw,  0, 0, 0))

    upd = search_df["updated_at"] if "updated_at" in search_df.columns else [None]*len(search_df)
    rec = _minmax(np.array([recency_score(ts) for ts in upd]))

    is_notice_q  = _contains_any(query, NOTICE_KWS)
    contact_like = _contains_any(query, CONTACT_KWS)

    eff_rec_w = recency_weight if is_notice_q else (0.0 if contact_like else 0.05)
    bonus = eff_rec_w * rec

    if is_notice_q and "notice_flag" in search_df.columns:
        nf = search_df["notice_flag"].to_numpy()
        bonus = bonus + notice_boost * nf

    title_col = search_df["norm_title"]
    text_col  = search_df["norm_text"]
    phone_ex = text_col.str.extract(PHONE_RE, expand=True)
    has_real_phone = phone_ex.notna().any(axis=1).astype(float).to_numpy()
    email_ex = text_col.str.extract(EMAIL_RE, expand=False)
    has_real_email = email_ex.notna().astype(float).to_numpy()

    if contact_like:
        doc_type  = search_df["doc_type"]
        phone_col = search_df["phone"]
        email_col = search_df["email"]

        is_contact = doc_type.eq("contact").astype(float).to_numpy()
        has_phone  = phone_col.str.len().gt(0).astype(float).to_numpy()
        has_email  = email_col.str.len().gt(0).astype(float).to_numpy()

        bonus += 2.00 * is_contact
        bonus += 1.20 * _minmax(has_real_phone)
        bonus += 0.60 * _minmax(has_real_email)

        if "notice_flag" in search_df.columns:
            nf = search_df["notice_flag"].to_numpy()
            bonus += (-0.80) * nf

        title_has_hint = title_col.str.contains(CONTACT_HINT_RE, na=False).astype(float).to_numpy()
        cand_mask = (is_contact > 0) | (has_real_phone > 0) | (has_real_email > 0)
        bonus += np.where(cand_mask, 0.0, -2.00)

        unit_pat = _unit_pattern_from_query(query)
        if unit_pat is not None:
            title_match = title_col.str.contains(unit_pat, na=False).astype(float).to_numpy()
            unit_col = search_df.get("unit", pd.Series([""]*len(search_df))).fillna("").astype(str)
            unit_match = unit_col.str.contains(unit_pat, na=False).astype(float).to_numpy()
            bonus += 0.80 * _minmax(title_match + unit_match)


    try:
        kw_signal = np.zeros(len(search_df), dtype=float)
        for rule in KW_RULES:
            if rule.get("_intent_pat") and rule["_intent_pat"].search(query or ""):
                has_title = title_col.str.contains(rule["_doc_pat"], na=False, regex=True).astype(float) if rule["_doc_pat"] else 0.0
                has_text  = text_col.str.contains(rule["_doc_pat"],  na=False, regex=True).astype(float) if rule["_doc_pat"] else 0.0
                kw_signal += rule["rule_w"] * (rule["title_w"] * has_title + rule["text_w"] * has_text)
        if np.any(kw_signal):
            if contact_like:
                gate = np.clip((has_real_phone > 0).astype(float) + (search_df["doc_type"].eq("contact")).astype(float), 0, 1)
                kw_signal = kw_signal * (0.25 + 0.75 * gate)
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
    out = out.sort_values(sort_cols, ascending=[False]*len(sort_cols), na_position="last")

    if contact_like and not (out["doc_type"].eq("contact").any()):
        title_has_hint = out["title"].str.contains(CONTACT_HINT_RE, na=False).astype(int)
        phone_ex2 = out["text"].str.extract(PHONE_RE, expand=True)
        has_real_phone2 = phone_ex2.notna().any(axis=1).astype(int)
        out = out.sort_values(
            ["doc_type","title_has_hint","has_real_phone","score"],
            ascending=[True, False, False, False]
        ).assign(
            title_has_hint=title_has_hint,
            has_real_phone=has_real_phone2
        )

    return out.head(top_k).reset_index(drop=True)


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

    if contact_like and top_contacts.empty:
        hits = hybrid_search(query, top_k=max(top_k, 25), contact_boost=1.2)
        doc_type_col = hits["doc_type"] if "doc_type" in hits.columns else pd.Series([""]*len(hits))
        top_contacts = hits[doc_type_col.eq("contact")]

    if contact_like and not top_contacts.empty:
        lines, srcs = [], []
        for _, r in top_contacts.head(3).iterrows():
            unit = r.get("unit") or r.get("title") or ""
            p = (r.get("phone") or "").strip()
            e = (r.get("email") or "").strip()

            if (not p or p.lower()=="nan") or (not e or e.lower()=="nan"):
                tp, te = _extract_from_text(_get_text_field(r))
                if not p and tp: p = tp
                if not e and te: e = te

            parts = []
            parts.append(f"전화 {p}" if p else "전화 없음")
            parts.append(f"이메일 {e}" if e else "이메일 없음")
            lines.append(f"- {unit}: " + " / ".join(parts))
            srcs.append({"title": r.get("title",""), "url": r.get("url","")})
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