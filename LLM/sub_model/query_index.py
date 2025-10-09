import os, re, json, pickle, numpy as np, pandas as pd
from pathlib import Path
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from LLM.sub_model.index_utils import get_tokenizer, load_json_gz  # 서버용
# from index_utils import get_tokenizer, load_json_gz
from dotenv import load_dotenv

load_dotenv()

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent.parent

GRAD_KWS = ["졸업", "졸업학점", "졸업 학점", "졸업요건", "졸업요구학점", "이수학점", "졸업 이수학점"]
UNIT_SUFFIX_RE = re.compile(r"(학부|학과|팀|센터|처|단|부|원|본부)$")
OTHER_UNIT_TOKEN_RE = re.compile(r"([가-힣A-Za-z0-9]{2,}(?:학부|학과|팀|센터|처|단|부|원|본부))")
FAX_RE = re.compile(r"(?:팩스|FAX)", re.I)
GENERIC_LABEL_RE = re.compile(r"(?:대표|교무|입학|장학|취업|총무|홍보|학생상담|콜센터)")

# === 추가: 직원/부서 통합 전화안내(교직원검색) 우선 적용용 패턴 ===
STAFF_TITLE_RE = re.compile(r"(?:교직원\s*검색|전화번호\s*안내)", re.I)
STAFF_URL_RE   = re.compile(r"/dmu/4408/subview\.do$", re.I)

_OLD_NEW_NUM = r"(\d{2,3})"
_BYPYO1_RE   = r"\[?\s*별표\s*1\s*\]?"
PHONE_RE = re.compile(r'(0\d{1,2})[^\d]{0,4}(\d{3,4})[^\d]{0,4}(\d{4})')
EMAIL_RE = re.compile(r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})')
_TOTAL_PATS = [
    re.compile(r"(?:총|전체)\s*(?:졸업|이수)\s*(?:요구)?\s*학점\s*[:：]?\s*(\d{3})\s*학점"),
    re.compile(r"졸업\s*(?:이수)?\s*학점\s*[:：]?\s*(\d{3})\s*학점"),
    re.compile(r"졸업\s*요구\s*학점\s*[:：]?\s*(\d{3})\s*학점"),
]
_PART_PATS = {
    "전공": re.compile(r"전공[^0-9]{0,8}(\d{2,3})\s*학점"),
    "교양": re.compile(r"(?:교양|공통교양)[^0-9]{0,8}(\d{2,3})\s*학점"),
    "자유선택": re.compile(r"(?:자유선택|자율선택)[^0-9]{0,8}(\d{2,3})\s*학점"),
}
NEAR_NUM_RE = re.compile(r"(\d{3})\s*학점")

def _canonical_unit_from_query(q: str) -> str | None:
    toks = re.findall(r"[가-힣A-Za-z0-9]{2,}", q or "")
    units = [t for t in toks if UNIT_SUFFIX_RE.search(t)]
    return max(units, key=len) if units else None

def _preclean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\u00A0", " ").replace("\u200B", "")
    s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212\uFE58\uFE63\uFF0D]", "-", s)
    return s

def _extract_phones_from_text(text: str):
    if not isinstance(text, str) or not text:
        return []
    t = _preclean_text(text)
    phones = ["-".join(m) for m in PHONE_RE.findall(t)]
    return list(dict.fromkeys(phones))

def _extract_contact_from_row(row, prefer_terms=None, target_unit=None):
    p = (row.get("phone") or "").strip()
    e = (row.get("email") or "").strip()

    texts = []
    base_text = (row.get("text") or row.get("content") or "")
    if base_text:
        texts.append(base_text)

    if (not p) or (not e):
        mask = (search_df["url"].eq(row.get("url","")) &
                search_df["doc_type"].eq(row.get("doc_type","")))
        sibs = search_df.loc[mask, "text"].dropna().astype(str)
        if not sibs.empty:
            texts.append("\n".join(sibs.tolist()))

    blob = "\n".join(texts)

    phones = _extract_phones_from_text(blob) if (not p) else [p]
    emails = [e] if e else EMAIL_RE.findall(_preclean_text(blob))
    if not phones:
        return [], emails

    ranked = []
    row_unit = (row.get("unit") or row.get("title") or "").strip()

    for ph in phones:
        m = re.search(re.escape(ph).replace(r'\-', '[-–—]?'), blob)
        score = 0.0
        if m:
            L = max(0, m.start() - 40)
            R = min(len(blob), m.end() + 40)
            ctx = blob[L:R]
            ctx_l = ctx.lower()

            if target_unit and target_unit in ctx:
                score += 2.5
            if target_unit and row_unit and (target_unit in row_unit):
                score += 0.6
            if prefer_terms and any(t.lower() in ctx_l for t in [*prefer_terms, "전화", "연락처"]):
                score += 0.5
            if FAX_RE.search(ctx): score -= 3.0
            if GENERIC_LABEL_RE.search(ctx): score -= 0.7
            if target_unit:
                others = [u for u in set(OTHER_UNIT_TOKEN_RE.findall(ctx)) if u != target_unit]
                if others: score -= 1.2

        ranked.append((score, ph))

    ranked.sort(reverse=True)
    phones = [ph for _, ph in ranked]
    return phones, emails

def _extract_grad_credits_from_text(text: str):
    t = _preclean_text(text)

    total = None
    for pat in _TOTAL_PATS:
        m = pat.search(t)
        if m:
            total = int(m.group(1))
            break

    parts = {}
    for name, pat in _PART_PATS.items():
        m = pat.search(t)
        if m:
            try:
                parts[name] = int(m.group(m.lastindex or 1))
            except Exception:
                pass

    if total is None:
        candidates = []
        for m in NEAR_NUM_RE.finditer(t):
            val = int(m.group(1))
            L = max(0, m.start() - 20)
            R = min(len(t), m.end() + 20)
            ctx = t[L:R]
            if re.search(r"(졸업|총|전체)", ctx) and not re.search(r"(전공|교양|자유선택|자율선택|핵심|기초)", ctx):
                candidates.append(val)
        good = [v for v in candidates if 100 <= v <= 200]
        total = max(good) if good else (max(candidates) if candidates else None)
    return total, parts

def _looks_like_grad_query(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in q or k in ql for k in GRAD_KWS)

def _extract_3yr_from_tables(text: str):
    t = _preclean_text(text)

    def _find_block(after_kw: str, window: int = 1200) -> str:
        m = re.search(after_kw, t)
        if not m:
            return ""
        return t[m.end(): m.end() + window]

    out = {}
    b1 = _find_block(r"졸업이수\s*학점")
    if b1:
        m = re.search(rf"3\s*년제[^\d]{{0,30}}{_OLD_NEW_NUM}[^\d]{{0,15}}{_OLD_NEW_NUM}", b1)
        if m:
            old_v, new_v = int(m.group(1)), int(m.group(2))
            out["grad_total"] = {"old": old_v, "new": new_v}

    b2 = _find_block(r"전공최저이수\s*학점")
    if b2:
        m = re.search(
            rf"3\s*년제[^\d\[]{{0,30}}(({_BYPYO1_RE})|{_OLD_NEW_NUM})[^\d\[]{{0,15}}{_OLD_NEW_NUM}",
            b2
        )
        if m:
            old_raw = m.group(2) or m.group(3)
            new_v   = int(m.group(4))
            if old_raw and re.search(_BYPYO1_RE, old_raw):
                old_v = "별표1"
            else:
                old_v = int(old_raw) if old_raw else None
            out["major_min"] = {"old": old_v, "new": new_v}
    return out

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
NOTICE_KWS       = load_list_from_txt(env_path("NOTICE_KWS_PATH"))
CONTACT_KWS      = load_list_from_txt(env_path("CONTACT_KWS_PATH"))

embeddings = np.load(EMB_PATH, mmap_mode="r").astype(np.float32)
row_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
row_norms[row_norms == 0] = 1.0
embeddings = embeddings / row_norms

UNIT_TOK_RE = re.compile(r"[가-힣A-Za-z0-9]{2,}")

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
    if "url" in out.columns:
        rep_idx = out.groupby(["url","doc_type"], dropna=False)["score"].idxmax()
        out = out.loc[rep_idx]
    out = out.sort_values(["score","updated_at"], ascending=[False, False], na_position="last").head(top_k)
    return out.reset_index(drop=True)

def build_answer(query, top_k=6):
    contact_like = any(k in query for k in CONTACT_KWS)
    notice_like = any(k in query for k in NOTICE_KWS)

    if contact_like:
        hits = hybrid_search(query, top_k=max(top_k, 30))

        toks = re.findall(r"[가-힣A-Za-z0-9]{2,}", query or "")
        prefer = [t for t in toks if re.search(r"(학부|과|팀|센터|처|단|부|원|본부)$", t)]
        terms = prefer if prefer else toks[:3]
        target_unit = _canonical_unit_from_query(query)

        # === 추가: 교직원검색(4408) 페이지를 후보에 반드시 포함
        staff_mask = (search_df["url"].astype(str).str.contains(STAFF_URL_RE) |
                      search_df["title"].astype(str).str.contains(STAFF_TITLE_RE))
        staff_rows = search_df.loc[staff_mask, ["doc_type","title","text","unit","url","phone","email"]].copy()
        if not staff_rows.empty:
            hits = pd.concat([hits, staff_rows], ignore_index=True)
            hits = hits.drop_duplicates(subset=["url","doc_type"], keep="first")

        # === 재정렬 가중치: 단위매칭/교직원검색/연락처문서/학부홈 가산, 게시판 감점
        urls   = hits["url"].fillna("")
        titles = hits["title"].fillna("")
        units  = hits["unit"].fillna("")

        unit_match = pd.Series([False]*len(hits))
        if target_unit:
            unit_match = (units.str.contains(re.escape(target_unit))) | (titles.str.contains(re.escape(target_unit)))
        unit_match = unit_match.astype(int)

        is_staff    = (urls.str.contains(STAFF_URL_RE) | titles.str.contains(STAFF_TITLE_RE)).astype(int)
        is_contact  = hits["doc_type"].eq("contact").astype(int)
        is_dept_home= urls.str.contains(r"/dmu/\d{4}/subview\.do$", case=False, regex=True).astype(int)
        is_board    = urls.str.contains(r"/bbs/|artclView\.do", case=False, regex=True).astype(int)

        _prio = 2.2*unit_match + 1.6*is_staff + 1.3*is_contact + 0.7*is_dept_home - 0.8*is_board
        hits = hits.assign(_prio=_prio).sort_values(["_prio","score"], ascending=[False, False])

        # === 번호 추출
        lines, seen = [], set()
        for _, r in hits.iterrows():
            phones, emails = _extract_contact_from_row(r, prefer_terms=terms, target_unit=target_unit)
            label = (r.get("unit") or r.get("title") or "").strip() or "연락처"
            if phones:
                picked = [ph for ph in phones if ph not in seen][:2]
                for ph in picked:
                    seen.add(ph)
                if picked:
                    lines.append(f"- {label}: 전화 {', '.join(picked)} (출처: {r.get('url','')})")
            elif emails:
                lines.append(f"- {label}: 이메일 {', '.join(dict.fromkeys(emails))} (출처: {r.get('url','')})")
            if len(lines) >= 4:
                break

        if not lines:
            picks = hits.head(top_k)[["title","url"]].fillna("")
            lines = [f"- {rv['title']}: {rv['url']}" for _, rv in picks.iterrows()]
        return {"answer": "\n".join(lines)}

    if _looks_like_grad_query(query):
        three_year_like = bool(re.search(r"\b3\s*년제\b", query))

        hits = hybrid_search(query, top_k=max(top_k, 25))
        urls = hits["url"].fillna("")
        home_like = urls.str.contains(r"/subview\.do|/intro|/dmu/\d+/subview", case=False, regex=True)
        bbs_like  = urls.str.contains(r"/bbs/|artclView\.do", case=False, regex=True)
        is_page   = hits["doc_type"].eq("page")
        rr = (1.20 * home_like.astype(int) + 1.00 * is_page.astype(int) - 0.80 * bbs_like.astype(int))
        hits = hits.assign(_rr=rr).sort_values(["_rr", "score"], ascending=[False, False])

        if three_year_like:
            for _, r in hits.iterrows():
                text = (r.get("text") or r.get("content") or "")
                tab = _extract_3yr_from_tables(text)
                if tab.get("grad_total") or tab.get("major_min"):
                    lines = []
                    gt = tab.get("grad_total")
                    mm = tab.get("major_min")
                    if gt:
                        lines.append(f"3년제 졸업이수 학점: 2022년 2·8월 대상자 {gt['old']}학점 / 2023년 2월 이후 {gt['new']}학점")
                    if mm:
                        old_txt = mm['old'] if isinstance(mm['old'], str) else (f"{mm['old']}학점" if mm['old'] is not None else "표기 없음")
                        lines.append(f"3년제 전공최저이수 학점: 2022년 2·8월 대상자 {old_txt} / 2023년 2월 이후 {mm['new']}학점")
                    lines.append(f"(출처: {r.get('title','').strip()} · {r.get('url','')})")
                    return {"answer": "\n".join(lines)}

        for _, r in hits.iterrows():
            text = (r.get("text") or r.get("content") or "")
            total, parts = _extract_grad_credits_from_text(text)

            if not (total or parts):
                if "doc_id" in search_df.columns and pd.notna(r.get("doc_id", np.nan)):
                    mask = search_df["doc_id"].eq(r["doc_id"])
                else:
                    mask = search_df["url"].eq(r.get("url","")) & search_df["doc_type"].eq(r.get("doc_type",""))
                sibs = search_df.loc[mask, "text"].dropna().astype(str)
                if not sibs.empty:
                    blob = "\n".join(sibs.tolist())
                    total, parts = _extract_grad_credits_from_text(blob)

            if total or parts:
                lines = []
                if total:
                    lines.append(f"총 졸업학점: **{total}학점**")
                if parts:
                    seg = " / ".join([f"{k} {v}학점" for k, v in parts.items()])
                    lines.append(f"세부: {seg}")
                lines.append(f"(출처: {r.get('title','').strip()} · {r.get('url','')})")
                lines.append("※ 학과·학번(교육과정 연도)에 따라 상이할 수 있으니, 출처 페이지 기준으로 확인하세요.")
                return {"answer": "\n".join(lines)}

        hits = hybrid_search(query, top_k=top_k)
        picks = hits.head(top_k)[["title", "url"]].fillna("")
        lines = [f"- {r['title']}: {r['url']}" for _, r in picks.iterrows()]
        return {"answer": "\n".join(lines)}

    hits = hybrid_search(query, top_k=max(top_k, 12))

    if not contact_like and not notice_like and not hits.empty:
        urls = hits["url"].fillna("")
        home_like = urls.str.contains(r"/subview\.do|/intro|/dmu/\d+/subview", case=False, regex=True)
        bbs_like  = urls.str.contains(r"/bbs/|artclView\.do", case=False, regex=True)
        is_page   = hits["doc_type"].eq("page")
        def _n(s: str) -> str: return re.sub(r"\s+", "", (s or "").lower())
        qn = _n(query)
        title_match = hits["title"].fillna("").map(lambda s: qn in _n(s))
        unit_match  = hits.get("unit", pd.Series([""]*len(hits))).fillna("").map(lambda s: qn in _n(s))
        rr = (
            1.20 * home_like.astype(int)
          + 1.00 * is_page.astype(int)
          + 0.60 * (title_match.astype(int) | unit_match.astype(int))
          - 0.80 * bbs_like.astype(int)
        )
        hits = hits.assign(_rr=rr).sort_values(["_rr", "score"], ascending=[False, False])

    picks = hits.head(top_k)[["title", "url"]].fillna("")
    lines = [f"- {r['title']}: {r['url']}" for _, r in picks.iterrows()]
    return {"answer": "\n".join(lines)}

if __name__ == "__main__":
    q1 = "학생성공지원팀 담당자 전화번호 알려줘"
    q2 = "컴퓨터공학부 담당자 전화번호 알려줘"
    q3 = "경영학부 담당자 연락처"
    q4 = "2년제 졸업조건"
    for q in [q1, q2, q3, q4]:
        res = build_answer(q, top_k=8)
        print("Q:", q)
        print("A:\n", res["answer"])