import os, re, json, pickle, numpy as np, pandas as pd
from pathlib import Path
from functools import lru_cache
from LLM.patterns import (
    BOARD_URL_PATTERN,
    DASH_CHARS_PATTERN,
    DEPT_HOME_URL_PATTERN,
    EMAIL_RE,
    GENERIC_ORG_LABEL_RE,
    HANGUL_TOKEN_PATTERN,
    HOME_LIKE_URL_PATTERN,
    OTHER_UNIT_TOKEN_RE,
    PHONE_ANY_AREA_RE,
    STAFF_TITLE_PATTERN,
    STAFF_URL_PATTERN,
    UNIT_QUERY_SUFFIX_PATTERN,
    UNIT_SUFFIX_RE,
)
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from LLM.sub_model.index_utils import get_tokenizer, load_json_gz  # 서버용
from LLM.sub_model.query_index_schema import normalize_search_df_schema
# from index_utils import get_tokenizer, load_json_gz
from dotenv import load_dotenv

load_dotenv()

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent.parent

# 졸업학점 관련 키워드 복원 (학교 기본 정보 포함)
GRAD_KWS = ["졸업", "졸업학점", "졸업 학점", "졸업요건", "졸업요구학점", "이수학점", "졸업 이수학점"]
FAX_RE = re.compile(r"(?:팩스|FAX)", re.I)
GENERIC_LABEL_RE = GENERIC_ORG_LABEL_RE

# 졸업학점 관련 변수들 복원 (학교 기본 정보 포함)
_OLD_NEW_NUM = r"(\d{2,3})"
_BYPYO1_RE   = r"\[?\s*별표\s*1\s*\]?"
PHONE_RE = PHONE_ANY_AREA_RE
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
GRAD_TABLE_TOTAL_RE = re.compile(r"구분\s*졸업\s*학점(?P<body>.{0,350})")
YEAR_PROGRAM_CREDIT_RE = re.compile(r"([23])\s*년제\s*(\d{2,3})")

CONTACT_SEARCH_POOL_SIZE = 30
GRAD_SEARCH_POOL_SIZE = 25
GENERAL_SEARCH_POOL_SIZE = 12
DEFAULT_DENSE_WEIGHT = 0.6
CONTACT_DOC_BOOST = 0.18
PHONE_EMAIL_BONUS_RATIO = 0.5
UNIT_MATCH_BOOST = 2.2
STAFF_PAGE_BOOST = 1.6
CONTACT_DOC_RERANK_BOOST = 1.3
DEPT_HOME_BOOST = 0.7
BOARD_PAGE_PENALTY = 0.8
HOME_PAGE_RERANK_BOOST = 1.2
PAGE_DOC_RERANK_BOOST = 1.0
TITLE_UNIT_MATCH_BOOST = 0.6
ANSWER_TEXT_COL = "text_for_answer"
PRIVACY_QUERY_RE = re.compile(r"(개인정보|영상정보|처리방침|이메일\s*무단|이전방침|이용안내)")
POLICY_QUERY_RE = re.compile(r"(휴학|복학|등록|장학|졸업|수강|학칙|규정|절차|신청|자격|요건)")
INTRO_QUERY_RE = re.compile(r"(소개|비전|상징|로고|캐릭터|연혁|학과|학부|전공)")

def _canonical_unit_from_query(q: str) -> str | None:
    toks = re.findall(HANGUL_TOKEN_PATTERN, q or "")
    units = [t for t in toks if UNIT_SUFFIX_RE.search(t)]
    return max(units, key=len) if units else None

def _preclean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\u00A0", " ").replace("\u200B", "")
    s = re.sub(f"[{DASH_CHARS_PATTERN}]", "-", s)
    return s

def _answer_text_from_row(row) -> str:
    return (row.get(ANSWER_TEXT_COL) or row.get("text") or row.get("content") or "")

def _answer_text_series(mask) -> pd.Series:
    col = ANSWER_TEXT_COL if ANSWER_TEXT_COL in search_df.columns else "text"
    return search_df.loc[mask, col].dropna().astype(str)

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
    base_text = _answer_text_from_row(row)
    if base_text:
        texts.append(base_text)

    if (not p) or (not e):
        mask = (search_df["url"].eq(row.get("url","")) &
                search_df["doc_type"].eq(row.get("doc_type","")))
        sibs = _answer_text_series(mask)
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
        m = GRAD_TABLE_TOTAL_RE.search(t)
        if m:
            vals = [
                int(val)
                for _, val in YEAR_PROGRAM_CREDIT_RE.findall(m.group("body"))
                if 60 <= int(val) <= 200
            ]
            if vals:
                total = max(vals)

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
ART_DIR          = env_path("ART_DIR")
SEARCH_DF_PATH   = env_path("SEARCH_DF_PATH")
EMB_PATH         = env_path("EMB_PATH")
BM25_PATH        = env_path("BM25_PATH")
TOK_PATH         = env_path("TOK_PATH")
CONTACTS_CSV     = env_path("CONTACTS_CSV")
META_PATH        = env_path("META_PATH")
CONTACT_KWS      = load_list_from_txt(env_path("CONTACT_KWS_PATH"))

embeddings = np.load(EMB_PATH, mmap_mode="r").astype(np.float32)
row_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
row_norms[row_norms == 0] = 1.0
embeddings = embeddings / row_norms

UNIT_TOK_RE = re.compile(HANGUL_TOKEN_PATTERN)

model_name = "intfloat/multilingual-e5-base"
model = SentenceTransformer(model_name)

search_df = normalize_search_df_schema(pd.read_parquet(SEARCH_DF_PATH))

tokenize_kor = get_tokenizer()

if BM25_PATH.exists():
    with open(BM25_PATH, 'rb') as f:
        bm25 = pickle.load(f)
else:
    if TOK_PATH.exists():
        tokenized_courpus = load_json_gz(str(TOK_PATH))
    else:
        tokenized_courpus = [tokenize_kor(t) for t in search_df["text_for_bm25"].astype(str).tolist()]
    bm25 = BM25Okapi(tokenized_courpus)



@lru_cache(maxsize=512)
def embed_query(q: str):
    return model.encode([f"query: {q}"], normalize_embeddings=True)[0].astype(np.float32)

def _minmax(x):
    x = np.asarray(x, float)
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo + 1e-9)

# 최신성 점수 함수 제거됨 (학교 기본 정보만 제공)

def hybrid_search(query, top_k=8, alpha=DEFAULT_DENSE_WEIGHT, contact_boost=CONTACT_DOC_BOOST):
    qv = embed_query(query)
    dense_raw = embeddings @ qv
    bm25_raw  = bm25.get_scores(tokenize_kor(query))
    base = alpha * _minmax(dense_raw) + (1 - alpha) * _minmax(bm25_raw)

    bonus = np.zeros(len(search_df))
    doc_type = search_df["doc_type"].astype(str)
    query_text = query or ""

    if any(k in query_text for k in CONTACT_KWS):
        is_contact = (doc_type == "contact").astype(float).to_numpy()
        phone_text = search_df["phone"].astype(str).str.strip()
        email_text = search_df["email"].astype(str).str.strip()
        has_phone = (
            search_df["has_phone"].astype(bool)
            | (phone_text.ne("") & phone_text.ne("없음"))
        ).astype(float).to_numpy()
        has_email = (
            search_df["has_email"].astype(bool)
            | (email_text.ne("") & email_text.ne("없음"))
        ).astype(float).to_numpy()
        bonus = contact_boost * (is_contact + PHONE_EMAIL_BONUS_RATIO*has_phone + PHONE_EMAIL_BONUS_RATIO*has_email)

    if _looks_like_grad_query(query_text) or POLICY_QUERY_RE.search(query_text):
        is_policy = doc_type.isin(["policy", "table_like"]).astype(float).to_numpy()
        has_policy = search_df["has_policy_keyword"].astype(float).to_numpy()
        has_credit = search_df["has_credit"].astype(float).to_numpy()
        bonus += 0.08 * is_policy + 0.05 * has_policy + 0.12 * has_credit

    if INTRO_QUERY_RE.search(query_text):
        is_intro = doc_type.isin(["intro", "department", "history"]).astype(float).to_numpy()
        bonus += 0.05 * is_intro

    if not PRIVACY_QUERY_RE.search(query_text):
        is_privacy = doc_type.eq("privacy").astype(float).to_numpy()
        is_old_privacy = search_df["is_privacy_old"].astype(float).to_numpy()
        bonus -= 0.12 * is_privacy + 0.18 * is_old_privacy

    scores = base + bonus
    out = search_df.copy()
    out["score"] = scores
    if "url" in out.columns:
        rep_idx = out.groupby(["url","doc_type"], dropna=False)["score"].idxmax()
        out = out.loc[rep_idx]
    out = out.sort_values(["score"], ascending=[False], na_position="last").head(top_k)
    return out.reset_index(drop=True)

def build_answer(query, top_k=6):
    contact_like = any(k in query for k in CONTACT_KWS)

    if contact_like:
        hits = hybrid_search(query, top_k=max(top_k, CONTACT_SEARCH_POOL_SIZE))

        toks = re.findall(HANGUL_TOKEN_PATTERN, query or "")
        prefer = [t for t in toks if re.search(UNIT_QUERY_SUFFIX_PATTERN, t)]
        terms = prefer if prefer else toks[:3]
        target_unit = _canonical_unit_from_query(query)

        # 교직원검색(4408) 페이지를 후보에 반드시 포함
        staff_mask = (search_df["url"].astype(str).str.contains(STAFF_URL_PATTERN, case=False, regex=True) |
                    search_df["title"].astype(str).str.contains(STAFF_TITLE_PATTERN, case=False, regex=True))
        staff_rows = search_df.loc[staff_mask].copy()
        if not staff_rows.empty:
            hits = pd.concat([hits, staff_rows], ignore_index=True)
            hits = hits.drop_duplicates(subset=["url","doc_type"], keep="first")

        # 재정렬 가중치: 단위매칭/교직원검색/연락처문서/학부홈 가산, 게시판 감점
        urls   = hits["url"].fillna("")
        titles = hits["title"].fillna("")
        units  = hits["unit"].fillna("")

        unit_match = pd.Series([False]*len(hits))
        if target_unit:
            unit_match = (units.str.contains(re.escape(target_unit))) | (titles.str.contains(re.escape(target_unit)))
        unit_match = unit_match.astype(int)

        is_staff    = (
            urls.str.contains(STAFF_URL_PATTERN, case=False, regex=True)
            | titles.str.contains(STAFF_TITLE_PATTERN, case=False, regex=True)
        ).astype(int)
        is_contact  = hits["doc_type"].eq("contact").astype(int)
        is_dept_home= urls.str.contains(DEPT_HOME_URL_PATTERN, case=False, regex=True).astype(int)
        is_board    = urls.str.contains(BOARD_URL_PATTERN, case=False, regex=True).astype(int)

        _prio = (
            UNIT_MATCH_BOOST * unit_match
            + STAFF_PAGE_BOOST * is_staff
            + CONTACT_DOC_RERANK_BOOST * is_contact
            + DEPT_HOME_BOOST * is_dept_home
            - BOARD_PAGE_PENALTY * is_board
        )
        hits = hits.assign(_prio=_prio).sort_values(["_prio","score"], ascending=[False, False])

        # 번호 추출
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

    # 졸업학점 관련 질문 처리
    if _looks_like_grad_query(query):
        three_year_like = bool(re.search(r"\b3\s*년제\b", query))

        hits = hybrid_search(query, top_k=max(top_k, GRAD_SEARCH_POOL_SIZE))
        urls = hits["url"].fillna("")
        home_like = urls.str.contains(HOME_LIKE_URL_PATTERN, case=False, regex=True)
        bbs_like  = urls.str.contains(BOARD_URL_PATTERN, case=False, regex=True)
        is_page   = hits["doc_type"].isin(["page", "policy", "table_like", "department"])
        rr = (
            HOME_PAGE_RERANK_BOOST * home_like.astype(int)
            + PAGE_DOC_RERANK_BOOST * is_page.astype(int)
            - BOARD_PAGE_PENALTY * bbs_like.astype(int)
        )
        hits = hits.assign(_rr=rr).sort_values(["_rr", "score"], ascending=[False, False])

        if three_year_like:
            for _, r in hits.iterrows():
                text = _answer_text_from_row(r)
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
            text = _answer_text_from_row(r)
            total, parts = _extract_grad_credits_from_text(text)

            if not (total or parts):
                if "doc_id" in search_df.columns and pd.notna(r.get("doc_id", np.nan)):
                    mask = search_df["doc_id"].eq(r["doc_id"])
                else:
                    mask = search_df["url"].eq(r.get("url","")) & search_df["doc_type"].eq(r.get("doc_type",""))
                sibs = _answer_text_series(mask)
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

    # 일반 검색 (연락처가 아닌 경우)
    hits = hybrid_search(query, top_k=max(top_k, GENERAL_SEARCH_POOL_SIZE))

    if not hits.empty:
        urls = hits["url"].fillna("")
        home_like = urls.str.contains(HOME_LIKE_URL_PATTERN, case=False, regex=True)
        bbs_like  = urls.str.contains(BOARD_URL_PATTERN, case=False, regex=True)
        is_page   = hits["doc_type"].isin(["page", "intro", "department", "policy", "table_like", "history"])
        def _n(s: str) -> str: return re.sub(r"\s+", "", (s or "").lower())
        qn = _n(query)
        title_match = hits["title"].fillna("").map(lambda s: qn in _n(s))
        unit_match  = hits.get("unit", pd.Series([""]*len(hits))).fillna("").map(lambda s: qn in _n(s))
        rr = (
            HOME_PAGE_RERANK_BOOST * home_like.astype(int)
          + PAGE_DOC_RERANK_BOOST * is_page.astype(int)
          + TITLE_UNIT_MATCH_BOOST * (title_match.astype(int) | unit_match.astype(int))
          - BOARD_PAGE_PENALTY * bbs_like.astype(int)
        )
        hits = hits.assign(_rr=rr).sort_values(["_rr", "score"], ascending=[False, False])

    picks = hits.head(top_k)[["title", "url"]].fillna("")
    lines = [f"- {r['title']}: {r['url']}" for _, r in picks.iterrows()]
    return {"answer": "\n".join(lines)}

if __name__ == "__main__":
    q1 = "학생성공지원팀 담당자 전화번호 알려줘"
    q2 = "컴퓨터공학부 담당자 전화번호 알려줘"
    q3 = "경영학부 담당자 연락처"
    q4 = "졸업학점 요구사항"
    q5 = "3년제 졸업학점"
    for q in [q1, q2, q3, q4, q5]:
        res = build_answer(q, top_k=8)
        print("Q:", q)
        print("A:\n", res["answer"])
        print("-" * 50)
