import re, json, gzip
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def try_parse_date(x: str):
    x = str(x)
    fmts = ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d", "%Y%m%d")
    for fmt in fmts:
        try:
            return datetime.strptime(x[:10], fmt).replace(tzinfo=KST)
        except:
            pass
    m = re.search(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", x)
    if m:
        y, mo, d = map(int, m.groups())
        return datetime(y, mo, d, tzinfo=KST)
    return None


DATE_PATTERNS = [
    re.compile(r"(?:작성일|등록일|게시일)\s*[:\-]?\s*(\d{4}[./-]\d{1,2}[./-]\d{1,2})"),
    re.compile(r"(?:작성일|등록일|게시일)\s*[:\-]?\s*(\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일)")
]


def extract_date_from_content(txt: str):
    if not isinstance(txt, str):
        return None
    for pat in DATE_PATTERNS:
        m = pat.search(txt)
        if m:
            return try_parse_date(m.group(1))
    return None


try:
    import kss
except ImportError:
    kss = None

_SENT_PAT = re.compile(r'.+?(?:다\.|요\.|[.!?])(?=\s+|$)')


def _regex_split(text: str):
    text = ' '.join(str(text).split())
    if not text:
        return []
    sents = _SENT_PAT.findall(text)
    return sents if sents else [text]


def sent_split(text: str):
    t = ' '.join(str(text).split())
    if not t:
        return []
    if kss is not None:
        try:
            return [s for s in kss.split_sentences(t) if s.strip()]
        except Exception:
            pass
    return _regex_split(t)


def chunk_text(text, max_tokens=400, overlap=0.15):
    sents = [s for s in sent_split(text) if s.strip()]
    if not sents:
        return []
    chunks, cur, tok = [], [], 0
    ovl = max(int(max_tokens * overlap), 0)
    for s in sents:
        t = len(s.split())
        if t >= max_tokens and not cur:
            chunks.append(s.strip()); continue
        if tok + t <= max_tokens:
            cur.append(s); tok += t
        else:
            if cur:
                blob = " ".join(cur).strip()
                chunks.append(blob)
                if ovl > 0:
                    tail_tokens = blob.split()[-ovl:]
                    tail = " ".join(tail_tokens)
                    cur = [tail, s]; tok = len(tail_tokens) + t
                else:
                    cur = [s]; tok = t
            else:
                chunks.append(s.strip()); cur, tok = [], 0
    if cur:
        chunks.append(" ".join(cur).strip())
    return chunks


def get_tokenizer():
    try:
        from konlpy.tag import Okt
        okt = Okt()
        def tokenize_kor(s):
            return [w for w, pos in okt.pos(s, norm=True, stem=True)
                    if pos not in ("Josa","Punctuation","Foreign")]
        return tokenize_kor
    except Exception:
        def tokenize_kor(s):
            return re.findall(r"[가-힣A-Za-z0-9]+", s)
        return tokenize_kor


PHONE_RE = re.compile(r"\b0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}\b")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
UNIT_RE  = re.compile(r"([가-힣A-Za-z·\s\-/()]{2,40}?(?:팀|센터|처|단|과|부|본부|위원회|연구소|지원실|실|연대|학부|학과))")
CONTACT_UNIT_SUFFIX_RE = re.compile(r"(팀|센터|처|단|본부|위원회|연구소|지원실|연대|학부|학과)$")
INVALID_CONTACT_UNIT_RE = re.compile(r"(담당부|서명과|시그니처|다운로드|파일|조합|타입|구분|항목|내용)")

DASH_CHARS_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212\uFE58\uFE63\uFF0D]")
EDITOR_ONLY_RE = re.compile(r"해당\s*영역은\s*편집모드에서만\s*보여지는\s*영역입니다\.?")
EDITOR_DEPT_RE = re.compile(r"담당부서명과\s*전화번호만\s*수정해주시면\s*됩니다\.?")
EDITOR_FOOTER_RE = re.compile(
    r"해당\s*영역은\s*편집모드에서만\s*보여지는\s*영역입니다\.?"
    r".*?(?:담당부서명과\s*전화번호만\s*수정해주시면\s*됩니다\.?)?"
    r"(?:\s*[가-힣A-Za-z·\s/()]{2,40}(?:\s*0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4})?)?\s*$"
)
FNCT_RE = re.compile(r"\bfnctId=[A-Za-z0-9_-]+,?\s*fnctNo=\d+\b")
DOWNLOAD_NOISE_RE = re.compile(r"(?:파일\s*)?다운로드")
POLICY_ARTICLE_RE = re.compile(r"(?=(?:제\s*\d+\s*조|[①-⑳]|\d+\.\s|[가-힣]\.\s))")
HISTORY_YEAR_RE = re.compile(r"(?=\b(?:19|20)\d{2}\s*년\b)")
TABLE_MARKER_RE = re.compile(r"(?=\b[A-Z]\s*\d{1,2}\b|\b[A-Z]\d{1,2}\b)")
DATE_RE = re.compile(r"(?:19|20)\d{2}[.\-/년]\s*\d{1,2}|\d{1,2}\s*월|\d{1,2}\s*일")
CREDIT_RE = re.compile(r"\d{2,3}\s*학점|졸업\s*학점|이수\s*학점")
POLICY_KEYWORD_RE = re.compile(r"학칙|규정|규칙|시행세칙|휴학|복학|등록|장학|졸업|수강|신청|절차|대상|자격")
PRIVACY_KEYWORD_RE = re.compile(r"개인정보|영상정보|처리방침|이메일무단수집거부|이전방침")
HISTORY_KEYWORD_RE = re.compile(r"연혁|창학|성장과\s*번영|변화와\s*혁신")
DEPARTMENT_KEYWORD_RE = re.compile(r"학부|학과|전공|계열")
INTRO_KEYWORD_RE = re.compile(r"대학소개|비전|상징|로고|캐릭터|교가|총장|설립이념")


def clean_content(raw: str, strip_editor_footer: bool = True) -> str:
    text = "" if raw is None else str(raw)
    text = text.replace("\u00A0", " ").replace("\u200B", "")
    text = DASH_CHARS_RE.sub("-", text)
    text = FNCT_RE.sub(" ", text)
    if strip_editor_footer:
        text = EDITOR_FOOTER_RE.sub(" ", text)
        text = EDITOR_ONLY_RE.sub(" ", text)
        text = EDITOR_DEPT_RE.sub(" ", text)
    text = DOWNLOAD_NOISE_RE.sub(" ", text)
    return normalize_text(text)


def _title_parts(title: str) -> tuple[str, str]:
    title = normalize_text(title)
    if not title:
        return "", ""
    parts = [p.strip() for p in title.split("/") if p.strip()]
    return "/".join(parts), (parts[-1] if parts else title)


def _count_table_markers(text: str) -> int:
    markers = re.findall(r"\b[A-Z]\s*\d{1,2}\b|\b[A-Z]\d{1,2}\b|구\s*분|항\s*목|내\s*용", text)
    return len(markers)


def classify_doc_type(title: str, url: str = "", content: str = "") -> str:
    title = normalize_text(title)
    url = str(url or "")
    text = clean_content(content, strip_editor_footer=False)
    haystack = f"{title} {url} {text[:1200]}"

    if PRIVACY_KEYWORD_RE.search(haystack) or "/49" in url and "subview.do" in url:
        return "privacy"
    if HISTORY_KEYWORD_RE.search(haystack) or len(HISTORY_YEAR_RE.findall(text)) >= 6:
        return "history"
    if "교직원" in title or "전화번호 안내" in title or re.search(r"/4408/subview\.do", url):
        return "contact"
    if DEPARTMENT_KEYWORD_RE.search(title) and not INTRO_KEYWORD_RE.search(title):
        return "department"
    if POLICY_KEYWORD_RE.search(haystack):
        return "policy"
    if _count_table_markers(text) >= 8:
        return "table_like"
    if INTRO_KEYWORD_RE.search(haystack):
        return "intro"
    return "page"


def _split_by_pattern(text: str, pattern: re.Pattern) -> list[str]:
    pieces = [p.strip() for p in pattern.split(text) if p and p.strip()]
    return pieces if pieces else ([text.strip()] if text.strip() else [])


def _token_len(text: str) -> int:
    return len(str(text).split())


def _split_long_text(text: str, max_tokens: int, overlap: int = 0) -> list[str]:
    words = str(text).split()
    if len(words) <= max_tokens:
        return [normalize_text(text)] if text else []
    step = max(max_tokens - overlap, 1)
    out = []
    for start in range(0, len(words), step):
        piece = words[start:start + max_tokens]
        if piece:
            out.append(" ".join(piece))
    return out


def _pack_segments(segments: list[str], max_tokens: int, overlap: int = 0) -> list[str]:
    chunks, cur, tok = [], [], 0
    for seg in segments:
        seg = normalize_text(seg)
        if not seg:
            continue
        seg_tok = _token_len(seg)
        if seg_tok >= max_tokens:
            if cur:
                chunks.append(normalize_text(" ".join(cur)))
                cur, tok = [], 0
            chunks.extend(_split_long_text(seg, max_tokens=max_tokens, overlap=overlap))
            continue
        if tok + seg_tok <= max_tokens:
            cur.append(seg)
            tok += seg_tok
            continue
        if cur:
            blob = normalize_text(" ".join(cur))
            chunks.append(blob)
            if overlap > 0:
                tail = " ".join(blob.split()[-overlap:])
                cur = [tail, seg] if tail else [seg]
                tok = _token_len(tail) + seg_tok
            else:
                cur, tok = [seg], seg_tok
    if cur:
        chunks.append(normalize_text(" ".join(cur)))
    return chunks


def _segments_for_doc_type(text: str, doc_type: str) -> tuple[list[str], int, int]:
    if doc_type == "history":
        return _split_by_pattern(text, HISTORY_YEAR_RE), 650, 30
    if doc_type == "policy":
        return _split_by_pattern(text, POLICY_ARTICLE_RE), 520, 35
    if doc_type == "table_like":
        return _split_by_pattern(text, TABLE_MARKER_RE), 520, 25
    if doc_type == "privacy":
        return _split_by_pattern(text, POLICY_ARTICLE_RE), 600, 25
    if doc_type == "contact":
        return [text], 220, 0
    if doc_type in ("intro", "department"):
        return sent_split(text), 360, 25
    return sent_split(text), 400, 40


def chunk_document(row, max_tokens: int | None = None) -> list[dict]:
    title = row.get("title", "") if hasattr(row, "get") else ""
    url = row.get("url", "") if hasattr(row, "get") else ""
    raw_content = row.get("content", "") if hasattr(row, "get") else ""
    breadcrumb, leaf_title = _title_parts(title)
    doc_type = classify_doc_type(title, url, raw_content)
    content = clean_content(raw_content)
    if not content:
        return []

    segments, default_max, overlap = _segments_for_doc_type(content, doc_type)
    limit = max_tokens or default_max
    chunks = _pack_segments(segments, max_tokens=limit, overlap=overlap)
    out = []
    for idx, chunk in enumerate(chunks):
        answer_text = chunk
        embedding_text = normalize_text(f"{breadcrumb} {leaf_title} {chunk}")
        bm25_text = normalize_text(f"{breadcrumb} {leaf_title} {chunk}")
        out.append({
            "chunk_index": idx,
            "title": title,
            "url": url,
            "breadcrumb": breadcrumb,
            "leaf_title": leaf_title,
            "doc_type": doc_type,
            "chunk_type": doc_type,
            "section_title": leaf_title,
            "text": answer_text,
            "text_for_embedding": embedding_text,
            "text_for_bm25": bm25_text,
            "text_for_answer": answer_text,
            "has_phone": bool(PHONE_RE.search(chunk)),
            "has_email": bool(EMAIL_RE.search(chunk)),
            "has_date": bool(DATE_RE.search(chunk)),
            "has_credit": bool(CREDIT_RE.search(chunk)),
            "has_policy_keyword": bool(POLICY_KEYWORD_RE.search(chunk)),
            "is_privacy_old": doc_type == "privacy" and bool(re.search(r"이전|구\)|\(\s*구", title)),
        })
    return out


def last_seg(title: str):
    return title.split("/")[-1].strip() if isinstance(title, str) else ""


def clean_name(s: str) -> str:
    s = re.sub(r"\([^)]*\)", "", str(s))
    s = re.sub(r"[\s\-/·]+", "", s)
    return s.lower()


def _is_contact_unit_candidate(unit: str) -> bool:
    unit = normalize_text(unit)
    compact = re.sub(r"[\s\-/·()]+", "", unit)
    if not compact or len(compact) < 2 or len(compact) > 25:
        return False
    if INVALID_CONTACT_UNIT_RE.search(unit):
        return False
    return bool(CONTACT_UNIT_SUFFIX_RE.search(compact))


def extract_units_and_contacts(row):
    text = row.get("content", "") if isinstance(row.get("content", ""), str) else ""
    title = row.get("title", ""); url = row.get("url", ""); source = row.get("source", "")
    updated_at = row.get("updated_at", None)
    out = []
    for m in UNIT_RE.finditer(text):
        unit = m.group(1).strip()
        if not _is_contact_unit_candidate(unit):
            continue
        win = 140
        start, end = max(0, m.start()-win), min(len(text), m.end()+win)
        near = text[start:end]
        phones = PHONE_RE.findall(near)
        emails = EMAIL_RE.findall(near)
        if not phones and not emails:
            continue
        out.append({
            "unit": unit, "phone": phones[-1] if phones else "없음",
            "email": emails[-1] if emails else "없음",
            "title": title, "url": url, "source": source,
            "updated_at": updated_at, "how": "content"
        })
    hint = last_seg(title)
    mh = UNIT_RE.search(hint)
    if mh:
        if not _is_contact_unit_candidate(mh.group(1).strip()):
            mh = None
    if mh:
        phones = PHONE_RE.findall(text)
        emails = EMAIL_RE.findall(text)
        if phones or emails:
            out.append({
                "unit": mh.group(1).strip(),
                "phone": phones[-1] if phones else "없음",
                "email": emails[-1] if emails else "없음",
                "title": title, "url": url, "source": source,
                "updated_at": updated_at, "how": "title"
            })
    if not out:
        phones = PHONE_RE.findall(text)
        emails = EMAIL_RE.findall(text)
        if phones or emails:
            cand = hint if hint else "미상"
            out.append({
                "unit": cand,
                "phone": phones[-1] if phones else "없음",
                "email": emails[-1] if emails else "없음",
                "title": title, "url": url, "source": source,
                "updated_at": updated_at, "how": "fallback"
            })
    return out


CONTACT_HINTS = "연락처 전화 전화번호 이메일 메일 문의 상담 교직원 직원 담당자"


def compose_contact_passage(unit, phone, email, origin_title, origin_url):
    phone_part = f"전화 {phone}" if phone and phone != "없음" else ""
    email_part = f"이메일 {email}" if email and email != "없음" else ""
    joined = " / ".join([p for p in [phone_part, email_part] if p])
    core = f"{unit} 연락처 안내. {joined}" if joined else f"{unit} 연락처 안내."
    tail = f" (출처: {origin_title})"
    return f"{core}{tail} {CONTACT_HINTS} {unit} {origin_title} {origin_url}"


def dump_json_gz(obj, path):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def load_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)
