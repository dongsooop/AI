# -*- coding: utf-8 -*-
import re, json, gzip
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))

# ---------- 텍스트 정규화 ----------
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

# ---------- 날짜 파싱 ----------
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

# ---------- 문장 분리 ----------
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

# ---------- 청킹 ----------
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

# ---------- 토크나이저 ----------
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

# ---------- 연락처/조직 추출 ----------
PHONE_RE = re.compile(r"\b0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}\b")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
UNIT_RE  = re.compile(r"([가-힣A-Za-z·\s\-/()]{2,40}?(?:팀|센터|처|단|과|부|본부|위원회|연구소|지원실|실|연대|학부|학과))")

def last_seg(title: str):
    return title.split("/")[-1].strip() if isinstance(title, str) else ""

def clean_name(s: str) -> str:
    s = re.sub(r"\([^)]*\)", "", str(s))
    s = re.sub(r"[\s\-/·]+", "", s)
    return s.lower()

def extract_units_and_contacts(row):
    text = row.get("content", "") if isinstance(row.get("content", ""), str) else ""
    title = row.get("title", ""); url = row.get("url", ""); source = row.get("source", "")
    updated_at = row.get("updated_at", None)
    out = []
    for m in UNIT_RE.finditer(text):
        unit = m.group(1).strip()
        win = 140
        start, end = max(0, m.start()-win), min(len(text), m.end()+win)
        near = text[start:end]
        phones = PHONE_RE.findall(near)
        emails = EMAIL_RE.findall(near)
        out.append({
            "unit": unit, "phone": phones[-1] if phones else "없음",
            "email": emails[-1] if emails else "없음",
            "title": title, "url": url, "source": source,
            "updated_at": updated_at, "how": "content"
        })
    hint = last_seg(title)
    mh = UNIT_RE.search(hint)
    if mh:
        phones = PHONE_RE.findall(text)
        emails = EMAIL_RE.findall(text)
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

# ---------- 직렬화 ----------
def dump_json_gz(obj, path):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def load_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)