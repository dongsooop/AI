import os, sys, re
from pathlib import Path
from typing import List, Dict, Optional
from jose import JWTError, jwt
from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import base64
from jose.exceptions import ExpiredSignatureError
import datetime as dt

from LLM.sub_model.query_index import build_answer
from LLM.sub_model.schedule_index import schedule_search

oss_router = APIRouter()
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ORG_NAME = os.getenv("ORG_NAME", "해당 기관")
BOT_NAME = os.getenv("BOT_NAME", "챗봇")
SERVICE_NAME = os.getenv("SERVICE_NAME", "챗봇 서비스")
ORG_HOMEPAGE_LABEL = os.getenv("ORG_HOMEPAGE_LABEL", "공식 홈페이지")
ORG_HOMEPAGE_URL = os.getenv("ORG_HOMEPAGE_URL", "https://example.com")
GRAD_PAGE_URL = os.getenv("GRAD_PAGE_URL", ORG_HOMEPAGE_URL)
STAFF_URL_PATTERN = os.getenv("STAFF_URL_PATTERN", r"/staff|/contact")
BOT_ALIASES = tuple(x.strip() for x in os.getenv("BOT_ALIASES", "").split(",") if x.strip())
OSS_BASE_URL = os.getenv("OSS_BASE_URL")
OSS_API_KEY  = os.getenv("OSS_API_KEY")
OSS_MODEL    = os.getenv("OSS_MODEL")

client = OpenAI(base_url=OSS_BASE_URL, api_key=OSS_API_KEY)
PHONE_GUARD_PAT = re.compile(r"(?:\+?\d[\d\s\--–—−]{6,}\d)")
URL_PAT         = re.compile(r"https?://\S+")
PHONE_PAT_KR  = re.compile(r'(0(?:2|[3-9]\d))\D{0,2}(\d{3,4})\D{0,2}(\d{4})')
TEL_FIRST_PAT = re.compile(
    r'(?:TEL\.?|Tel\.?|T\.?|전화(?:번호)?|연락처)\s*[:.\-]?\s*'
    r'(0(?:2|[3-9]\d))\D{0,2}(\d{3,4})\D{0,2}(\d{4})'
)
LABEL_PAT       = re.compile(r'^\s*-\s*([^:：]+)[:：]\s*', re.MULTILINE)
LINE_PAT        = re.compile(r'^\s*-\s*(?P<label>[^:：]+)[:：]\s*(?P<body>.+)$', re.MULTILINE)
SRC_URL_PAT     = re.compile(r'\(출처:\s*(?P<url>https?://[^)]+)\)')
SRC_URL_DOT_PAT = re.compile(r'\(출처:\s*[^·\)]*·\s*(?P<url>https?://[^)]+)\)')
TITLE_URL_PAT   = re.compile(r"^\s*-\s*(?P<title>[^:]+):\s*(?P<url>\S+)", re.MULTILINE)
GOOD_URL_RE     = re.compile(r"/subview\.do($|\?)", re.I)
STAFF_URL_RE    = re.compile(STAFF_URL_PATTERN, re.I)
BAD_URL_RE      = re.compile(r"(?:/bbs/|artclView\.do|combBbs)", re.I)
SAFETY_URL_RE   = re.compile(r"(safety|lab|ehs|env|환경|안전)", re.I)
SAFETY_TITLE_RE = re.compile(r"(연구실|실험실|실습실|안전|환경안전|EHS)", re.I)
CONTACT_WORD_RE = re.compile(r"(연락처|전화|전화번호|문의)", re.I)
INTRO_WORD_RE   = re.compile(r"(학부\s*소개|학과\s*소개|소개|안내|개요)", re.I)
SCHED_LINE_RE = re.compile(
    r"^\s*-\s*(?P<title>[^:]+):\s*(?P<s>\d{4}-\d{2}-\d{2})(?:\s*~\s*(?P<e>\d{4}-\d{2}-\d{2}))?$",
    re.MULTILINE
)
GREETING_RE = re.compile(r"^\s*(안녕|안녕하세요|하이|hello|hi)\b", re.I)
WHOAMI_EN   = re.compile(r"\b(who\s+are\s+you|what\s+are\s+you|your\s+name)\b", re.I)
WHOAMI_KO   = re.compile(r"(무슨\s*(챗봇|봇)|뭐\s*하는\s*(챗봇|봇)|역할|무엇을\s*할\s*수|기능|소개해\s*줘)")
RELATION_RE = re.compile(r"(우리\s*관계|우린\s*무슨\s*관계|너와\s*나|무슨\s*사이|사이(?:야|냐)?|관계(?:야|냐)?)")
UNIT_SUFFIX_RE = re.compile(r"(학부|학과|과|전공|대학|대학원|본부|센터|팀|처|단|부|원)$")
DEPT_ROOT_SUFFIX_RE = re.compile(r"(공학부|공학과|학부|학과|과)$")
UNIT_LIKE_LABEL_RE  = re.compile(r"(학과|학부)")
NON_DEPT_LABEL_RE   = re.compile(r"(팀|지원팀|실|센터|본부)$")
GENERIC_LABEL_RE    = re.compile(r"(담당부|연락처|대표|대표번호)")
NON_TARGET_TEAM_PAT = re.compile(r"(정보지원팀|전산실|시설관리팀|재무팀|홍보팀|홍보대사단)")
CEREMONY_RE = re.compile(r"(졸업식|학위수여식)")
GRAD_POLICY_RE = re.compile(r"(졸업학점|이수학점|졸업요건|전공최저|최저이수|학위수여(?!식)|졸업(?!식))")

PRONOUNS    = ("너","넌","니","네가","당신","챗봇","봇","ai","에이아이", *BOT_ALIASES)
QWORDS      = ("누구","정체","이름","뭐야","뭐냐","무엇","뉘신","누구야","누구냐")
TOPIC_KWS = ("학부","학과","전공","소개","안내","위치","학사","졸업")
COUNCIL_KWS = (
    "총학생회","총학생","총학","대의원회","대의원실",
    "학생회","학생대표기구","홍보대사단","동아리연합회","동양학보","학보사","총학생실"
)
SCHEDULE_HINTS_BASE = (
    "학사일정","학사 일정","중간","중간고사","기말","기말고사",
    "수강","정정","성적","등록","보강","개강","종강",
    "휴일","공휴","시험","고사","이번주","다음주","이번달","다음달",
    "졸업식","학위수여식"
)
STOP_TOKENS = {"담당","담당부","전화","전화번호","연락처","문의","상담","번호"}
CONTACT_INTENT_RE = re.compile(r"(연락처|전화|전화번호|문의|상담|담당자)")


def verify_jwt_token(request: Request):
    if not SECRET_KEY or not ALGORITHM:
        raise HTTPException(status_code=500, detail="Server auth configuration missing")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or malformed")

    token = auth_header.split(" ")[1]
    try:
        padded_key = SECRET_KEY + '=' * (-len(SECRET_KEY) % 4)
        sc = base64.urlsafe_b64decode(padded_key)
        payload = jwt.decode(token, sc, algorithms=[ALGORITHM])

        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token: no subject")
        return username

    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def _resolve_repo_root() -> Path:
    env_root = os.getenv("ROOT_BASE_PATH")
    if env_root:
        p = Path(os.path.expanduser(env_root)).resolve()
    else:
        p = Path(__file__).resolve().parents[2]
    return p

REPO_ROOT = _resolve_repo_root()
if REPO_ROOT.exists() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _normalize_heading_from_query(user_text: str) -> str:
    t = re.sub(r"\s+", " ", (user_text or "").strip())
    t = re.sub(r"[?？!！]+$", "", t).strip()
    if not t:
        return "학사일정 안내"
    if t.endswith("일정"):
        return t
    if t.endswith("안내"):
        return t
    return f"{t} 일정"


def render_chatty_schedule(bullets: str, user_text: str = "") -> str:
    w = "월화수목금토일"
    lines = []
    for m in SCHED_LINE_RE.finditer(bullets or ""):
        title = m.group("title").strip()
        s = dt.date.fromisoformat(m.group("s"))
        e = dt.date.fromisoformat(m.group("e")) if m.group("e") else s
        when = (
            f"{s.isoformat()}({w[s.weekday()]})"
            if s == e else
            f"{s.isoformat()}({w[s.weekday()]}) ~ {e.isoformat()}({w[e.weekday()]})"
        )
        lines.append(f"+ {title}: {when}")

    if not lines:
        return "관련 일정을 찾지 못했어요."

    heading = _normalize_heading_from_query(user_text)
    return heading + "\n" + "\n".join(lines)


def is_whoami(text: str) -> bool:
    if not text: return False
    t = text.strip()
    if WHOAMI_EN.search(t) or WHOAMI_KO.search(t): return True
    ko = re.sub(r"\s+", "", t).lower()
    return any(p in ko for p in PRONOUNS) and any(q in ko for q in QWORDS)


def is_relation(text: str) -> bool:
    if not text: return False
    return bool(RELATION_RE.search(text))


def looks_like_topic(text: str) -> bool:
    if not text: return False
    t = text.strip()
    if any(k in t for k in TOPIC_KWS): return True
    return bool(UNIT_SUFFIX_RE.search(t)) and len(t) <= 12


def _kcompact(s: str) -> str:
    return re.sub(r"[^가-힣A-Za-z0-9]", "", s or "")


def _canon_unit(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "", s).replace("·", "").replace(".", "")
    s = re.sub(r"(학부|학과|공학과|과)$", "", s)
    return s


def _dept_map_path() -> Path:
    env_p = os.getenv("DEPT_MAP_PATH")
    if env_p:
        return Path(os.path.expanduser(env_p)).resolve()
    return (REPO_ROOT / "data" / "department.txt").resolve()

def _make_aliases(name: str) -> set[str]:
    raw = name.strip()
    variants = {
        raw,
        re.sub(r"학과$", "과", raw),
        re.sub(r"공학과$", "학과", raw),
        re.sub(r"공학부$", "학부", raw),
    }
    canon = _canon_unit(raw)
    root  = re.sub(DEPT_ROOT_SUFFIX_RE, "", canon)
    if canon:
        variants |= {canon, canon+"학부", canon+"학과", canon+"공학과", canon+"과", canon+"전공"}
    if root and len(root) >= 2:
        variants |= {
            root, root+"학부", root+"학과", root+"공학과", root+"과", root+"전공",
            root+"정보", root+"소프트웨어", root+"SW"
        }
    variants = {v for v in variants if v and isinstance(v, str)}
    return variants

def load_dept_map():
    path = _dept_map_path()
    dept_map, alias = {}, {}
    if not path.exists():
        return dept_map, alias
    with path.open(encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                name, rest = line.split(",", 1)
                url_part = rest.split(",")[0].strip()
                base = url_part.split("?")[0]
                c = _canon_unit(name)
                dept_map[c] = {"name": name, "path_base": base}
                als = _make_aliases(name)
                alias[c] = als
            except Exception:
                continue
    return dept_map, alias

DEPT_MAP, DEPT_ALIAS = load_dept_map()

ALL_DEPT_ALIASES: set[str] = set()
for _s in DEPT_ALIAS.values():
    ALL_DEPT_ALIASES |= set(_s)


def extract_roots(user_text: str) -> set[str]:
    roots: set[str] = set()
    for tok in re.findall(r"[가-힣A-Za-z0-9·]{2,}", user_text or ""):
        base = _canon_unit(tok)
        base = re.sub(r"공$", "", base)
        if len(base) >= 2:
            roots.add(base)
        if len(base) >= 3:
            roots.add(base[:3])
        if len(base) >= 2:
            roots.add(base[:2])
    return roots


def detect_dept_hint(text: str):
    toks = re.findall(r"[가-힣A-Za-z0-9·]{2,}", text or "")
    for t in toks:
        c = _canon_unit(t)
        if c in DEPT_MAP:
            data = DEPT_MAP[c]
            return {
                "canon": c,
                "aliases": DEPT_ALIAS.get(c, {data["name"]}),
                "path_base": data["path_base"],
            }
    roots = extract_roots(text)
    if roots:
        best = None
        best_len = -1
        for canon_key in DEPT_MAP.keys():
            for r in roots:
                if r and r in canon_key and len(r) > best_len:
                    best, best_len = canon_key, len(r)
        if best:
            data = DEPT_MAP[best]
            return {
                "canon": best,
                "aliases": DEPT_ALIAS.get(best, {data["name"]}),
                "path_base": data["path_base"],
            }
    return None


def _extract_dept_query_core(text: str) -> str:
    t = (text or "").strip()
    t = CONTACT_INTENT_RE.sub(" ", t)
    t = re.sub(r"(알려줘|알려\s*주세요|어디|뭐야|뭐지|좀)", " ", t)
    t = re.sub(r"[^가-힣A-Za-z0-9·\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _find_dept_candidates(core: str, limit: int = 5) -> list[str]:
    if not core:
        return []
    toks = [x for x in re.findall(r"[가-힣A-Za-z0-9·]{2,}", core) if x not in STOP_TOKENS]
    if not toks:
        return []
    scored = []
    for canon, meta in DEPT_MAP.items():
        aliases = DEPT_ALIAS.get(canon, {meta["name"]})
        score = 0.0
        for tok in toks:
            if any((tok in a) or (a in tok) for a in aliases):
                score += 3.0
            elif tok in canon:
                score += 2.0
            elif len(tok) >= 2 and tok[:2] in canon:
                score += 1.0
        if score > 0:
            scored.append((score, meta["name"]))
    scored.sort(key=lambda x: (-x[0], x[1]))
    names = [n for _, n in scored]
    return list(dict.fromkeys(names))[:limit]


def dept_clarification_message(user_text: str) -> Optional[str]:
    if not CONTACT_INTENT_RE.search(user_text or ""):
        return None
    core = _extract_dept_query_core(user_text)
    if not core:
        return "학과(또는 부서) 풀네임으로 다시 입력해 주세요. 예: 컴퓨터소프트웨어공학과 담당자 연락처"

    hint = detect_dept_hint(core)
    candidates = _find_dept_candidates(core, limit=5)
    compact = re.sub(r"\s+", "", core)
    short_like = (len(compact) <= 4) and not re.search(r"(학과|학부|전공|공학과|공학부)$", compact)

    if short_like or (len(candidates) >= 2 and hint is None):
        if candidates:
            picks = ", ".join(candidates[:4])
            return f"학과명이 축약되어 후보가 여러 개예요. 풀네임으로 입력해 주세요. 예: {picks}"
        return "학과명이 축약되어 정확한 매칭이 어려워요. 학과 풀네임으로 입력해 주세요. 예: 컴퓨터소프트웨어공학과 담당자 연락처"
    return None


def expand_synonyms(user_text: str) -> list[str]:
    s = user_text or ""
    syn = []
    if ("학생성공" in s) or ("성공지원" in s) or ("학생지원" in s):
        syn += ["학생성공", "학생지원", "학생지원팀"]
    if "컴퓨터공학" in s and "소프트웨어" not in s:
        syn += ["컴퓨터소프트웨어공학과","컴퓨터소프트웨어"]
    return syn


def expand_policy_synonyms(user_text: str) -> list[str]:
    t = user_text or ""
    syn = []
    if "복학" in t or "휴복학" in t or "휴·복학" in t or "휴 학" in t:
        syn += ["복학", "휴학", "휴복학", "휴·복학", "휴학/복학"]
    if "휴학" in t:
        for w in ["휴학", "복학", "휴복학", "휴·복학", "휴학/복학"]:
            if w not in syn: syn.append(w)
    if "학적" in t:
        syn += ["학적", "학적변동", "휴학", "복학", "재입학", "자퇴", "전과"]
    return syn


def expand_dorm_synonyms(user_text: str) -> list[str]:
    t = user_text or ""
    syn = []
    if any(k in t for k in ["기숙사","생활관","학생생활관","사생","입사","퇴사","입실","퇴실","생활관비"]):
        syn += ["기숙사","생활관","학생생활관","입사","퇴사","입사신청","생활관비","생활관 안내","생활관 규정"]
    return syn


def ensure_layout_unknown(url: str) -> str:
    if not url: return url
    if "layout=" in url: return url
    return (url + ("&layout=unknown" if "?" in url else "?layout=unknown"))


def _clean_url(u: str) -> str:
    return (u or "").rstrip(").,]〉」』>…")


def _phone_near_alias(text: str, aliases: set[str]) -> Optional[str]:
    phones = list(PHONE_PAT_KR.finditer(text or ""))
    if not phones:
        return None
    best = None
    best_dist = 10**9
    t = text or ""
    for m in phones:
        p_start = m.start()
        for a in aliases or []:
            idx = t.find(a)
            if idx == -1:
                continue
            dist = abs(p_start - idx)
            if dist < best_dist:
                best_dist = dist
                best = "-".join(m.groups())
    if best:
        return best
    return "-".join(phones[0].groups())

def _hint_matches_line(label: str, body: str, hint: dict | None) -> bool:
    if not hint:
        return False
    aliases = set(hint.get("aliases") or [])
    if not aliases:
        return False
    return any((a in (label or "")) or (a in (body or "")) for a in aliases)

def _fallback_phone_from_sub_answer(user_text: str, sub_answer: str, hint: dict | None = None) -> Optional[str]:
    if not sub_answer:
        return None

    if hint:
        for mline in LINE_PAT.finditer(sub_answer):
            label = (mline.group("label") or "").strip()
            body = (mline.group("body") or "").strip()
            if not _hint_matches_line(label, body, hint):
                continue
            mph = TEL_FIRST_PAT.search(body) or PHONE_PAT_KR.search(body)
            if mph:
                return "-".join(mph.groups())

        p = _phone_near_alias(sub_answer, set(hint.get("aliases") or []))
        if p:
            return p
        return None

    m = TEL_FIRST_PAT.search(sub_answer) or PHONE_PAT_KR.search(sub_answer)
    if m:
        return "-".join(m.groups())
    return None


def looks_like_schedule(text: str) -> bool:
    t = (text or "")
    if any(k in t for k in SCHEDULE_HINTS_BASE):
        return True
    if "언제" in t and any(k in t for k in ("중간","기말","시험","고사","수강","등록","성적","개강","종강","졸업식","학위수여식")):
        return True
    return False


def call_oss(messages: List[Dict[str, str]], **kwargs) -> str:
    if not any(m.get("role") == "system" for m in messages):
        messages = [{"role":"system","content":
            "Reasoning: low\n한국어로 단 한 문장으로만 답하라.\n연락처/전화/번호/문의 요청이 없는 한 전화번호나 이메일, URL을 임의로 만들지 말고 포함하지 마라.\n모르면 모른다고 답하라."
        }] + messages
    try:
        r = client.chat.completions.create(
            model=OSS_MODEL,
            messages=messages,
            temperature=kwargs.get("temperature", 0.3),
            max_tokens=kwargs.get("max_tokens", 64),
        )
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return ""


def call_submodel(user_text: str) -> str:
    base = ""
    try:
        res = build_answer(user_text, top_k=12)
        base = (res or {}).get("answer", "").strip()
    except Exception:
        base = ""

    try:
        sched = schedule_search(user_text, top_k=8)
    except Exception:
        sched = ""

    if sched and base:
        return sched + "\n" + base
    return sched or base


def scrub_non_contact(text: str) -> str:
    if not text:
        return ""
    t = URL_PAT.sub("", text)
    t = PHONE_GUARD_PAT.sub("", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t if t else "좋아요, 무엇을 이야기해 볼까요?"


def _parse_bullets_and_pick(user_text: str, sub_answer: str, hint: dict | None = None) -> tuple[str|None, str|None, str|None]:
    tokens_raw = re.findall(r"[가-힣A-Za-z0-9]{2,}", user_text or "")
    tokens = [t for t in tokens_raw if t not in STOP_TOKENS]
    syns = expand_synonyms(user_text)
    dept_intent = bool(re.search(r"(학과|학부|과|담당자\s*연락처)", user_text or "")) or (hint is not None)
    cand = []

    for m in LINE_PAT.finditer(sub_answer or ""):
        label = (m.group("label") or "").strip()
        body  = (m.group("body")  or "").strip()

        url_m = SRC_URL_PAT.search(body) or SRC_URL_DOT_PAT.search(body)
        url = _clean_url(url_m.group("url").strip()) if url_m else ""

        if not url:
            any_urls = URL_PAT.findall(body)
            if any_urls:
                picked = next((u for u in any_urls if GOOD_URL_RE.search(u)), any_urls[0])
                url = _clean_url(picked)

        ph_m = TEL_FIRST_PAT.search(body) or PHONE_PAT_KR.search(body)
        phone = "-".join(ph_m.groups()) if ph_m else None

        s = 0.0
        if GOOD_URL_RE.search(url):  s += 2.0
        if STAFF_URL_RE.search(url): s += 1.0
        if BAD_URL_RE.search(url):   s -= 3.0

        if any(t in label for t in tokens): s += 2.0
        if dept_intent:
            if UNIT_LIKE_LABEL_RE.search(label): s += 2.0
            if NON_DEPT_LABEL_RE.search(label):  s -= 2.0
            if GENERIC_LABEL_RE.search(label):   s -= 2.0

        if syns:
            if any(x in label for x in syns): s += 8.0
            if any(x in body  for x in syns): s += 4.0
            if NON_TARGET_TEAM_PAT.search(label): s -= 6.0

        if hint:
            if any(a in label for a in hint["aliases"]) or any(a in body for a in hint["aliases"]):
                s += 8.0
            else:
                s -= 6.0
            if hint["path_base"] and hint["path_base"] in url:
                s += 8.0
            if any((a in label) for a in (ALL_DEPT_ALIASES - set(hint["aliases"]))):
                s -= 4.0
            if any((a in body) for a in (ALL_DEPT_ALIASES - set(hint["aliases"]))):
                s -= 4.0
            if GENERIC_LABEL_RE.search(label):
                if any(a in body for a in hint["aliases"]):
                    s += 8.0
                else:
                    s -= 10.0

        if CONTACT_WORD_RE.search(body): s += 0.5
        if not phone and not url: s -= 2.0

        cand.append((s, label, phone, url, body))

    if not cand:
        return (None, None, None)

    roots = extract_roots(user_text)
    if roots:
        cand_root = [c for c in cand if any((r and (r in c[1] or r in c[4])) for r in roots)]
        if cand_root:
            cand = cand_root

    if hint:
        cand_hint = [c for c in cand if any(a in c[1] or a in c[4] for a in hint["aliases"])]
        if cand_hint:
            cand = cand_hint

    cand.sort(key=lambda x: x[0], reverse=True)
    best = cand[0]
    _, label, phone, url, body = best

    if hint:
        label_has_hint = any(a in (label or "") for a in hint["aliases"])
        body_has_hint  = any(a in (body  or "") for a in hint["aliases"])
        if not (label_has_hint or body_has_hint):
            strict = [c for c in cand if any(a in c[1] or a in c[4] for a in hint["aliases"])]
            if strict:
                strict.sort(key=lambda x: x[0], reverse=True)
                _, label, phone, url, body = strict[0]
                phone = phone or _phone_near_alias(body, hint["aliases"])
            else:
                try:
                    label = DEPT_MAP[hint['canon']]['name']
                except Exception:
                    pass
                if not url and hint.get("path_base"):
                    url = next((c[3] for c in cand if c[3] and hint["path_base"] in c[3]), None)
                phone = None

    return (label, phone, url or None)


def one_sentence_from_sub_answer(user_text: str, sub_answer: str) -> tuple[str, Optional[str]]:
    if not sub_answer:
        return ("요청하신 정보를 찾지 못했습니다.", None)
    hint = detect_dept_hint(user_text)
    label, phone, url = _parse_bullets_and_pick(user_text, sub_answer, hint)
    contact_intent = bool(CONTACT_INTENT_RE.search(user_text or ""))
    if contact_intent and not phone:
        phone = _fallback_phone_from_sub_answer(user_text, sub_answer, hint)

    url = ensure_layout_unknown(url) if url else None

    if contact_intent and hint and label and not _hint_matches_line(label, "", hint):
        label = None

    if label and phone:
        return (f"{label} 전화번호는 {phone}입니다.", url)
    if contact_intent and phone:
        return (f"요청하신 부서 담당자 전화번호는 {phone}입니다.", url)
    if contact_intent and not phone:
        return ("담당자 연락처를 바로 찾지 못했습니다. 학과(또는 부서) 풀네임으로 다시 입력해 주세요.", url)
    if label and url:
        return (f"{label} 정보는 {url}에서 확인할 수 있습니다.", url)

    first = sub_answer.strip().splitlines()[0].lstrip("- ").strip()
    text = first if first.endswith(("다.","요.")) else (first + "." if first else "요청하신 정보를 찾지 못했습니다.")
    return (text, url)


def one_sentence_topic(user_text: str, sub_answer: str) -> tuple[str, Optional[str]]:
    if not sub_answer:
        return (f"‘{user_text}’ 관련 정보는 {ORG_HOMEPAGE_LABEL}에서 확인할 수 있습니다.", ORG_HOMEPAGE_URL)
    items = list(TITLE_URL_PAT.finditer(sub_answer))
    if not items:
        first = sub_answer.strip().splitlines()[0].lstrip("- ").strip()
        txt = first if first.endswith(("다.","요.")) else (first + "." if first else f"‘{user_text}’ 관련 정보는 {ORG_HOMEPAGE_LABEL}에서 확인할 수 있습니다.")
        return (txt, None)

    hint = detect_dept_hint(user_text)
    q = re.sub(r"\s+", "", user_text)
    candidates = []
    for m in items:
        title = (m.group("title") or "").strip()
        url   = (m.group("url") or "").strip()
        if SAFETY_URL_RE.search(url) or SAFETY_TITLE_RE.search(title):
            continue
        s = 0
        if GOOD_URL_RE.search(url): s += 3
        if BAD_URL_RE.search(url):  s -= 4
        title_compact = re.sub(r"\s+", "", title)
        if title_compact == q: s += 6
        if INTRO_WORD_RE.search(title): s += 3
        if CONTACT_WORD_RE.search(title): s -= 5
        toks = [t for t in re.findall(r"[가-힣A-Za-z0-9]{2,}", user_text)]
        if any(t in title for t in toks): s += 1
        if hint:
            if hint["path_base"] and hint["path_base"] in url: s += 8
            if any(a in title for a in hint["aliases"]): s += 4
        candidates.append((s, title, url))

    if not candidates:
        return (f"‘{user_text}’ 관련 정보는 {ORG_HOMEPAGE_LABEL}에서 확인할 수 있습니다.", ORG_HOMEPAGE_URL)

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_s, best_title, best_url = candidates[0]
    if best_s <= 0:
        return (f"‘{user_text}’ 관련 정보는 {ORG_HOMEPAGE_LABEL}에서 확인할 수 있습니다.", ORG_HOMEPAGE_URL)
    best_url = ensure_layout_unknown(best_url)
    return (f"‘{user_text}’ 관련 정보는 ‘{best_title}’ 페이지({best_url})에서 확인할 수 있습니다.", best_url)


def one_sentence_policy(user_text: str, sub_answer: str) -> tuple[str, Optional[str]]:
    if not sub_answer:
        return (f"‘{user_text}’ 관련 정보는 {ORG_HOMEPAGE_LABEL}의 학사안내에서 확인할 수 있습니다.", ORG_HOMEPAGE_URL)
    items = list(TITLE_URL_PAT.finditer(sub_answer))
    if not items:
        first = sub_answer.strip().splitlines()[0].lstrip("- ").strip()
        txt = first if first.endswith(("다.","요.")) else (first + "." if first else f"‘{user_text}’ 관련 정보는 {ORG_HOMEPAGE_LABEL}의 학사안내에서 확인할 수 있습니다.")
        return (txt, None)
    syns = expand_policy_synonyms(user_text)
    toks = [t for t in re.findall(r"[가-힣A-Za-z0-9·]{2,}", user_text)]
    candidates = []
    for m in items:
        title = (m.group("title") or "").strip()
        url   = (m.group("url") or "").strip()
        if SAFETY_URL_RE.search(url) or SAFETY_TITLE_RE.search(title):
            continue
        s = 0.0
        if GOOD_URL_RE.search(url): s += 3.5
        if BAD_URL_RE.search(url):  s -= 8.0
        if any(k in title for k in syns): s += 7.0
        if any(k in title for k in ("휴학","복학","휴·복학","휴복학","학적","학사안내")): s += 3.0
        if any(t in title for t in toks): s += 1.0
        if CONTACT_WORD_RE.search(title): s -= 3.0
        candidates.append((s, title, url))
    if not candidates:
        return (f"‘{user_text}’ 관련 정보는 {ORG_HOMEPAGE_LABEL}의 학사안내에서 확인할 수 있습니다.", ORG_HOMEPAGE_URL)
    candidates.sort(key=lambda x: x[0], reverse=True)
    good_only = [c for c in candidates if GOOD_URL_RE.search(c[2])]
    pool = good_only or candidates
    _, best_title, best_url = pool[0]
    best_url = ensure_layout_unknown(best_url)
    return (f"‘{user_text}’ 관련 공식 안내는 ‘{best_title}’ 페이지({best_url})에서 확인할 수 있습니다.", best_url)


def one_sentence_dorm(user_text: str, sub_answer: str) -> tuple[str, Optional[str]]:
    if not sub_answer:
        return (f"‘{user_text}’ 관련 정보는 {ORG_HOMEPAGE_LABEL}의 생활관 안내에서 확인할 수 있습니다.", ORG_HOMEPAGE_URL)
    items = list(TITLE_URL_PAT.finditer(sub_answer))
    if not items:
        first = sub_answer.strip().splitlines()[0].lstrip("- ").strip()
        txt = first if first.endswith(("다.","요.")) else (first + "." if first else f"‘{user_text}’ 관련 정보는 {ORG_HOMEPAGE_LABEL}의 생활관 안내에서 확인할 수 있습니다.")
        return (txt, None)
    syns = expand_dorm_synonyms(user_text)
    toks = [t for t in re.findall(r"[가-힣A-Za-z0-9·]{2,}", user_text)]
    candidates = []
    for m in items:
        title = (m.group("title") or "").strip()
        url   = (m.group("url") or "").strip()
        if SAFETY_URL_RE.search(url) or SAFETY_TITLE_RE.search(title):
            continue
        s = 0.0
        if GOOD_URL_RE.search(url): s += 3.5
        if BAD_URL_RE.search(url):  s -= 8.0
        if any(k in title for k in syns): s += 8.0
        if any(k in title for k in ("학생생활관","생활관","기숙사","입사","생활관비","생활관 안내")): s += 3.0
        if any(t in title for t in toks): s += 1.0
        if CONTACT_WORD_RE.search(title): s -= 3.0
        candidates.append((s, title, url))
    if not candidates:
        return (f"‘{user_text}’ 관련 정보는 {ORG_HOMEPAGE_LABEL}의 생활관 안내에서 확인할 수 있습니다.", ORG_HOMEPAGE_URL)
    candidates.sort(key=lambda x: x[0], reverse=True)
    good_only = [c for c in candidates if GOOD_URL_RE.search(c[2])]
    pool = good_only or candidates
    _, best_title, best_url = pool[0]
    best_url = ensure_layout_unknown(best_url)
    return (f"‘{user_text}’ 관련 공식 안내는 ‘{best_title}’ 페이지({best_url})에서 확인할 수 있습니다.", best_url)


def one_sentence_grad(user_text: str, sub_answer: str) -> tuple[str, Optional[str]]:
    T2_TOTAL, T2_MAJOR = 75, 52
    T3_TOTAL, T3_MAJOR = 110, 78
    ask_2 = bool(re.search(r"2\s*년제", user_text))
    ask_3 = bool(re.search(r"3\s*년제", user_text))
    if ask_2 and not ask_3:
        return (f"2년제 졸업학점은 총 {T2_TOTAL}학점(전공최저이수 {T2_MAJOR}학점)이며 자세한 내용은 {GRAD_PAGE_URL}에서 확인할 수 있습니다.", GRAD_PAGE_URL)
    if ask_3 and not ask_2:
        return (f"3년제 졸업학점은 총 {T3_TOTAL}학점(전공최저이수 {T3_MAJOR}학점)이며 자세한 내용은 {GRAD_PAGE_URL}에서 확인할 수 있습니다.", GRAD_PAGE_URL)
    return (f"졸업학점은 2년제 총 {T2_TOTAL}학점(전공최저 {T2_MAJOR}학점), 3년제 총 {T3_TOTAL}학점(전공최저 {T3_MAJOR}학점)이며 자세한 내용은 {GRAD_PAGE_URL}에서 확인할 수 있습니다.", GRAD_PAGE_URL)


class ChatReq(BaseModel):
    text: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    engine: Optional[str] = None


def extract_user_text(req: ChatReq) -> str:
    if req.text:
        return req.text.strip()
    return next(
        ((m.get("content") or "").strip()
        for m in reversed(req.messages or [])
        if m.get("role") == "user" and (m.get("content") or "").strip()),
        ""
    )


def ensure_messages(req: ChatReq, user_text: str) -> List[Dict[str, str]]:
    if req.messages and isinstance(req.messages, list) and len(req.messages) > 0:
        return req.messages
    return [{"role": "user", "content": user_text}]



@oss_router.post("/chatbot")
def chat(req: ChatReq, username: str = Depends(verify_jwt_token)):
    user_text = extract_user_text(req)
    messages_for_oss = ensure_messages(req, user_text)
    GOVERNANCE_REMOVE_RE = re.compile(r"(없애|폐지|해체)\s*(시키|하는\s*법)?")
    GOVERNANCE_TARGET_RE = re.compile(r"(총학생회|대의원회)")
    if GOVERNANCE_REMOVE_RE.search(user_text) and GOVERNANCE_TARGET_RE.search(user_text):
        return {"engine": "guard", "text": "해당 요청은 도움을 드리기 어려워요; 공식 절차나 문의는 학생자치기구 페이지의 연락처를 이용해 주세요."}

    mode = req.engine or decide_mode(user_text)

    if mode == "greet":
        return {"engine":"greet", "text":"안녕하세요, 무엇을 도와드릴까요?"}
    if mode == "whoami":
        return {"engine":"whoami", "text":f"{SERVICE_NAME}의 {BOT_NAME}입니다."}
    if mode == "relation":
        return {"engine":"relation", "text":f"우리는 {ORG_NAME} 정보를 함께 해결하는 대화 파트너이고, 저는 {SERVICE_NAME}의 {BOT_NAME}입니다."}
    if mode == "fast":
        clarify = dept_clarification_message(user_text)
        if clarify:
            return {"engine":"fast", "text": clarify}

        sched_only = schedule_search(user_text, top_k=8)
        if sched_only:
            pretty = render_chatty_schedule(sched_only, user_text)
            return {"engine":"fast", "text": pretty}

        sub = call_submodel(user_text)
        text, url = one_sentence_from_sub_answer(user_text, sub)
        resp = {"engine":"fast", "text": text}
        if url: resp["url"] = url
        return resp

    if mode == "policy":
        sub = call_submodel(user_text)
        text, url = one_sentence_policy(user_text, sub)
        resp = {"engine":"policy", "text": text}
        if url: resp["url"] = url
        return resp

    if mode == "dorm":
        sub = call_submodel(user_text)
        text, url = one_sentence_dorm(user_text, sub)
        resp = {"engine":"dorm", "text": text}
        if url: resp["url"] = url
        return resp

    if mode == "grad":
        sub = call_submodel(user_text)
        text, url = one_sentence_grad(user_text, sub)
        resp = {"engine":"grad", "text": text}
        if url: resp["url"] = url
        return resp

    if mode == "topic":
        sub = call_submodel(user_text)
        text, url = one_sentence_topic(user_text, sub)
        resp = {"engine":"topic", "text": text}
        if url: resp["url"] = url
        return resp

    if mode == "oss":
        out = call_oss(messages_for_oss)
        if not any(k in user_text for k in ("연락처","전화","번호","문의")) and not any(k in user_text for k in COUNCIL_KWS):
            out = scrub_non_contact(out)

        if not out:
            if looks_like_schedule(user_text):
                sched_only = schedule_search(user_text, top_k=8)
                if sched_only:
                    return {"engine":"fast", "text": render_chatty_schedule(sched_only, user_text)}

            if len(user_text) <= 2 or GREETING_RE.search(user_text):
                out = "안녕하세요, 무엇을 도와드릴까요?"
            else:
                sub = call_submodel(user_text)
                if looks_like_schedule(user_text) and sub:
                    return {"engine":"fast", "text": render_chatty_schedule(sub, user_text)}
                text, _ = one_sentence_topic(user_text, sub)
                out = text if looks_like_topic(user_text) else "잘 이해하지 못했어요. 다시 질문해주세요."
        return {"engine":"oss", "text": out}

    sub = call_submodel(user_text)
    fused = call_oss(
        [{"role":"system","content":
            "Reasoning: low\n다음 <context>의 사실만 사용해 한국어로 한 문장으로만 답하라. 불릿/개행 금지. 임의의 전화번호/URL을 생성하지 말라."
        },
        {"role":"user","content": user_text + "\n\n<context>\n" + (sub or "") + "\n</context>"}],
        max_tokens=64, temperature=0.2
    )
    if not fused:
        text, _ = one_sentence_topic(user_text, sub)
        fused = text if looks_like_topic(user_text) else "좋아요, 무엇을 이야기해 볼까요?"
    return {"engine": f"{mode}", "text": fused}


def decide_mode(user_text: str) -> str:
    text = (user_text or "").strip()
    if not text:
        return "greet"
    if GREETING_RE.search(text):
        return "greet"
    if is_whoami(text):
        return "whoami"
    if is_relation(text):
        return "relation"
    if CEREMONY_RE.search(text):
        return "fast"
    if "학사일정" in text or "학사 일정" in text or looks_like_schedule(text):
        return "fast"

    contact_kws = ("연락처","전화","번호","상담","문의")
    if any(k in text for k in COUNCIL_KWS): return "fast"
    if any(k in text for k in contact_kws): return "fast"

    policy_kws = ("복학","휴학","휴·복학","휴복학","학적","학적변동","자퇴","전과","재입학",
                "제증명","수강신청","등록","장학","성적","계절학기","학점포기","학기포기")
    if any(k in text for k in policy_kws): return "policy"

    dorm_kws = ("기숙사","생활관","학생생활관","사생","입사","생활관비","입사신청","퇴사")
    if any(k in text for k in dorm_kws): return "dorm"

    if GRAD_POLICY_RE.search(text):
        return "grad"

    if looks_like_topic(text): return "topic"
    return "oss"


router = oss_router
