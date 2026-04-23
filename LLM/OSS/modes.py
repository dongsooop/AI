import re

from core.settings import get_settings


settings = get_settings()

RELATIVE_DATE_KEYWORDS = frozenset({"오늘", "내일", "이번주", "다음주", "이번달", "다음달"})

GREETING_RE = re.compile(r"^\s*(안녕|안녕하세요|하이|hello|hi)\b", re.I)
WHOAMI_EN = re.compile(r"\b(who\s+are\s+you|what\s+are\s+you|your\s+name)\b", re.I)
WHOAMI_KO = re.compile(r"(무슨\s*(챗봇|봇)|뭐\s*하는\s*(챗봇|봇)|역할|무엇을\s*할\s*수|기능|소개해\s*줘)")
RELATION_RE = re.compile(
    r"(?:우리\s*(?:무슨\s*)?(?:관계|사이)"
    r"|우린\s*무슨\s*(?:관계|사이)"
    r"|너와\s*나"
    r"|무슨\s*사이"
    r"|(?:사이|관계)\s*(?:야|냐)(?:\?|$))"
    )

UNIT_SUFFIX_RE = re.compile(r"(학부|학과|과|전공|대학|대학원|본부|센터|팀|처|단|부|원)$")
CEREMONY_RE = re.compile(r"(졸업식|학위수여식)")
GRAD_POLICY_RE = re.compile(r"(졸업학점|이수학점|졸업요건|전공최저|최저이수|학위수여(?!식)|졸업(?!식))")
CONTACT_INTENT_RE = re.compile(r"(연락처|전화|전화번호|문의|상담|담당자)")
GOVERNANCE_REMOVE_RE = re.compile(r"(없애|폐지|해체)\s*(시키|하는\s*법)?")
GOVERNANCE_TARGET_RE = re.compile(r"(총학생회|대의원회)")

TOPIC_KWS = ("학부", "학과", "전공", "소개", "안내", "위치", "학사", "졸업")
COUNCIL_KWS = (
    "총학생회", "총학생", "총학", "대의원회", "대의원실",
    "학생회", "학생대표기구", "홍보대사단", "동아리연합회", "동양학보", "학보사", "총학생실",
)
SCHEDULE_HINTS_BASE = (
    "학사일정", "학사 일정", "중간", "중간고사", "기말", "기말고사",
    "수강", "정정", "성적", "등록", "보강", "개강", "종강",
    "휴일", "공휴", "시험", "고사", "이번주", "다음주", "이번달", "다음달",
    "졸업식", "학위수여식",
)
RULE_BOOK_KWS = ("규정", "규정집", "학칙", "준칙", "회칙", "규약", "세칙", "강령", "운영규칙", "선발 규칙", "선발규칙")

PRONOUNS = ("너", "넌", "니", "네가", "당신", "챗봇", "봇", "ai", "에이아이", *settings.bot_aliases)
QWORDS = ("누구", "정체", "이름", "뭐야", "뭐냐", "무엇", "뉘신", "누구야", "누구냐")


def is_whoami(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if WHOAMI_EN.search(stripped) or WHOAMI_KO.search(stripped):
        return True
    compact = re.sub(r"\s+", "", stripped).lower()
    return any(pronoun in compact for pronoun in PRONOUNS) and any(question in compact for question in QWORDS)


def is_relation(text: str) -> bool:
    if not text:
        return False
    return bool(RELATION_RE.search(text))


def looks_like_topic(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if any(keyword in stripped for keyword in TOPIC_KWS):
        return True
    return bool(UNIT_SUFFIX_RE.search(stripped)) and len(stripped) <= 12


def looks_like_schedule(text: str) -> bool:
    source = text or ""
    if any(keyword in source for keyword in SCHEDULE_HINTS_BASE):
        return True
    return "언제" in source and any(
        keyword in source
        for keyword in ("중간", "기말", "시험", "고사", "수강", "등록", "성적", "개강", "종강", "졸업식", "학위수여식")
    )


def decide_mode(user_text: str) -> str:
    text = (user_text or "").strip()
    compact = re.sub(r"\s+", "", text)
    if not text:
        return "greet"
    if GREETING_RE.search(text):
        return "greet"
    if is_whoami(text):
        return "whoami"
    if is_relation(text):
        return "relation"
    if any(keyword in text for keyword in RULE_BOOK_KWS):
        return "rule_book"
    if CEREMONY_RE.search(text):
        return "fast"
    if "학사일정" in text or "학사 일정" in text or "학사일정" in compact or looks_like_schedule(text):
        return "fast"

    contact_keywords = ("연락처", "전화", "번호", "상담", "문의")
    if any(keyword in text for keyword in COUNCIL_KWS):
        return "fast"
    if any(keyword in text for keyword in contact_keywords) or CONTACT_INTENT_RE.search(text):
        return "fast"

    policy_keywords = (
        "복학", "휴학", "휴·복학", "휴복학", "학적", "학적변동", "자퇴", "전과", "재입학",
        "제증명", "수강신청", "등록", "장학", "성적", "계절학기", "학점포기", "학기포기",
    )
    if any(keyword in text for keyword in policy_keywords):
        return "policy"

    dorm_keywords = ("기숙사", "생활관", "학생생활관", "사생", "입사", "생활관비", "입사신청", "퇴사")
    if any(keyword in text for keyword in dorm_keywords):
        return "dorm"

    if GRAD_POLICY_RE.search(text):
        return "grad"
    if looks_like_topic(text):
        return "topic"
    return "oss"
