import re

from core.settings import get_settings


settings = get_settings()

DASH_CHARS_PATTERN = r"\-\u2010\u2011\u2012\u2013\u2014\u2212\uFE58\uFE63\uFF0D"
HANGUL_TOKEN_PATTERN = r"[가-힣A-Za-z0-9]{2,}"
HANGUL_DOT_TOKEN_PATTERN = r"[가-힣A-Za-z0-9·]{2,}"

URL_PATTERN = r"https?://\S+"
PHONE_KR_PATTERN = r"(0(?:2|[3-9]\d))\D{0,2}(\d{3,4})\D{0,2}(\d{4})"
PHONE_ANY_AREA_PATTERN = r"(0\d{1,2})[^\d]{0,4}(\d{3,4})[^\d]{0,4}(\d{4})"
PHONE_GUARD_PATTERN = rf"(?:\+?\d[\d\s\-{DASH_CHARS_PATTERN}]{{6,}}\d)"
EMAIL_PATTERN = r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
TEL_FIRST_PATTERN = (
    r"(?:TEL\.?|Tel\.?|T\.?|전화(?:번호)?|연락처)\s*[:.\-]?\s*"
    + PHONE_KR_PATTERN
)

GOOD_URL_PATTERN = r"/subview\.do($|\?)"
BOARD_URL_PATTERN = r"/bbs/|artclView\.do"
BAD_URL_PATTERN = r"(?:/bbs/|artclView\.do|combBbs)"
DEPT_HOME_URL_PATTERN = r"/dmu/\d{4}/subview\.do$"
HOME_LIKE_URL_PATTERN = r"/subview\.do|/intro|/dmu/\d+/subview"
STAFF_DEFAULT_URL_PATTERN = r"/dmu/4408/subview\.do$"
STAFF_URL_PATTERN = rf"(?:{settings.staff_url_pattern})|(?:{STAFF_DEFAULT_URL_PATTERN})"
STAFF_TITLE_PATTERN = r"(?:교직원\s*검색|전화번호\s*안내)"

UNIT_SUFFIX_PATTERN = r"(학부|학과|팀|센터|처|단|부|원|본부)$"
UNIT_QUERY_SUFFIX_PATTERN = r"(학부|과|팀|센터|처|단|부|원|본부)$"
OTHER_UNIT_TOKEN_PATTERN = r"([가-힣A-Za-z0-9]{2,}(?:학부|학과|팀|센터|처|단|부|원|본부))"
DEPT_ROOT_SUFFIX_PATTERN = r"(공학부|공학과|학부|학과|과)$"
UNIT_LIKE_LABEL_PATTERN = r"(학과|학부)"
NON_DEPT_LABEL_PATTERN = r"(팀|지원팀|실|센터|본부)$"
GENERIC_CONTACT_LABEL_PATTERN = r"(담당부|연락처|대표|대표번호)"
GENERIC_ORG_LABEL_PATTERN = r"(?:대표|교무|입학|장학|취업|총무|홍보|학생상담|콜센터)"
NON_TARGET_TEAM_PATTERN = r"(정보지원팀|전산실|시설관리팀|재무팀|홍보팀|홍보대사단)"

SAFETY_URL_PATTERN = r"(safety|lab|ehs|env|환경|안전)"
SAFETY_TITLE_PATTERN = r"(연구실|실험실|실습실|안전|환경안전|EHS)"
CONTACT_WORD_PATTERN = r"(연락처|전화|전화번호|문의)"
INTRO_WORD_PATTERN = r"(학부\s*소개|학과\s*소개|소개|안내|개요)"
DEPT_CONTACT_INTENT_PATTERN = r"(학과|학부|과|담당자\s*연락처)"

SCHED_LINE_PATTERN = (
    r"^\s*-\s*(?P<title>[^:]+):\s*(?P<s>\d{4}-\d{2}-\d{2})"
    r"(?:\s*~\s*(?P<e>\d{4}-\d{2}-\d{2}))?$"
)
LINE_PATTERN = r"^\s*-\s*(?P<label>[^:：]+)[:：]\s*(?P<body>.+)$"
SRC_URL_PATTERN = r"\(출처:\s*(?P<url>https?://[^)]+)\)"
SRC_URL_DOT_PATTERN = r"\(출처:\s*[^·\)]*·\s*(?P<url>https?://[^)]+)\)"
TITLE_URL_PATTERN = r"^\s*-\s*(?P<title>[^:]+):\s*(?P<url>\S+)"

URL_RE = re.compile(URL_PATTERN)
PHONE_KR_RE = re.compile(PHONE_KR_PATTERN)
PHONE_ANY_AREA_RE = re.compile(PHONE_ANY_AREA_PATTERN)
PHONE_GUARD_RE = re.compile(PHONE_GUARD_PATTERN)
EMAIL_RE = re.compile(EMAIL_PATTERN)
TEL_FIRST_RE = re.compile(TEL_FIRST_PATTERN)

GOOD_URL_RE = re.compile(GOOD_URL_PATTERN, re.I)
STAFF_URL_RE = re.compile(STAFF_URL_PATTERN, re.I)
STAFF_TITLE_RE = re.compile(STAFF_TITLE_PATTERN, re.I)
BAD_URL_RE = re.compile(BAD_URL_PATTERN, re.I)
BOARD_URL_RE = re.compile(BOARD_URL_PATTERN, re.I)

UNIT_SUFFIX_RE = re.compile(UNIT_SUFFIX_PATTERN)
UNIT_QUERY_SUFFIX_RE = re.compile(UNIT_QUERY_SUFFIX_PATTERN)
OTHER_UNIT_TOKEN_RE = re.compile(OTHER_UNIT_TOKEN_PATTERN)
DEPT_ROOT_SUFFIX_RE = re.compile(DEPT_ROOT_SUFFIX_PATTERN)
UNIT_LIKE_LABEL_RE = re.compile(UNIT_LIKE_LABEL_PATTERN)
NON_DEPT_LABEL_RE = re.compile(NON_DEPT_LABEL_PATTERN)
GENERIC_CONTACT_LABEL_RE = re.compile(GENERIC_CONTACT_LABEL_PATTERN)
GENERIC_ORG_LABEL_RE = re.compile(GENERIC_ORG_LABEL_PATTERN)
NON_TARGET_TEAM_RE = re.compile(NON_TARGET_TEAM_PATTERN)

SAFETY_URL_RE = re.compile(SAFETY_URL_PATTERN, re.I)
SAFETY_TITLE_RE = re.compile(SAFETY_TITLE_PATTERN, re.I)
CONTACT_WORD_RE = re.compile(CONTACT_WORD_PATTERN, re.I)
INTRO_WORD_RE = re.compile(INTRO_WORD_PATTERN, re.I)
DEPT_CONTACT_INTENT_RE = re.compile(DEPT_CONTACT_INTENT_PATTERN)

SCHED_LINE_RE = re.compile(SCHED_LINE_PATTERN, re.MULTILINE)
LINE_RE = re.compile(LINE_PATTERN, re.MULTILINE)
SRC_URL_RE = re.compile(SRC_URL_PATTERN)
SRC_URL_DOT_RE = re.compile(SRC_URL_DOT_PATTERN)
TITLE_URL_RE = re.compile(TITLE_URL_PATTERN, re.MULTILINE)
