"""Conservative purpose selection for scholarship, housing and leave queries."""
import re


NEUTRAL = r"(?:은|이|에대해(?:서)?|관련(?:해서)?)?(?:정보|안내)?(?:좀)?(?:알려(?:줘|주세요|줄래요?)|궁금(?:해요?|합니다)|해줘|해주세요)?"
RULES = (
    ("장학금", "policy", "4794", r"장학금?",
     "장학금은 교내 장학금과 국가·교외 장학금 등 종류를 구분해 확인해야 해요. "
     "'교내 장학금 안내', '국가장학금 신청 조건', '교외 장학금 안내' 중 궁금한 내용을 입력해 주세요.",
     r"(?:신청)?(?:조건|자격|대상|금액|얼마(?:야|예요)?|신청방법|신청기간)",
     "어떤 장학금이 궁금한가요? 장학금 이름이나 교내·국가·교외 구분과 대상 학기를 함께 알려주세요. "
     "장학금 종류를 확인하기 전에는 자격이나 지급액을 확정하기 어려워요."),
    ("기숙사", "dorm", "4841", r"기숙사|생활관|학생생활관",
     "기숙사는 학생들이 함께 생활하는 주거 공간이며, 시설과 실 유형은 기숙사 안내에서 확인할 수 있어요. "
     "'기숙사 입사 신청 방법', '기숙사 비용', '기숙사 시설', '기숙사 위치' 중 궁금한 내용을 입력해 주세요.",
     r"(?:비|비용|요금|금액|비얼마(?:야|예요)?|얼마(?:야|예요)?|입사(?:신청)?(?:방법|조건|기간)|신청(?:방법|조건|기간))",
     "기숙사 이용 시기(학기 중·방학)와 희망하는 기숙사·실 유형을 알려주세요. "
     "비용과 입사 조건은 해당 모집 공고를 확인해야 하며, 시설 소개만으로 확정하기 어려워요."),
    ("휴학", "policy", "4784", r"휴학",
     "휴학은 일반휴학과 군입대휴학 등 종류에 따라 절차와 구비서류가 달라요. "
     "'일반휴학 신청 방법', '군휴학 필요서류', '휴학연기 안내' 중 궁금한 내용을 입력해 주세요.",
     r"(?:신청)?(?:방법|절차|조건|기간|서류|필요서류|준비서류|등록금|등록금처리)",
     "일반휴학·군휴학·휴학연기 중 어떤 경우인가요? 희망 학기와 휴학 종류를 알려주세요. "
     "등록금 처리 질문이라면 등록금 납부 여부도 함께 알려주세요."),
)


def has_scholarship_name(query):
    # Presence of a name is distinct from verifying its eligibility in sources.
    # Do not restrict users to a small allowlist of scholarship programs.
    generic = {'교내', '교외', '학교', '대학', '국가', '해당', '어떤', '무슨', '장학'}
    text = re.sub(r'20\d{2}(?:학년도|년)|[12]학기|이번학기|다음학기', ' ', query or '')
    names = re.findall(r'([가-힣]+)\s*장학금', text)
    return any(name not in generic for name in names) or bool(re.search(r'국가\s*장학금', text))


def support_guidance(query):
    compact = re.sub(r"\s+", "", query or "").rstrip("?!？！.。~…")
    for topic, mode, page, aliases, overview, details, clarification in RULES:
        if re.fullmatch(f"(?:{aliases}){NEUTRAL}", compact):
            return topic, mode, page, overview
        if re.fullmatch(f"(?:{aliases})(?:{details}){NEUTRAL}", compact):
            return topic, mode, page, clarification
    if re.search(r'연락처|전화|담당|규정|말고|아니|제외', compact):
        return None
    year = bool(re.search(r'20\d{2}(?:년|학년도)', compact))
    term = bool(re.search(r'[12]학기|여름방학|겨울방학', compact))
    missing = []
    if '장학' in compact and re.search(r'조건|자격|금액|얼마|신청기간', compact):
        topic, mode, page = '장학금', 'policy', '4794'
        if not has_scholarship_name(query):
            missing.append('구체적인 장학금 이름')
        if not year:
            missing.append('대상 연도')
        if not term:
            missing.append('대상 학기')
    elif re.search(r'기숙사|생활관', compact) and re.search(r'비용|요금|기숙사비|생활관비|얼마|입사.*(?:조건|기간)', compact):
        topic, mode, page = '기숙사', 'dorm', '4841'
        if not year:
            missing.append('이용 연도')
        if not term:
            missing.append('이용 학기 또는 여름·겨울방학')
        if not re.search(r'[1-4]인실|[A-E]형|아파트형', compact):
            missing.append('기숙사·실 유형')
    elif '휴학' in compact and re.search(r'기간|등록금', compact):
        topic, mode, page = '휴학', 'policy', '4784'
        if not re.search(r'일반휴학|군(?:입대)?휴학|휴학연기', compact):
            missing.append('휴학 종류')
        if not year:
            missing.append('대상 연도')
        if not term:
            missing.append('대상 학기')
        if '등록금' in compact and not re.search(r'완납|미납|분납|납부했|납부안|납부하지않', compact):
            missing.append('등록금 납부 여부(완납·미납·분납)')
    else:
        return None
    if missing:
        return topic, mode, page, f"{topic} 기준 확인에 필요한 정보: {', '.join(missing)}. 이미 적은 조건과 함께 질문해 주세요."
    return None
