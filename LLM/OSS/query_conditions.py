"""Conditions required for personal academic calendar queries."""
import re
from LLM.OSS.modes import is_academic_procedure_query

UNVERIFIED_CALENDAR_MESSAGE = "입력한 학기·대상 조건이 명시된 일정을 자료에서 확인하지 못했어요. 해당 학기 공지를 확인해 주세요."


def calendar_conditions(query):
    text = re.sub(r"\s+", "", query or "")
    if is_academic_procedure_query(query):
        return None
    if re.search(r"방법|절차|어떻게|규정|연락처|전화|담당|증명|전체|모든", text):
        return None
    if not re.search(r"언제|기간|일정|날짜|[12]학기|[1-4]학년|20\d{2}년", text):
        return None
    if '수강신청' in text and not re.search(r"정정|계절", text):
        topic, needs = '수강신청', ('year', 'semester', 'grade', 'student')
    elif re.search(r"등록금.*납부|등록기간|등록일정", text) and not re.search(r"분할|졸업|재입학|계절", text):
        topic, needs = '등록', ('year', 'semester', 'student')
    elif re.search(r"성적열람|성적이의신청", text):
        topic, needs = '성적열람', ('year', 'semester')
    else:
        return None
    patterns = {'year': r'(20\d{2})(?:년|학년도)', 'semester': r'([12])학기',
                'grade': r'([1-4])학년(?!도)',
                'student': r'(신입생|재학생|복학생|편입생|전공심화)'}
    values = {k: re.findall(p, text) for k, p in patterns.items()}
    if re.search(r'아니|제외|말고', text) or any(len(set(v)) > 1 for v in values.values()):
        return topic, ['조건을 하나로 명확하게 입력해 주세요'], {}
    scope = {k: v[0] for k, v in values.items() if v}
    labels = {'year': '대상 연도', 'semester': '학기', 'grade': '학년',
              'student': '학생 유형(신입생·재학생·복학생·편입생·전공심화)'}
    return topic, [labels[k] for k in needs if k not in scope], scope


def matches_calendar_conditions(title, scope):
    text = re.sub(r'\s+', '', title)
    # Unlabelled dates cannot establish which semester the event applies to.
    if scope.get('semester') and f"{scope['semester']}학기" not in text:
        return False
    if scope.get('grade'):
        grades = set(re.findall(r'([1-4])학년', text))
        for group in re.findall(r'([1-4](?:[,·ㆍ][1-4])+)학년', text):
            grades.update(re.findall(r'[1-4]', group))
        if not grades or scope['grade'] not in grades:
            return False
    if scope.get('student'):
        students = set(re.findall(r'신입생|재학생|복학생|편입생|전공심화', text))
        if not students or scope['student'] not in students:
            return False
    return True
