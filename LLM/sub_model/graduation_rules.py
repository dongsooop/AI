"""Source-derived graduation credit rules with explicit applicability.

Only recognized table layouts are parsed. Unknown layouts must not fall back to
selecting a nearby (or largest) credit number from flattened page text.
"""
import re


YEAR_LABELS = {"graduation": "졸업대상", "admission": "입학", "transfer": "편입"}
STUDENT_LABELS = {"general": "일반학생", "transfer": "편입학생", "change": "전과학생", "free": "자유전공 전공배정학생"}


def _rule(student, program):
    return dict(student=student, program=program, year_basis="", year_from=0,
                year_to=0, department="", grade=0, semester=0, total=None,
                major=None, note="")


def valid_rule(rule):
    if not isinstance(rule, dict) or not _rule('general', 0).keys() <= rule.keys():
        return False
    if rule['student'] not in STUDENT_LABELS or rule['year_basis'] not in ('', *YEAR_LABELS):
        return False
    if any(type(rule[k]) is not int or rule[k] < 0 for k in
           ('program', 'year_from', 'year_to', 'grade', 'semester')):
        return False
    if any(rule[k] is not None and (type(rule[k]) is not int or not 0 < rule[k] <= 300)
           for k in ('total', 'major')):
        return False
    if rule['total'] is not None and rule['major'] is not None and rule['major'] > rule['total']:
        return False
    return all(isinstance(rule[k], str) for k in ('department', 'note'))


def render_rule(rule):
    labels = [STUDENT_LABELS[rule["student"]]]
    if rule["department"]:
        labels.append(rule["department"])
    if rule["program"]:
        labels.append(f"{rule['program']}년제")
    if rule["year_basis"]:
        year = str(rule["year_from"])
        year += "년 이후" if not rule["year_to"] else "년"
        labels.append(f"{year} {YEAR_LABELS[rule['year_basis']]} 기준")
    if rule["grade"]:
        labels.append(f"{rule['grade']}학년 {rule['semester']}학기 {STUDENT_LABELS[rule['student']][:2]}")
    facts = []
    if rule["total"] is not None:
        facts.append(f"총 졸업학점 {rule['total']}학점")
    if rule["major"] is not None:
        facts.append(f"전공최저이수 {rule['major']}학점")
    if rule["note"]:
        facts.append(rule["note"])
    return " / ".join(labels) + ": " + ", ".join(facts)


def parse_graduation_rules(title, content):
    if not re.search(r"졸업|전문학사", title or ""):
        return []
    text = re.sub(r"\s+", " ", content or "").strip()
    rules = []

    def add(student, program, **values):
        rule = _rule(student, program)
        rule.update(values)
        rules.append(rule)

    # Actual column headings, not the duplicated accessibility description.
    header = (r"구분\s*졸업\s*학점\s*(\d{4})년\s*2월\s*및\s*8월\s*졸업대상자\s*"
              r"(\d{4})년\s*2월\s*이후\s*졸업대상자\s*")
    totals = re.search(header + r"2년제\s+(\d+)\s+(\d+)\s+3년제\s+(\d+)\s+(\d+)", text)
    major_section = re.search(r"전공최저이수\s*학점(.*)", text)
    majors = re.search(header + r"2년제\s+(\d+)\s+(\d+)\s+3년제\s+(\[별표\s*1\]|\d+)\s+(\d+)",
                       major_section[1]) if major_section else None
    if totals:
        for program, offset in ((2, 3), (3, 5)):
            for column in (0, 1):
                year = int(totals.group(1 + column))
                major_value = majors.group(offset + column) if majors and majors.group(1, 2) == totals.group(1, 2) else ""
                note = "2·8월 졸업대상자" if column == 0 else "2월 이후 졸업대상자"
                if major_value.startswith("["):
                    note += "; 전공최저이수학점은 입학연도·학과별 별표1 확인"
                add("general", program, year_basis="graduation", year_from=year,
                    year_to=year if column == 0 else 0, total=int(totals.group(offset + column)),
                    major=int(major_value) if major_value.isdigit() else None, note=note)

    # Split appendix sections before interpreting any row-spanned conditions.
    sections = re.split(r"\d\s*\[별표\s*([123])\]", text)
    appendices = dict(zip(sections[1::2], sections[2::2]))
    historical = appendices.get("1", "")
    groups = re.split(r"(\d{4}(?:,\s*\d{4})*)학년도\s*입학자", historical)
    for years, body in zip(groups[1::2], groups[2::2]):
        for match in re.finditer(r"(\d+)학점\s*(.*?)(?=\d+학점|$)", body):
            for department in match.group(2).split(","):
                department = department.strip()
                if not re.fullmatch(r"[가-힣A-Za-z·]+과", department):
                    continue
                for year in re.findall(r"\d{4}", years):
                    add("general", 0, department=department, year_basis="admission",
                        year_from=int(year), year_to=int(year), major=int(match.group(1)))

    transfers = re.split(r"(\d{4})\s*학년도\s*(이후)?", appendices.get("2", ""))
    for idx in range(1, len(transfers), 3):
        year, onward, body = transfers[idx:idx + 3]
        program = 0
        pattern = (r"(?:(?P<program>[23])년\s*)?(?P<grade>[23])학년\s*(?P<semester>[12])학기\s*"
                   r"편입학생\s*(?P<major>\d+)학점\s*\d+(?:~\d+)?학점\s*(?P<total>\d+)학점")
        for match in re.finditer(pattern, body):
            program = int(match['program']) if match['program'] else program
            if program:
                add("transfer", program, year_basis="transfer", year_from=int(year),
                    year_to=0 if onward else int(year), grade=int(match['grade']),
                    semester=int(match['semester']), total=int(match['total']), major=int(match['major']),
                    note="편입학 학기부터 적용되는 교육과정·전공필수 과목 확인")

    changes, _, free = appendices.get("3", "").partition("자유전공학과 전공배정학생")
    program, total = 0, None
    pattern = (r"(?:(?P<program>[23])년\s*)?(?P<grade>[12])학년\s*(?P<semester>[12])학기\s*"
               r"전과학생\s*(?P<major>\d+)학점(?:\s*(?:수강학점\s*)?(?P<total>\d+)학점)?")
    for match in re.finditer(pattern, changes):
        if match['program'] and int(match['program']) != program:
            program, total = int(match['program']), None
        if match['total']:
            total = int(match['total'])
        if program and total is not None:
            add("change", program, grade=int(match['grade']), semester=int(match['semester']),
                total=total, major=int(match['major']), note="전과 학기부터 취득한 전공학점 기준")
    match = re.search(r"([23])년\s*자유전공학과\s*전공배정학생\s*(\d+)학점\s*(\d+)학점", free)
    if match:
        add("free", int(match[1]), total=int(match[3]), major=int(match[2]),
            note="전공 배정 학기부터 취득한 전공학점 기준")
    return rules


def query_scope(query):
    text = re.sub(r"\s+", "", query or "")
    student = next((value for key, value in (("자유전공", "free"), ("편입", "transfer"),
                    ("전과", "change"), ("일반", "general"), ("신입학", "general")) if key in text), "")
    program = re.search(r"([234])년제", text)
    grade = re.search(r"([1234])학년([12])학기", text)
    graduation_month = re.search(r"20\d{2}년(\d{1,2})월졸업", text)
    years = {}
    for basis, keyword in (("admission", "입학"), ("graduation", "졸업"), ("transfer", "편입")):
        match = re.search(r"(20\d{2})(?:학년도|년)(?:[12]?\d월)?" + keyword, text)
        if not match:
            match = re.search(keyword + r"(?:예정)?(?:연도)?(?:는)?(20\d{2})년?", text)
        if match:
            years[basis] = int(match[1])
    admission = re.search(r"(?<!\d)(\d{2}|20\d{2})학번", text)
    if admission:
        years['admission'] = int(admission[1]) + (2000 if len(admission[1]) == 2 else 0)
    return dict(student=student, program=int(program[1]) if program else 0,
                grade=int(grade[1]) if grade else 0, semester=int(grade[2]) if grade else 0,
                graduation_month=int(graduation_month[1]) if graduation_month else 0,
                years=years, major=bool(re.search(r"전공(?:최저|학점|이수)|최저이수", text)))


def answer_graduation_scope(query, records):
    """Return an answer or a targeted clarification, never a best-effort number."""
    scope = query_scope(query)
    compact = re.sub(r"\s+", "", query)
    urls = {r['url'] for r in records if r.get('url')}
    url = next(iter(urls)) if len(urls) == 1 else ""

    def respond(message, clarification=True):
        result = {"answer": message, "needs_clarification": clarification}
        if url:
            result.update(url=url)
            result['answer'] += f"\n(출처: {url})"
        return result

    if re.search(r"아니|제외|말고|않", compact) or sum(bool(re.search(pattern, compact)) for pattern in
            (r"일반학생|신입학", r"편입", r"전과", r"자유전공")) > 1:
        return respond("졸업학점 적용 대상이 명확하지 않아요. 일반학생/편입/전과/자유전공 중 해당하는 구분을 하나로 적어 주세요.")
    if scope['grade'] and re.search(r"현재|지금", compact):
        return respond("졸업학점 확인에는 현재 학년이 아니라 편입·전과 당시 학년과 학기가 필요해요. 당시 조건을 적어 주세요.")
    if re.search(r"복학|복학생|재입학|전공심화|학사학위|(?<!전문)학사(?:과정)?졸업|외국인|산업체|계약학과|학력자", compact):
        return respond("해당 졸업학점은 본인에게 적용되는 교육과정 확인이 필요해요. 학과와 적용 교육과정 연도를 확인해 주세요.")
    if not records:
        return respond("현재 자료에서 적용 조건이 확인되는 졸업학점 기준을 찾지 못했어요. 학과 사무실에 적용 교육과정을 확인해 주세요.")

    candidates = records
    for key in ('student', 'program', 'grade', 'semester'):
        if scope[key]:
            candidates = [r for r in candidates if r[key] == scope[key] or (key == 'program' and not r[key])]
    if scope['major']:
        candidates = [r for r in candidates if r['major'] is not None]
    else:
        candidates = [r for r in candidates if r['total'] is not None]
    departments = {r['department'] for r in records if r['department']}
    department = next((name for name in sorted(departments, key=len, reverse=True) if name in compact), "")
    if department:
        candidates = [r for r in candidates if not r['department'] or r['department'] == department]
    for basis, year in scope['years'].items():
        candidates = [r for r in candidates if r['year_basis'] != basis or
                      (year >= r['year_from'] and (not r['year_to'] or year <= r['year_to']))]
    month = scope['graduation_month']
    if month:
        year = scope['years'].get('graduation', 0)
        candidates = [r for r in candidates if r['year_basis'] != 'graduation' or
                      (month in (2, 8) if r['year_to'] else 1 <= month <= 12 and (year > r['year_from'] or month >= 2))]
    if scope['major'] and (scope['student'] == 'general' or {r['student'] for r in candidates} == {'general'}):
        preferred_basis = 'admission' if 'admission' in scope['years'] else 'graduation'
        preferred = [r for r in candidates if r['year_basis'] == preferred_basis]
        if preferred:
            candidates = preferred
    if not candidates:
        return respond("입력한 조건에 맞는 졸업학점 기준을 자료에서 확인하지 못했어요. 학제·학생 구분·적용 연도를 확인해 주세요.")
    if scope['years'] and all(not r['year_basis'] for r in candidates):
        return respond("해당 졸업학점 표에는 적용 연도가 명시되어 있지 않아 입력한 연도에 적용되는지 확정하기 어려워요. 학과에 적용 교육과정을 확인해 주세요.")

    missing = []
    if not scope['student']:
        return respond("졸업학점은 학생 구분에 따라 기준이 달라요. 일반학생/편입/전과/자유전공 여부와 학제를 함께 알려주세요. 예: 일반학생 3년제 2026년 졸업학점")
    if not department and any(r['department'] for r in candidates):
        missing.append("입학 당시 학과명")
    if not scope['program'] and any(r['program'] for r in candidates):
        missing.append("2년제/3년제")
    for basis in sorted({r['year_basis'] for r in candidates if r['year_basis']}):
        if basis not in scope['years']:
            missing.append(YEAR_LABELS[basis] + " 연도")
    if any(r['grade'] for r in candidates) and not scope['grade']:
        missing.append("편입·전과한 학년과 학기")
    if missing:
        return respond("졸업학점 기준 확인에 필요한 정보: " + ", ".join(missing) + ". 조건을 함께 입력해 주세요. 입학연도와 졸업대상 연도는 서로 다른 기준이에요.")
    # Multiple rules for the same supplied conditions require verification.
    unique = {render_rule(r): r for r in candidates}
    if len(unique) != 1:
        return respond("입력한 조건에 해당하는 졸업학점 기준이 여러 개라 확정하기 어려워요. 학과와 적용 교육과정을 확인해 주세요.")
    selected = next(iter(unique.values()))
    url = selected.get('url', '')
    return respond(render_rule(selected) + "\n학과별 필수과목 등 추가 졸업요건은 원문에서 함께 확인해 주세요.", False)
