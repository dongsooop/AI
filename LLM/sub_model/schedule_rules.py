from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ScheduleTagRule:
    tag: str
    title_pattern: re.Pattern[str]
    query_keywords: tuple[str, ...]


SCHEDULE_TAG_RULES = (
    ScheduleTagRule("MIDTERM", re.compile(r"중간"), ("중간", "중간고사")),
    ScheduleTagRule("FINAL", re.compile(r"기말|학기말\s*고사"), ("기말", "기말고사", "학기말", "학기말고사")),
    ScheduleTagRule("REGISTRATION", re.compile(r"수강\s*신청|예비\s*수강"), ("수강",)),
    ScheduleTagRule("ADD_DROP", re.compile(r"수강\s*정정|정정"), ("정정",)),
    ScheduleTagRule("GRADE", re.compile(r"성적|성적공시|성적입력|성적열람|성적정정"), ("성적",)),
    ScheduleTagRule("TUITION", re.compile(r"등록금|등록\s*기간|등록"), ("등록", "등록금")),
    ScheduleTagRule("MAKEUP", re.compile(r"보강"), ("보강",)),
    ScheduleTagRule("SEMESTER_START", re.compile(r"개강"), ("개강",)),
    ScheduleTagRule("SEMESTER_END", re.compile(r"종강"), ("종강",)),
    ScheduleTagRule("HOLIDAY", re.compile(r"휴일|공휴일|추석|설날|현충일|한글날|크리스마스"), ("휴일", "공휴")),
    ScheduleTagRule("CLASSDAY", re.compile(r"수업일수|수업"), ("수업", "수업일수")),
    ScheduleTagRule("COMMENCEMENT", re.compile(r"졸업식|학위수여식"), ("졸업식", "종업식", "학위수여식")),
)

SCHEDULE_BASE_HINTS = (
    "학사일정",
    "언제",
    "시험",
    "고사",
    "이번주",
    "다음주",
    "이번달",
    "다음달",
)

SCHEDULE_INTENT_HINTS = tuple(
    dict.fromkeys(
        [
            *SCHEDULE_BASE_HINTS,
            *(keyword for rule in SCHEDULE_TAG_RULES for keyword in rule.query_keywords),
        ]
    )
)


def tags_for_title(title: str) -> set[str]:
    source = str(title)
    return {rule.tag for rule in SCHEDULE_TAG_RULES if rule.title_pattern.search(source)}


def tags_for_query(query: str) -> set[str]:
    source = (query or "").lower()
    tags: set[str] = set()
    for rule in SCHEDULE_TAG_RULES:
        if any(keyword in source for keyword in rule.query_keywords):
            tags.add(rule.tag)
    if not tags and any(keyword in source for keyword in ("시험", "고사")):
        tags.update({"MIDTERM", "FINAL"})
    if "수강" in source and "정정" in source:
        tags.discard("REGISTRATION")
        tags.add("ADD_DROP")
    return tags
