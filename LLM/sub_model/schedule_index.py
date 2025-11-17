import os, re, calendar, datetime as dt
import pandas as pd
from pathlib import Path

CSV_PATH = os.getenv("SCHEDULE_CSV_PATH")

TAG_RULES = [
    (re.compile(r"중간"), "MIDTERM"),
    (re.compile(r"기말"), "FINAL"),
    (re.compile(r"수강\s*신청|예비\s*수강"), "REGISTRATION"),
    (re.compile(r"수강\s*정정|정정"), "ADD_DROP"),
    (re.compile(r"성적|성적공시|성적입력|성적열람|성적정정"), "GRADE"),
    (re.compile(r"등록금|등록\s*기간|등록"), "TUITION"),
    (re.compile(r"보강"), "MAKEUP"),
    (re.compile(r"개강"), "SEMESTER_START"),
    (re.compile(r"종강"), "SEMESTER_END"),
    (re.compile(r"휴일|공휴일|추석|설날|현충일|한글날|크리스마스"), "HOLIDAY"),
    (re.compile(r"수업일수|수업"), "CLASSDAY"),
    (re.compile(r"졸업식|학위수여식"), "COMMENCEMENT"),
]

INTENT_HINTS = (
    "학사일정","중간","기말","수강","정정","성적",
    "등록","보강","개강","종강","휴일","언제","졸업식","학위수여식"
)

def _academic_year(today=None):
    today = today or dt.date.today()
    return today.year if today.month >= 3 else (today.year - 1)


RANGE_SPLIT_RE       = re.compile(r"\s*[~∼〜]\s*")
WEEKDAY_PARENS_RE    = re.compile(r"\([^)]*\)")
NON_NUM_SEP_CLEAN_RE = re.compile(r"[^\d./\-]")

def _safe_month(m: int) -> int:
    return max(1, min(int(m), 12))

def _safe_date(y: int, m: int, d: int) -> dt.date:
    m = _safe_month(m)
    last = calendar.monthrange(int(y), m)[1]
    d = max(1, min(int(d), last))
    return dt.date(int(y), m, d)

def _parse_mmdd_token(tok: str, default_m: int) -> tuple[int, int]:
    t = WEEKDAY_PARENS_RE.sub("", tok or "")
    t = NON_NUM_SEP_CLEAN_RE.sub("", t)
    t = t.replace("-", ".").replace("/", ".")
    t = re.sub(r"\.+", ".", t).strip(".")
    m = re.match(r"^(?:(\d{1,2})\.)?(\d{1,2})$", t)

    if m:
        mon = _safe_month(m.group(1)) if m.group(1) else _safe_month(default_m)
        day = int(m.group(2))
        return mon, day
    only_d = re.match(r"^(\d{1,2})$", t)

    if only_d:
        return _safe_month(default_m), int(only_d.group(1))
    
    raise ValueError(f"Unparsable date token: {tok!r}")

def _daterange_from_parts(y, m, day_str):
    parts = RANGE_SPLIT_RE.split(str(day_str or ""))
    if not parts or not parts[0].strip():
        raise ValueError("empty date field")

    sm, sd = _parse_mmdd_token(parts[0], default_m=int(m))
    if len(parts) >= 2 and parts[1].strip():
        em, ed = _parse_mmdd_token(parts[1], default_m=sm)
    else:
        em, ed = sm, sd

    start = _safe_date(int(y), int(sm), int(sd))
    end   = _safe_date(int(y), int(em), int(ed))

    if end < start:
        end = _safe_date(int(y) + 1, int(em), int(ed))
    return start, end

def _tag_title(title):
    tags = set()
    t = str(title)
    for rx, tag in TAG_RULES:
        if rx.search(t):
            tags.add(tag)
    return tags

def _week_range(date_obj):
    start = date_obj - dt.timedelta(days=date_obj.weekday())
    end = start + dt.timedelta(days=6)
    return start, end

def _parse_time_constraints(q, today=None):
    today = today or dt.date.today()
    q = re.sub(r"\s+", "", q)

    m = re.search(r"(?:(\d{4})년)?(\d{1,2})월", q)
    if m:
        yy = int(m.group(1)) if m.group(1) else today.year
        mm = int(m.group(2))
        s = dt.date(yy, mm, 1)
        e = dt.date(yy, 12, 31) if mm == 12 else (dt.date(yy, mm+1, 1) - dt.timedelta(days=1))
        return s, e

    if "이번주" in q:
        return _week_range(today)
    if "다음주" in q:
        return _week_range(today + dt.timedelta(days=7))

    if "이번달" in q:
        s = dt.date(today.year, today.month, 1)
        e = dt.date(today.year, 12, 31) if today.month == 12 else (dt.date(today.year, today.month+1, 1) - dt.timedelta(days=1))
        return s, e
    if "다음달" in q:
        ny, nm = (today.year + (1 if today.month == 12 else 0), 1 if today.month == 12 else today.month+1)
        s = dt.date(ny, nm, 1)
        e = dt.date(ny, 12, 31) if nm == 12 else (dt.date(ny, nm+1, 1) - dt.timedelta(days=1))
        return s, e
    return None

def _load_df():
    if not CSV_PATH:
        raise ValueError("SCHEDULE_CSV_PATH 환경변수가 비어있습니다.")
    csv_path = Path(CSV_PATH).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"학사일정 CSV가 존재하지 않습니다: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")

    for c in ["학년도", "년도", "월", "날짜", "일정명"]:
        if c not in df.columns:
            df[c] = ""

    df["학년도"] = pd.to_numeric(df["학년도"], errors="coerce")
    df["년도"]   = pd.to_numeric(df["년도"],   errors="coerce")
    df["월"]     = pd.to_numeric(df["월"],     errors="coerce")

    starts, ends, tags = [], [], []
    for y, m, d, title in zip(df["년도"], df["월"], df["날짜"], df["일정명"]):
        try:
            if pd.isna(y) or pd.isna(m) or not str(d).strip():
                starts.append(pd.NaT); ends.append(pd.NaT); tags.append(set()); continue
            s, e = _daterange_from_parts(int(y), int(m), str(d))
        except Exception as ex:
            raise ValueError(f"잘못된 날짜 행 -> 년도:{y}, 월:{m}, 날짜:{d}, 일정명:{title} :: {ex}")
        starts.append(pd.Timestamp(s)); ends.append(pd.Timestamp(e)); tags.append(_tag_title(title))
    df["start_date"] = starts
    df["end_date"]   = ends
    df["tags"]       = tags
    return df

_DF = _load_df()

def _looks_like_schedule_query(q):
    return any(k in (q or "") for k in INTENT_HINTS)

def schedule_search(query: str, top_k=8, today=None):
    if not _looks_like_schedule_query(query):
        return ""

    today = today or dt.date.today()
    ay = _academic_year(today)

    over = os.getenv("ACADEMIC_YEAR_OVERRIDE")
    if over and over.isdigit():
        ay = int(over)

    df = _DF.copy()
    df1 = df[df["학년도"] == ay]
    if df1.empty:
        df1 = df

    tr = _parse_time_constraints(query, today=today)
    if tr:
        s, e = tr
        df1 = df1[(df1["start_date"] <= pd.Timestamp(e)) & (df1["end_date"] >= pd.Timestamp(s))]

    ql = (query or "").lower()
    tag_need = set()

    if "중간" in ql: tag_need.add("MIDTERM")
    if "기말" in ql: tag_need.add("FINAL")
    if "수강" in ql and "정정" in ql: tag_need.add("ADD_DROP")
    elif "수강" in ql: tag_need.add("REGISTRATION")
    if "정정" in ql: tag_need.add("ADD_DROP")
    if "성적" in ql: tag_need.add("GRADE")
    if "등록" in ql: tag_need.add("TUITION")
    if "보강" in ql: tag_need.add("MAKEUP")
    if "개강" in ql: tag_need.add("SEMESTER_START")
    if "종강" in ql: tag_need.add("SEMESTER_END")
    if "휴일" in ql or "공휴" in ql: tag_need.add("HOLIDAY")
    if ("졸업식" in ql) or ("학위수여식" in ql): tag_need.add("COMMENCEMENT")

    if tag_need:
        df1 = df1[df1["tags"].apply(lambda s: bool(s & tag_need))]

    today_ts = pd.Timestamp(today)
    df1 = df1.dropna(subset=["start_date"]).copy()
    df1["_is_future"] = (df1["start_date"] >= today_ts)
    df1 = df1.sort_values(by=["_is_future", "start_date"], ascending=[False, True])

    if df1.empty:
        return ""

    def _fmt(r):
        s = r["start_date"].date().isoformat()
        e = r["end_date"].date().isoformat()
        when = s if s == e else f"{s} ~ {e}"
        return f"- {r['일정명']}: {when}"

    lines = [_fmt(rv) for _, rv in df1.head(top_k).iterrows()]
    return "\n".join(lines)