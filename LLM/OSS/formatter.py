import datetime as dt
import re
from pathlib import Path
from typing import Optional

from core.settings import get_settings
from LLM.OSS.modes import CONTACT_INTENT_RE
from LLM.patterns import (
    BAD_URL_RE,
    CONTACT_WORD_RE,
    DEPT_CONTACT_INTENT_RE,
    DEPT_ROOT_SUFFIX_RE,
    GENERIC_CONTACT_LABEL_RE,
    GOOD_URL_RE,
    HANGUL_DOT_TOKEN_PATTERN,
    HANGUL_TOKEN_PATTERN,
    INTRO_WORD_RE,
    LINE_RE,
    NON_DEPT_LABEL_RE,
    NON_TARGET_TEAM_RE,
    PHONE_GUARD_RE,
    PHONE_KR_RE,
    SAFETY_TITLE_RE,
    SAFETY_URL_RE,
    SCHED_LINE_RE,
    SRC_URL_DOT_RE,
    SRC_URL_RE,
    STAFF_URL_RE,
    TEL_FIRST_RE,
    TITLE_URL_RE,
    UNIT_LIKE_LABEL_RE,
    URL_RE,
)
from LLM.OSS.postprocess.context import PostProcessContext
from LLM.OSS.postprocess.synonym_table import DORM_SYNONYM_RULES, POLICY_SYNONYM_RULES


settings = get_settings()

ORG_HOMEPAGE_LABEL = settings.org_homepage_label
ORG_HOMEPAGE_URL = settings.org_homepage_url
GRAD_PAGE_URL = settings.grad_page_url

PHONE_GUARD_PAT = PHONE_GUARD_RE
URL_PAT = URL_RE
PHONE_PAT_KR = PHONE_KR_RE
TEL_FIRST_PAT = TEL_FIRST_RE
LINE_PAT = LINE_RE
SRC_URL_PAT = SRC_URL_RE
SRC_URL_DOT_PAT = SRC_URL_DOT_RE
TITLE_URL_PAT = TITLE_URL_RE
GENERIC_LABEL_RE = GENERIC_CONTACT_LABEL_RE
NON_TARGET_TEAM_PAT = NON_TARGET_TEAM_RE

STOP_TOKENS = {"담당", "담당부", "전화", "전화번호", "연락처", "문의", "상담", "번호"}


def _resolve_repo_root() -> Path:
    return settings.repo_root


REPO_ROOT = _resolve_repo_root()


def _normalize_heading_from_query(user_text: str) -> str:
    text = re.sub(r"\s+", " ", (user_text or "").strip())
    text = re.sub(r"[?？!！]+$", "", text).strip()
    if not text:
        return "학사일정 안내"
    if text.endswith("일정") or text.endswith("안내"):
        return text
    return f"{text} 일정"


def render_chatty_schedule(bullets: str, user_text: str = "") -> str:
    weekdays = "월화수목금토일"
    lines = []
    for match in SCHED_LINE_RE.finditer(bullets or ""):
        title = match.group("title").strip()
        start = dt.date.fromisoformat(match.group("s"))
        end = dt.date.fromisoformat(match.group("e")) if match.group("e") else start
        if start == end:
            when = f"{start.isoformat()}({weekdays[start.weekday()]})"
        else:
            when = f"{start.isoformat()}({weekdays[start.weekday()]}) ~ {end.isoformat()}({weekdays[end.weekday()]})"
        lines.append(f"+ {title}: {when}")

    if not lines:
        return "관련 일정을 찾지 못했어요."
    return _normalize_heading_from_query(user_text) + "\n" + "\n".join(lines)


def _canon_unit(value: str) -> str:
    text = (value or "").strip()
    text = re.sub(r"\s+", "", text).replace("·", "").replace(".", "")
    return re.sub(r"(학부|학과|공학과|과)$", "", text)


def _dept_map_path() -> Path:
    return settings.resolved_dept_map_path


def _make_aliases(name: str) -> set[str]:
    raw = name.strip()
    variants = {
        raw,
        re.sub(r"학과$", "과", raw),
        re.sub(r"공학과$", "학과", raw),
        re.sub(r"공학부$", "학부", raw),
    }
    canon = _canon_unit(raw)
    root = re.sub(DEPT_ROOT_SUFFIX_RE, "", canon)
    if canon:
        variants |= {canon, canon + "학부", canon + "학과", canon + "공학과", canon + "과", canon + "전공"}
    if root and len(root) >= 2:
        variants |= {
            root, root + "학부", root + "학과", root + "공학과", root + "과", root + "전공",
            root + "정보", root + "소프트웨어", root + "SW",
        }
    return {variant for variant in variants if variant and isinstance(variant, str)}


def load_dept_map() -> tuple[dict[str, dict[str, str]], dict[str, set[str]]]:
    path = _dept_map_path()
    dept_map: dict[str, dict[str, str]] = {}
    aliases: dict[str, set[str]] = {}
    if not path.exists():
        return dept_map, aliases

    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                name, rest = line.split(",", 1)
                url_part = rest.split(",")[0].strip()
                base = url_part.split("?")[0]
                canon = _canon_unit(name)
                dept_map[canon] = {"name": name, "path_base": base}
                aliases[canon] = _make_aliases(name)
            except Exception:
                continue
    return dept_map, aliases


DEPT_MAP, DEPT_ALIAS = load_dept_map()
ALL_DEPT_ALIASES: set[str] = set()
for alias_set in DEPT_ALIAS.values():
    ALL_DEPT_ALIASES |= set(alias_set)


def extract_roots(user_text: str) -> set[str]:
    roots: set[str] = set()
    for token in re.findall(HANGUL_DOT_TOKEN_PATTERN, user_text or ""):
        base = re.sub(r"공$", "", _canon_unit(token))
        if len(base) >= 2:
            roots.add(base)
            roots.add(base[:2])
        if len(base) >= 3:
            roots.add(base[:3])
    return roots


def detect_dept_hint(text: str) -> Optional[dict[str, object]]:
    tokens = re.findall(HANGUL_DOT_TOKEN_PATTERN, text or "")
    for token in tokens:
        canon = _canon_unit(token)
        if canon in DEPT_MAP:
            data = DEPT_MAP[canon]
            return {"canon": canon, "aliases": DEPT_ALIAS.get(canon, {data["name"]}), "path_base": data["path_base"]}

    roots = extract_roots(text)
    if roots:
        best = None
        best_len = -1
        for canon_key in DEPT_MAP:
            for root in roots:
                if root and root in canon_key and len(root) > best_len:
                    best = canon_key
                    best_len = len(root)
        if best:
            data = DEPT_MAP[best]
            return {"canon": best, "aliases": DEPT_ALIAS.get(best, {data["name"]}), "path_base": data["path_base"]}
    return None


def _extract_dept_query_core(text: str) -> str:
    text = (text or "").strip()
    text = CONTACT_INTENT_RE.sub(" ", text)
    text = re.sub(r"(알려줘|알려\s*주세요|어디|뭐야|뭐지|좀)", " ", text)
    text = re.sub(r"[^가-힣A-Za-z0-9·\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _find_dept_candidates(core: str, limit: int = 5) -> list[str]:
    if not core:
        return []
    tokens = [token for token in re.findall(HANGUL_DOT_TOKEN_PATTERN, core) if token not in STOP_TOKENS]
    if not tokens:
        return []

    scored: list[tuple[float, str]] = []
    for canon, meta in DEPT_MAP.items():
        aliases = DEPT_ALIAS.get(canon, {meta["name"]})
        score = 0.0
        for token in tokens:
            if any((token in alias) or (alias in token) for alias in aliases):
                score += 3.0
            elif token in canon:
                score += 2.0
            elif len(token) >= 2 and token[:2] in canon:
                score += 1.0
        if score > 0:
            scored.append((score, meta["name"]))

    scored.sort(key=lambda item: (-item[0], item[1]))
    names = [name for _, name in scored]
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
            if len(candidates) == 1:
                return f"혹시 '{candidates[0]}'을(를) 찾으시나요? 학과 풀네임으로 다시 입력해 주세요."
            return f"학과명이 축약되어 후보가 여러 개예요. 풀네임으로 입력해 주세요. 예: {picks}"
        return "학과명이 축약되어 정확한 매칭이 어려워요. 학과 풀네임으로 입력해 주세요. 예: 컴퓨터소프트웨어공학과 담당자 연락처"
    return None


def expand_synonyms(user_text: str) -> list[str]:
    synonyms: list[str] = []
    source = user_text or ""
    if ("학생성공" in source) or ("성공지원" in source) or ("학생지원" in source):
        synonyms += ["학생성공", "학생지원", "학생지원팀"]

    hint = detect_dept_hint(source)
    if hint:
        aliases = list(hint.get("aliases") or [])
        synonyms += sorted(set(aliases), key=len, reverse=True)[:30]
    return list(dict.fromkeys([value for value in synonyms if value and isinstance(value, str)]))


def expand_policy_synonyms(user_text: str) -> list[str]:
    text = user_text or ""
    synonyms: list[str] = []
    for rule in POLICY_SYNONYM_RULES:
        if any(trigger in text for trigger in rule["triggers"]):
            for word in rule["synonyms"]:
                if word not in synonyms:
                    synonyms.append(word)
    return synonyms


def expand_dorm_synonyms(user_text: str) -> list[str]:
    text = user_text or ""
    synonyms: list[str] = []
    for rule in DORM_SYNONYM_RULES:
        if any(trigger in text for trigger in rule["triggers"]):
            for word in rule["synonyms"]:
                if word not in synonyms:
                    synonyms.append(word)
    return synonyms


def ensure_layout_unknown(url: str) -> str:
    if not url or "layout=" in url:
        return url
    return url + ("&layout=unknown" if "?" in url else "?layout=unknown")


def _clean_url(url: str) -> str:
    return (url or "").rstrip(").,]〉」』>…")


def _phone_near_alias(text: str, aliases: set[str]) -> Optional[str]:
    phones = list(PHONE_PAT_KR.finditer(text or ""))
    if not phones:
        return None
    best = None
    best_dist = 10**9
    source = text or ""
    for match in phones:
        for alias in aliases or []:
            index = source.find(alias)
            if index == -1:
                continue
            distance = abs(match.start() - index)
            if distance < best_dist:
                best_dist = distance
                best = "-".join(match.groups())
    return best or "-".join(phones[0].groups())


def _hint_matches_line(label: str, body: str, hint: Optional[dict[str, object]]) -> bool:
    if not hint:
        return False
    aliases = set(hint.get("aliases") or [])
    return bool(aliases) and any((alias in (label or "")) or (alias in (body or "")) for alias in aliases)


def _fallback_phone_from_sub_answer(user_text: str, sub_answer: str, hint: Optional[dict[str, object]] = None) -> Optional[str]:
    if not sub_answer:
        return None

    if hint:
        for match in LINE_PAT.finditer(sub_answer):
            label = (match.group("label") or "").strip()
            body = (match.group("body") or "").strip()
            if not _hint_matches_line(label, body, hint):
                continue
            phone_match = TEL_FIRST_PAT.search(body) or PHONE_PAT_KR.search(body)
            if phone_match:
                return "-".join(phone_match.groups())

        phone = _phone_near_alias(sub_answer, set(hint.get("aliases") or []))
        if phone:
            return phone
        return None

    phone_match = TEL_FIRST_PAT.search(sub_answer) or PHONE_PAT_KR.search(sub_answer)
    if phone_match:
        return "-".join(phone_match.groups())
    return None


def scrub_non_contact(text: str) -> str:
    if not text:
        return ""
    text = URL_PAT.sub("", text)
    text = PHONE_GUARD_PAT.sub("", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text if text else "좋아요, 무엇을 이야기해 볼까요?"


def _parse_bullets_and_pick(
    user_text: str,
    sub_answer: str,
    hint: Optional[dict[str, object]] = None,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    tokens_raw = re.findall(HANGUL_TOKEN_PATTERN, user_text or "")
    tokens = [token for token in tokens_raw if token not in STOP_TOKENS]
    synonyms = expand_synonyms(user_text)
    dept_intent = bool(DEPT_CONTACT_INTENT_RE.search(user_text or "")) or (hint is not None)
    candidates: list[tuple[float, str, Optional[str], str, str]] = []

    for match in LINE_PAT.finditer(sub_answer or ""):
        label = (match.group("label") or "").strip()
        body = (match.group("body") or "").strip()

        url_match = SRC_URL_PAT.search(body) or SRC_URL_DOT_PAT.search(body)
        url = _clean_url(url_match.group("url").strip()) if url_match else ""
        if not url:
            urls = URL_PAT.findall(body)
            if urls:
                picked = next((item for item in urls if GOOD_URL_RE.search(item)), urls[0])
                url = _clean_url(picked)

        phone_match = TEL_FIRST_PAT.search(body) or PHONE_PAT_KR.search(body)
        phone = "-".join(phone_match.groups()) if phone_match else None

        score = 0.0
        if GOOD_URL_RE.search(url):
            score += 2.0
        if STAFF_URL_RE.search(url):
            score += 1.0
        if BAD_URL_RE.search(url):
            score -= 3.0
        if any(token in label for token in tokens):
            score += 2.0

        if dept_intent:
            if UNIT_LIKE_LABEL_RE.search(label):
                score += 2.0
            if NON_DEPT_LABEL_RE.search(label):
                score -= 2.0
            if GENERIC_LABEL_RE.search(label):
                score -= 2.0

        if synonyms:
            if any(value in label for value in synonyms):
                score += 8.0
            if any(value in body for value in synonyms):
                score += 4.0
            if NON_TARGET_TEAM_PAT.search(label):
                score -= 6.0

        if hint:
            aliases = set(hint["aliases"])
            if any(alias in label for alias in aliases) or any(alias in body for alias in aliases):
                score += 8.0
            else:
                score -= 6.0
            if hint["path_base"] and hint["path_base"] in url:
                score += 8.0
            if any(alias in label for alias in (ALL_DEPT_ALIASES - aliases)):
                score -= 4.0
            if any(alias in body for alias in (ALL_DEPT_ALIASES - aliases)):
                score -= 4.0
            if GENERIC_LABEL_RE.search(label):
                if any(alias in body for alias in aliases):
                    score += 8.0
                else:
                    score -= 10.0

        if CONTACT_WORD_RE.search(body):
            score += 0.5
        if not phone and not url:
            score -= 2.0

        candidates.append((score, label, phone, url, body))

    if not candidates:
        return None, None, None

    roots = extract_roots(user_text)
    if roots:
        filtered = [candidate for candidate in candidates if any(root and (root in candidate[1] or root in candidate[4]) for root in roots)]
        if filtered:
            candidates = filtered

    if hint:
        filtered = [candidate for candidate in candidates if any(alias in candidate[1] or alias in candidate[4] for alias in hint["aliases"])]
        if filtered:
            candidates = filtered

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, label, phone, url, body = candidates[0]

    if hint:
        label_has_hint = any(alias in (label or "") for alias in hint["aliases"])
        body_has_hint = any(alias in (body or "") for alias in hint["aliases"])
        if not (label_has_hint or body_has_hint):
            strict = [candidate for candidate in candidates if any(alias in candidate[1] or alias in candidate[4] for alias in hint["aliases"])]
            if strict:
                strict.sort(key=lambda item: item[0], reverse=True)
                _, label, phone, url, body = strict[0]
                phone = phone or _phone_near_alias(body, set(hint["aliases"]))
            else:
                try:
                    label = DEPT_MAP[hint["canon"]]["name"]
                except Exception:
                    pass
                if not url and hint.get("path_base"):
                    url = next((candidate[3] for candidate in candidates if candidate[3] and hint["path_base"] in candidate[3]), None)
                phone = None

    return label, phone, url or None


def one_sentence_from_sub_answer(user_text: str, sub_answer: str) -> tuple[str, Optional[str]]:
    if not sub_answer:
        return "요청하신 정보를 찾지 못했습니다.", None
    ctx = build_contact_context(user_text, sub_answer)
    return render_contact_response(ctx)


def matches_hint_label(label: str, hint: Optional[dict[str, object]]) -> bool:
    return _hint_matches_line(label, "", hint)


def extract_contact_fields(
    user_text: str,
    sub_answer: str,
    hint: Optional[dict[str, object]] = None,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    return _parse_bullets_and_pick(user_text, sub_answer, hint)


def extract_fallback_phone(
    user_text: str,
    sub_answer: str,
    hint: Optional[dict[str, object]] = None,
) -> Optional[str]:
    return _fallback_phone_from_sub_answer(user_text, sub_answer, hint)


def build_contact_context(user_text: str, sub_answer: str, mode: str = "fast") -> PostProcessContext:
    hint = detect_dept_hint(user_text)
    label, phone, url = _parse_bullets_and_pick(user_text, sub_answer, hint)
    contact_intent = bool(CONTACT_INTENT_RE.search(user_text or ""))
    if contact_intent and not phone:
        phone = _fallback_phone_from_sub_answer(user_text, sub_answer, hint)

    url = ensure_layout_unknown(url) if url else None
    if contact_intent and hint and label and not _hint_matches_line(label, "", hint):
        label = None

    return PostProcessContext(
        user_text=user_text,
        mode=mode,
        sub_answer=sub_answer,
        label=label,
        phone=phone,
        url=url,
        hint=hint,
        first_line=_first_line_or_default(sub_answer, "요청하신 정보를 찾지 못했습니다."),
        contact_intent=contact_intent,
    )


def render_contact_response(ctx: PostProcessContext) -> tuple[str, Optional[str]]:
    if ctx.label and ctx.phone:
        return f"{ctx.label} 전화번호는 {ctx.phone}입니다.", ctx.url
    if ctx.contact_intent and ctx.phone:
        return f"요청하신 부서 담당자 전화번호는 {ctx.phone}입니다.", ctx.url
    if ctx.contact_intent and not ctx.phone:
        return "담당자 연락처를 바로 찾지 못했습니다. 학과(또는 부서) 풀네임으로 다시 입력해 주세요.", ctx.url
    if ctx.label and ctx.url:
        return f"{ctx.label} 정보는 {ctx.url}에서 확인할 수 있습니다.", ctx.url
    return ctx.first_line, ctx.url


def _build_default_info_text(user_text: str, suffix: str = "") -> str:
    return f"‘{user_text}’ 관련 정보는 {ORG_HOMEPAGE_LABEL}{suffix}에서 확인할 수 있습니다."


def _first_line_or_default(sub_answer: str, default_text: str) -> str:
    if not sub_answer:
        return default_text
    first = sub_answer.strip().splitlines()[0].lstrip("- ").strip()
    if not first:
        return default_text
    return first if first.endswith(("다.", "요.")) else first + "."


def _collect_title_url_candidates(
    user_text: str,
    sub_answer: str,
    scorer,
) -> list[tuple[float, str, str]]:
    candidates: list[tuple[float, str, str]] = []
    for item in TITLE_URL_PAT.finditer(sub_answer or ""):
        title = (item.group("title") or "").strip()
        url = (item.group("url") or "").strip()
        if SAFETY_URL_RE.search(url) or SAFETY_TITLE_RE.search(title):
            continue
        candidates.append((scorer(user_text, title, url), title, url))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _topic_candidate_score(user_text: str, title: str, url: str) -> float:
    hint = detect_dept_hint(user_text)
    compact_query = re.sub(r"\s+", "", user_text)
    score = 0.0
    if GOOD_URL_RE.search(url):
        score += 3.0
    if BAD_URL_RE.search(url):
        score -= 4.0
    if re.sub(r"\s+", "", title) == compact_query:
        score += 6.0
    if INTRO_WORD_RE.search(title):
        score += 3.0
    if CONTACT_WORD_RE.search(title):
        score -= 5.0
    tokens = [token for token in re.findall(HANGUL_TOKEN_PATTERN, user_text)]
    if any(token in title for token in tokens):
        score += 1.0
    if hint:
        if hint["path_base"] and hint["path_base"] in url:
            score += 8.0
        if any(alias in title for alias in hint["aliases"]):
            score += 4.0
    return score


def _policy_candidate_score(user_text: str, title: str, url: str) -> float:
    synonyms = expand_policy_synonyms(user_text)
    tokens = [token for token in re.findall(HANGUL_DOT_TOKEN_PATTERN, user_text)]
    score = 0.0
    if GOOD_URL_RE.search(url):
        score += 3.5
    if BAD_URL_RE.search(url):
        score -= 8.0
    if any(keyword in title for keyword in synonyms):
        score += 7.0
    if any(keyword in title for keyword in ("휴학", "복학", "휴·복학", "휴복학", "학적", "학사안내")):
        score += 3.0
    if any(token in title for token in tokens):
        score += 1.0
    if CONTACT_WORD_RE.search(title):
        score -= 3.0
    return score


def _dorm_candidate_score(user_text: str, title: str, url: str) -> float:
    synonyms = expand_dorm_synonyms(user_text)
    tokens = [token for token in re.findall(HANGUL_DOT_TOKEN_PATTERN, user_text)]
    score = 0.0
    if GOOD_URL_RE.search(url):
        score += 3.5
    if BAD_URL_RE.search(url):
        score -= 8.0
    if any(keyword in title for keyword in synonyms):
        score += 8.0
    if any(keyword in title for keyword in ("학생생활관", "생활관", "기숙사", "입사", "생활관비", "생활관 안내")):
        score += 3.0
    if any(token in title for token in tokens):
        score += 1.0
    if CONTACT_WORD_RE.search(title):
        score -= 3.0
    return score


def _best_title_url_candidate(
    user_text: str,
    sub_answer: str,
    default_text: str,
    scorer,
    prefer_good_url: bool = False,
    require_positive_score: bool = False,
) -> tuple[Optional[str], Optional[str], str]:
    items = list(TITLE_URL_PAT.finditer(sub_answer or ""))
    if not sub_answer:
        return None, None, default_text
    if not items:
        return None, None, _first_line_or_default(sub_answer, default_text)

    candidates = _collect_title_url_candidates(user_text, sub_answer, scorer)
    if not candidates:
        return None, None, default_text

    selected = candidates
    if prefer_good_url:
        selected = [candidate for candidate in candidates if GOOD_URL_RE.search(candidate[2])] or candidates

    best_score, best_title, best_url = selected[0]
    if require_positive_score and best_score <= 0:
        return None, None, default_text
    return best_title, best_url, _first_line_or_default(sub_answer, default_text)


def one_sentence_topic(user_text: str, sub_answer: str) -> tuple[str, Optional[str]]:
    title, url, first_line = extract_topic_candidate(user_text, sub_answer)
    default_text = _build_default_info_text(user_text)
    if not sub_answer:
        return default_text, ORG_HOMEPAGE_URL
    if title and url:
        best_url = ensure_layout_unknown(url)
        return f"‘{user_text}’ 관련 정보는 ‘{title}’ 페이지({best_url})에서 확인할 수 있습니다.", best_url
    return first_line, None


def extract_topic_candidate(user_text: str, sub_answer: str) -> tuple[Optional[str], Optional[str], str]:
    default_text = _build_default_info_text(user_text)
    return _best_title_url_candidate(
        user_text,
        sub_answer,
        default_text,
        scorer=_topic_candidate_score,
        require_positive_score=True,
    )


def one_sentence_policy(user_text: str, sub_answer: str) -> tuple[str, Optional[str]]:
    title, url, first_line = extract_policy_candidate(user_text, sub_answer)
    default_text = _build_default_info_text(user_text, "의 학사안내")
    if not sub_answer:
        return default_text, ORG_HOMEPAGE_URL
    if title and url:
        best_url = ensure_layout_unknown(url)
        return f"‘{user_text}’ 관련 공식 안내는 ‘{title}’ 페이지({best_url})에서 확인할 수 있습니다.", best_url
    return first_line, None


def extract_policy_candidate(user_text: str, sub_answer: str) -> tuple[Optional[str], Optional[str], str]:
    default_text = _build_default_info_text(user_text, "의 학사안내")
    return _best_title_url_candidate(
        user_text,
        sub_answer,
        default_text,
        scorer=_policy_candidate_score,
        prefer_good_url=True,
    )


def one_sentence_dorm(user_text: str, sub_answer: str) -> tuple[str, Optional[str]]:
    title, url, first_line = extract_dorm_candidate(user_text, sub_answer)
    default_text = _build_default_info_text(user_text, "의 생활관 안내")
    if not sub_answer:
        return default_text, ORG_HOMEPAGE_URL
    if title and url:
        best_url = ensure_layout_unknown(url)
        return f"‘{user_text}’ 관련 공식 안내는 ‘{title}’ 페이지({best_url})에서 확인할 수 있습니다.", best_url
    return first_line, None


def extract_dorm_candidate(user_text: str, sub_answer: str) -> tuple[Optional[str], Optional[str], str]:
    default_text = _build_default_info_text(user_text, "의 생활관 안내")
    return _best_title_url_candidate(
        user_text,
        sub_answer,
        default_text,
        scorer=_dorm_candidate_score,
        prefer_good_url=True,
    )


def _extract_url_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    match = SRC_URL_PAT.search(text) or SRC_URL_DOT_PAT.search(text)
    if match:
        return ensure_layout_unknown(_clean_url(match.group("url").strip()))
    urls = URL_PAT.findall(text)
    if urls:
        return ensure_layout_unknown(_clean_url(urls[0]))
    return None


def one_sentence_grad(user_text: str, sub_answer: str) -> tuple[str, Optional[str]]:
    if not sub_answer:
        return f"졸업 관련 정보는 {GRAD_PAGE_URL}에서 확인할 수 있습니다.", GRAD_PAGE_URL
    summary, source_url = extract_grad_summary(user_text, sub_answer)
    if "확인할 수 있습니다." in summary:
        return summary, source_url
    return f"{summary} 자세한 내용은 {source_url}에서 확인할 수 있습니다.", source_url


def extract_grad_summary(user_text: str, sub_answer: str) -> tuple[str, str]:
    if not sub_answer:
        return "졸업 관련 정보를 찾지 못했습니다.", ensure_layout_unknown(GRAD_PAGE_URL)

    lines = [re.sub(r"\*\*", "", line.strip().lstrip("- ").strip()) for line in sub_answer.splitlines() if line.strip()]
    ask_two_year = bool(re.search(r"2\s*년제", user_text))
    ask_three_year = bool(re.search(r"3\s*년제", user_text))

    preferred = []
    for line in lines:
        if ask_two_year and "2년제" in line:
            preferred.append(line)
        elif ask_three_year and "3년제" in line:
            preferred.append(line)
        elif any(keyword in line for keyword in ("졸업학점", "졸업이수 학점", "전공최저", "총 졸업학점", "졸업이수학점")):
            preferred.append(line)

    summary = preferred[0] if preferred else (lines[0] if lines else "졸업 관련 정보를 찾지 못했습니다.")
    source_url = _extract_url_from_text(sub_answer) or ensure_layout_unknown(GRAD_PAGE_URL)
    return summary, source_url
