import re
from dataclasses import asdict, dataclass
from typing import Literal

from text_filtering.normalization import normalize_text_variants


MatchSeverity = Literal["high", "medium"]
RuleStage = Literal["strong_candidate", "observe_only"]

BENIGN_KOREAN_SUBSTRINGS = {
    "시발점",
}

KOREAN_COMPACT_PATTERNS = {
    "korean_sibal": ("시발", "high"),
    "korean_ssibal": ("씨발", "high"),
    "korean_initial_s_b": ("ㅅㅂ", "high"),
    "korean_number_sibal": ("ㅅㅣ발", "high"),
}

LATIN_WORD_PATTERNS = {
    "romanized_sibal": ("sibal", "high"),
    "english_shit": ("shit", "high"),
}

STRONG_RULE_CANDIDATE_PATTERN_IDS = {
    "korean_sibal",
    "korean_ssibal",
    "korean_initial_s_b",
    "korean_number_sibal",
    "romanized_sibal",
    "english_shit",
}


@dataclass(frozen=True)
class WordMatch:
    pattern_id: str
    matched_text: str
    normalized: str
    candidate_kind: str
    severity: MatchSeverity
    rule_stage: RuleStage
    strong_rule_candidate: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _has_benign_korean_context(text: str, term: str) -> bool:
    return any(benign in text for benign in BENIGN_KOREAN_SUBSTRINGS if term in benign)


def _append_match_once(matches: list[WordMatch], seen: set[str], match: WordMatch) -> None:
    if match.pattern_id in seen:
        return
    seen.add(match.pattern_id)
    matches.append(match)


def _make_word_match(
    *,
    pattern_id: str,
    matched_text: str,
    normalized: str,
    candidate_kind: str,
    severity: MatchSeverity,
) -> WordMatch:
    is_strong_candidate = pattern_id in STRONG_RULE_CANDIDATE_PATTERN_IDS
    return WordMatch(
        pattern_id=pattern_id,
        matched_text=matched_text,
        normalized=normalized,
        candidate_kind=candidate_kind,
        severity=severity,
        rule_stage="strong_candidate" if is_strong_candidate else "observe_only",
        strong_rule_candidate=is_strong_candidate,
    )


def _find_korean_compact_matches(variants: dict[str, str]) -> list[WordMatch]:
    matches: list[WordMatch] = []
    seen: set[str] = set()
    candidate_kinds = [
        "compact",
        "symbol_removed_compact",
        "korean_number_mapped_compact",
        "repeated_collapsed_compact",
    ]

    for candidate_kind in candidate_kinds:
        candidate = variants.get(candidate_kind, "")
        if not candidate:
            continue
        for pattern_id, (term, severity) in KOREAN_COMPACT_PATTERNS.items():
            if term not in candidate or _has_benign_korean_context(candidate, term):
                continue
            _append_match_once(
                matches,
                seen,
                _make_word_match(
                    pattern_id=pattern_id,
                    matched_text=term,
                    normalized=candidate,
                    candidate_kind=candidate_kind,
                    severity=severity,
                ),
            )

    return matches


def _find_latin_word_matches(variants: dict[str, str]) -> list[WordMatch]:
    matches: list[WordMatch] = []
    seen: set[str] = set()
    candidate_kinds = [
        "lower",
        "symbol_removed",
        "latin_number_mapped",
    ]

    for candidate_kind in candidate_kinds:
        candidate = variants.get(candidate_kind, "")
        if not candidate:
            continue
        for pattern_id, (term, severity) in LATIN_WORD_PATTERNS.items():
            if not re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", candidate):
                continue
            _append_match_once(
                matches,
                seen,
                _make_word_match(
                    pattern_id=pattern_id,
                    matched_text=term,
                    normalized=candidate,
                    candidate_kind=candidate_kind,
                    severity=severity,
                ),
            )

    return matches


def detect_bad_word_matches(text: str) -> list[WordMatch]:
    variants = normalize_text_variants(text)
    return [
        *_find_korean_compact_matches(variants),
        *_find_latin_word_matches(variants),
    ]


def detect_bad_word_match_dicts(text: str) -> list[dict[str, str]]:
    return [match.to_dict() for match in detect_bad_word_matches(text)]
