import re
from dataclasses import asdict, dataclass


WHITESPACE_RE = re.compile(r"\s+")
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")
KOREAN_VOWEL_EXTENSION_RE = re.compile(r"([가-힣ㄱ-ㅎ])([ㅏ-ㅣ이]){2,}(?=[가-힣ㄱ-ㅎ])")
KOREAN_SIBAL_VOWEL_EXTENSION_RE = re.compile(r"([시씨])이+(?=발)")

LATIN_NUMBER_MAP = str.maketrans({
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
})
KOREAN_NUMBER_MAP = str.maketrans({
    "1": "ㅣ",
})


@dataclass(frozen=True)
class NormalizationCandidate:
    kind: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def remove_symbols(text: str) -> str:
    return "".join(char for char in text if char.isalnum() or char.isspace())


def collapse_repeated_characters(text: str) -> str:
    return REPEATED_CHAR_RE.sub(r"\1", text)


def collapse_korean_vowel_extensions(text: str) -> str:
    collapsed = KOREAN_SIBAL_VOWEL_EXTENSION_RE.sub(r"\1", text)
    return KOREAN_VOWEL_EXTENSION_RE.sub(r"\1", collapsed)


def compact_text(text: str) -> str:
    return WHITESPACE_RE.sub("", text)


def normalize_text_variants(text: str) -> dict[str, str]:
    lower = text.lower()
    symbol_removed = remove_symbols(lower)
    compact = compact_text(lower)
    symbol_removed_compact = compact_text(symbol_removed)
    latin_number_mapped = lower.translate(LATIN_NUMBER_MAP)
    korean_number_mapped = lower.translate(KOREAN_NUMBER_MAP)
    repeated_collapsed = collapse_korean_vowel_extensions(collapse_repeated_characters(lower))

    return {
        "original": text,
        "lower": lower,
        "compact": compact,
        "symbol_removed": symbol_removed,
        "symbol_removed_compact": symbol_removed_compact,
        "latin_number_mapped": latin_number_mapped,
        "latin_number_mapped_compact": compact_text(remove_symbols(latin_number_mapped)),
        "korean_number_mapped": korean_number_mapped,
        "korean_number_mapped_compact": compact_text(remove_symbols(korean_number_mapped)),
        "repeated_collapsed": repeated_collapsed,
        "repeated_collapsed_compact": compact_text(remove_symbols(repeated_collapsed)),
    }


def build_text_filter_normalization_candidates(text: str) -> list[NormalizationCandidate]:
    candidates: list[NormalizationCandidate] = []
    seen: set[tuple[str, str]] = set()

    for kind, candidate_text in normalize_text_variants(text).items():
        key = (kind, candidate_text)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(NormalizationCandidate(kind=kind, text=candidate_text))

    return candidates
