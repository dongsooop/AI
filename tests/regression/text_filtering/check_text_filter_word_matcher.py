#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from text_filtering.word_matcher import detect_bad_word_matches


CASES = [
    {
        "id": "spacing_variant",
        "text": "시 발 이라고 띄어 쓰는 표현",
        "expected_pattern": "korean_sibal",
    },
    {
        "id": "special_character_variant",
        "text": "씨@발처럼 특수문자를 섞은 표현",
        "expected_pattern": "korean_ssibal",
    },
    {
        "id": "repeated_character_variant",
        "text": "씨이발처럼 글자를 늘린 표현",
        "expected_pattern": "korean_ssibal",
    },
    {
        "id": "initial_consonant_variant",
        "text": "댓글에 ㅅㅂ 같은 초성 욕설",
        "expected_pattern": "korean_initial_s_b",
    },
    {
        "id": "english_number_variant",
        "text": "That was a sh1t response.",
        "expected_pattern": "english_shit",
    },
    {
        "id": "romanized_korean_variant",
        "text": "댓글에 sibal 같은 로마자 욕설",
        "expected_pattern": "romanized_sibal",
    },
    {
        "id": "korean_number_variant",
        "text": "ㅅ1발처럼 숫자를 섞은 표현",
        "expected_pattern": "korean_number_sibal",
    },
    {
        "id": "benign_substring_and_spacing_variant",
        "text": "시발점은 논의고 시 발은 욕설입니다.",
        "expected_pattern": "korean_sibal",
    },
]

NEGATIVE_CASES = [
    {
        "id": "false_positive_guard",
        "text": "이 논의의 시발점은 수업 공지 내용이었습니다.",
    },
    {
        "id": "normal_english_sentence",
        "text": "Please check the class notice before tomorrow.",
    },
]


def main() -> int:
    failures: list[str] = []

    for case in CASES:
        matches = detect_bad_word_matches(case["text"])
        pattern_ids = {match.pattern_id for match in matches}
        if case["expected_pattern"] not in pattern_ids:
            failures.append(f"{case['id']}:missing_pattern:{case['expected_pattern']}:actual={sorted(pattern_ids)}")
            continue
        expected_matches = [match for match in matches if match.pattern_id == case["expected_pattern"]]
        if not any(match.strong_rule_candidate and match.rule_stage == "strong_candidate" for match in expected_matches):
            failures.append(f"{case['id']}:expected_strong_rule_candidate:{case['expected_pattern']}")

    for case in NEGATIVE_CASES:
        matches = detect_bad_word_matches(case["text"])
        if matches:
            pattern_ids = sorted(match.pattern_id for match in matches)
            failures.append(f"{case['id']}:unexpected_match:{pattern_ids}")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1

    print(f"[OK] text filter word matcher cases passed: {len(CASES) + len(NEGATIVE_CASES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
