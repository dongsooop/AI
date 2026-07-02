#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from text_filtering.normalization import normalize_text_variants


CASES = [
    {
        "id": "spacing_variant",
        "text": "시 발",
        "expected": ("compact", "시발"),
    },
    {
        "id": "special_character_variant",
        "text": "씨@발",
        "expected": ("symbol_removed_compact", "씨발"),
    },
    {
        "id": "repeated_character_variant",
        "text": "씨이발",
        "expected": ("repeated_collapsed_compact", "씨발"),
    },
    {
        "id": "english_number_variant",
        "text": "sh1t",
        "expected": ("latin_number_mapped_compact", "shit"),
    },
    {
        "id": "romanized_korean_variant",
        "text": "SI BAL",
        "expected": ("compact", "sibal"),
    },
    {
        "id": "korean_number_variant",
        "text": "ㅅ1발",
        "expected": ("korean_number_mapped_compact", "ㅅㅣ발"),
    },
    {
        "id": "false_positive_guard",
        "text": "시발점",
        "expected": ("original", "시발점"),
    },
]


def main() -> int:
    failures: list[str] = []

    for case in CASES:
        variants = normalize_text_variants(case["text"])
        key, expected_value = case["expected"]
        actual_value = variants.get(key)
        if actual_value != expected_value:
            failures.append(f"{case['id']}:{key}:expected={expected_value}:actual={actual_value}")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1

    print(f"[OK] text filter normalization cases passed: {len(CASES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
