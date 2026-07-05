#!/usr/bin/env python3
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> int:
    try:
        from core.settings import get_settings
        from text_filtering import service as text_filter_service
    except ModuleNotFoundError as exc:
        print(f"[SKIP] text filtering dependency is missing: {exc.name}")
        return 0

    original_predict = text_filter_service.predict
    original_contains_english_profanity = text_filter_service.contains_english_profanity

    try:
        text_filter_service.predict = lambda text: (0, "정상")
        text_filter_service.contains_english_profanity = lambda text: False
        os.environ.pop("TEXT_FILTER_STRONG_RULE_OVERRIDE", None)
        get_settings.cache_clear()
        result = text_filter_service.analyze_field("content", "ㅅ1발처럼 숫자를 섞은 표현", should_log=False)

        os.environ["TEXT_FILTER_STRONG_RULE_OVERRIDE"] = "1"
        get_settings.cache_clear()
        override_result = text_filter_service.analyze_field("content", "ㅅ1발처럼 숫자를 섞은 표현", should_log=False)
    finally:
        text_filter_service.predict = original_predict
        text_filter_service.contains_english_profanity = original_contains_english_profanity
        os.environ.pop("TEXT_FILTER_STRONG_RULE_OVERRIDE", None)
        get_settings.cache_clear()

    failures: list[str] = []
    if result.get("has_profanity") is not False:
        failures.append("has_profanity_changed_by_match_details")
    if result.get("results") != [{"sentence": "ㅅ1발처럼 숫자를 섞은 표현", "label": "정상"}]:
        failures.append(f"results_contract_changed:{result.get('results')}")

    matches = result.get("matches")
    if not isinstance(matches, list) or not matches:
        failures.append("missing_matches")
    else:
        first_match = matches[0]
        if first_match.get("pattern_id") != "korean_number_sibal":
            failures.append(f"unexpected_pattern:{first_match.get('pattern_id')}")
        if first_match.get("sentence_index") != 1:
            failures.append(f"unexpected_sentence_index:{first_match.get('sentence_index')}")
        if first_match.get("strong_rule_candidate") is not True:
            failures.append("expected_strong_rule_candidate")

    override = result.get("strong_rule_override")
    if override != {"enabled": False, "applied": False, "match_count": 1}:
        failures.append(f"unexpected_default_override_state:{override}")

    override_state = override_result.get("strong_rule_override")
    if override_result.get("has_profanity") is not True:
        failures.append("strong_rule_override_did_not_raise_has_profanity")
    if override_result.get("results") != [{"sentence": "ㅅ1발처럼 숫자를 섞은 표현", "label": "정상"}]:
        failures.append(f"override_changed_model_results:{override_result.get('results')}")
    if override_state != {"enabled": True, "applied": True, "match_count": 1}:
        failures.append(f"unexpected_enabled_override_state:{override_state}")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1

    print("[OK] text filter match details response is backward compatible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
