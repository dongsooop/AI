#!/usr/bin/env python3
import asyncio
import importlib
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> int:
    try:
        from text_filtering import service as text_filter_service
    except ModuleNotFoundError as exc:
        print(f"[SKIP] text filtering dependency is missing: {exc.name}")
        return 0

    original_log_path = text_filter_service.LOG_PATH
    original_predict = text_filter_service.predict
    original_contains_english_profanity = text_filter_service.contains_english_profanity

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "bad_text_sample.txt"
            text_filter_service.LOG_PATH = log_path
            text_filter_service.predict = lambda text: (1, "비속어") if "욕설" in text else (0, "정상")
            text_filter_service.contains_english_profanity = lambda text: False

            labels = text_filter_service.analyze_text_labels(
                "정상 문장입니다. 욕설 문장입니다.",
                store_raw_text=True,
            )
            stored_lines = log_path.read_text(encoding="utf-8").splitlines()
    finally:
        text_filter_service.LOG_PATH = original_log_path
        text_filter_service.predict = original_predict
        text_filter_service.contains_english_profanity = original_contains_english_profanity

    failures: list[str] = []
    if labels != ["정상", "비속어"]:
        failures.append(f"unexpected_labels:{labels}")
    if stored_lines != ["정상 문장입니다.|0", "욕설 문장입니다.|1"]:
        failures.append(f"unexpected_stored_lines:{stored_lines}")

    endpoint_module = importlib.import_module("text_filtering.text_filtering")
    original_analyze_text_labels = endpoint_module.analyze_text_labels
    endpoint_calls: list[bool] = []

    def fake_analyze_text_labels(text: str, *, store_raw_text: bool = False) -> list[str]:
        endpoint_calls.append(store_raw_text)
        return ["정상"]

    try:
        endpoint_module.analyze_text_labels = fake_analyze_text_labels
        payload = endpoint_module.TextRequest(text="테스트")
        asyncio.run(endpoint_module.text_filter_single_api(payload))
        asyncio.run(endpoint_module.text_filter_nickname_api(payload))
    finally:
        endpoint_module.analyze_text_labels = original_analyze_text_labels

    if endpoint_calls != [True, True]:
        failures.append(f"raw_logging_not_enabled_for_both_endpoints:{endpoint_calls}")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1

    print("[OK] raw text and labels are stored for later review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
