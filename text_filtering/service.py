import hashlib
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from transformers import ElectraForSequenceClassification, ElectraTokenizer

from core.logging import (
    RuntimeComponent,
    RuntimeOperation,
    RuntimeStatus,
    get_logger,
    runtime_log_message,
)
from core.settings import get_settings
from text_filtering.word_matcher import detect_bad_word_match_dicts


MODEL_PATH = Path("model/my_electra_finetuned")
LOG_PATH = Path("data/bad_text_sample.txt")
ENGLISH_BAD_WORDS_PATH = Path("data/eng_bad_text.txt")
logger = get_logger(__name__)

SENTENCE_ENDINGS = [
    "다",
    "요",
    "죠",
    "네",
    "습니다",
    "습니까",
    "해요",
    "했어요",
    "하였습니다",
    "하네요",
    "해봐요",
]


def load_english_bad_words(file_path: Path) -> set[str]:
    bad_words: set[str] = set()
    try:
        with file_path.open("r", encoding="utf-8") as file:
            for line in file:
                word = line.strip().lower()
                if word:
                    bad_words.add(word)
    except FileNotFoundError:
        logger.warning("영어 비속어 사전 파일을 찾을 수 없습니다: %s", file_path)
    return bad_words


ENGLISH_BAD_WORDS = load_english_bad_words(ENGLISH_BAD_WORDS_PATH)
_text_filter_ready_cache: dict[str, object] | None = None


def get_text_filter_readiness() -> dict[str, object]:
    global _text_filter_ready_cache
    if _text_filter_ready_cache and _text_filter_ready_cache["status"] == "ready":
        return _text_filter_ready_cache

    required_files = [
        MODEL_PATH / "config.json",
        MODEL_PATH / "tokenizer_config.json",
        MODEL_PATH / "vocab.txt",
    ]
    model_weight_candidates = [
        MODEL_PATH / "model.safetensors",
        MODEL_PATH / "pytorch_model.bin",
    ]
    model_weights_found = any(path.exists() for path in model_weight_candidates)
    missing = [str(path) for path in required_files if not path.exists()]

    ready = not missing and model_weights_found and ENGLISH_BAD_WORDS_PATH.exists()
    result = {
        "status": "ready" if ready else "not_ready",
        "required": True,
        "model_path": str(MODEL_PATH),
        "missing_files": missing,
        "model_weights_found": model_weights_found,
        "model_weight_candidates": [str(path) for path in model_weight_candidates],
        "english_dictionary": ENGLISH_BAD_WORDS_PATH.exists(),
    }
    _text_filter_ready_cache = result
    return result


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def get_text_filter_model() -> tuple[ElectraTokenizer, ElectraForSequenceClassification, torch.device]:
    tokenizer = ElectraTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = ElectraForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
    device = get_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"([.!?])\s+", r"\1\n", text)
    for ending in SENTENCE_ENDINGS:
        text = re.sub(rf"({ending})(?=\s)", r"\1\n", text)
    return [sentence.strip() for sentence in text.split("\n") if sentence.strip()]


def contains_english_profanity(text: str) -> bool:
    lower_text = text.lower()
    return any(bad_word in lower_text for bad_word in ENGLISH_BAD_WORDS)


def predict(text: str) -> tuple[int, str]:
    tokenizer, model, device = get_text_filter_model()
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred_label = torch.argmax(logits, dim=-1).item()

    return pred_label, "비속어" if pred_label == 1 else "정상"


def _log_ml_filter_runtime(
    start: float,
    *,
    status: RuntimeStatus,
    sentence_count: int,
    result_count: int,
    model_unavailable: bool,
    fallback: bool,
    fallback_reason: str | None,
    error_code: str | None,
    english_rule_override_count: int = 0,
    field_count: int = 1,
) -> None:
    runtime_status = RuntimeStatus.FALLBACK if fallback and status == RuntimeStatus.SUCCESS else status
    logger.info(
        runtime_log_message(
            "text_filter_ml_runtime",
            component=RuntimeComponent.TEXT_FILTERING,
            operation=RuntimeOperation.ML_FILTER,
            status=runtime_status,
            duration_ms=int((time.monotonic() - start) * 1000),
            result_count=result_count,
            fallback=fallback,
            fallback_reason=fallback_reason,
            error_code=error_code,
            sentence_count=sentence_count,
            field_count=field_count,
            model_unavailable=model_unavailable,
            english_rule_override_count=english_rule_override_count,
        )
    )


def _is_model_unavailable() -> bool:
    try:
        return get_text_filter_readiness().get("status") != "ready"
    except Exception:
        return True


def _sanitized_pending_log_line(sentence: str, label_num: int, sentence_index: int) -> str:
    text_hash = hashlib.sha256(sentence.encode("utf-8")).hexdigest()[:12]
    return f"text_hash={text_hash} text_length={len(sentence)} sentence_index={sentence_index}|{label_num}\n"


def _strong_rule_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [match for match in matches if match.get("strong_rule_candidate") is True]


def analyze_field(field_name: str, text: str, should_log: bool = False) -> dict[str, Any]:
    results: list[dict[str, str]] = []
    matches: list[dict[str, Any]] = []
    has_profanity = False
    log_lines: list[str] = []
    english_rule_override_count = 0
    strong_rule_override_enabled = get_settings().text_filter_strong_rule_override

    for sentence_index, sentence in enumerate(split_sentences(text), start=1):
        label_num, label_text = predict(sentence)
        if label_text == "정상" and contains_english_profanity(sentence):
            label_text = "비속어"
            label_num = 1
            english_rule_override_count += 1

        results.append({"sentence": sentence, "label": label_text})
        for match in detect_bad_word_match_dicts(sentence):
            matches.append({**match, "sentence_index": sentence_index})
        if should_log:
            log_lines.append(_sanitized_pending_log_line(sentence, label_num, sentence_index))
        if label_text == "비속어":
            has_profanity = True

    strong_rule_matches = _strong_rule_matches(matches)
    strong_rule_override_applied = strong_rule_override_enabled and bool(strong_rule_matches)
    if strong_rule_override_applied:
        has_profanity = True

    return {
        "field": field_name,
        "has_profanity": has_profanity,
        "results": results,
        "matches": matches,
        "strong_rule_override": {
            "enabled": strong_rule_override_enabled,
            "applied": strong_rule_override_applied,
            "match_count": len(strong_rule_matches),
        },
        "log_lines": log_lines,
        "english_rule_override_count": english_rule_override_count,
    }


def analyze_fields(specs: list[tuple[str, str, bool]]) -> dict[str, dict[str, Any]]:
    start = time.monotonic()
    analyzed: dict[str, dict[str, Any]] = {}
    pending_logs: list[str] = []
    sentence_count = 0
    english_rule_override_count = 0

    try:
        for field_name, text, should_log in specs:
            result = analyze_field(field_name, text, should_log=should_log)
            sentence_count += len(result.get("results", []))
            english_rule_override_count += int(result.pop("english_rule_override_count", 0))
            pending_logs.extend(result.pop("log_lines"))
            analyzed[field_name] = result

        if pending_logs:
            with LOG_PATH.open("a", encoding="utf-8") as file:
                file.writelines(pending_logs)
    except Exception as exc:
        _log_ml_filter_runtime(
            start,
            status=RuntimeStatus.FAILED,
            sentence_count=sentence_count,
            result_count=sum(len(result.get("results", [])) for result in analyzed.values()),
            model_unavailable=_is_model_unavailable(),
            fallback=True,
            fallback_reason="analyze_fields_failed",
            error_code=type(exc).__name__,
            english_rule_override_count=english_rule_override_count,
            field_count=len(specs),
        )
        raise

    _log_ml_filter_runtime(
        start,
        status=RuntimeStatus.SUCCESS,
        sentence_count=sentence_count,
        result_count=sum(len(result.get("results", [])) for result in analyzed.values()),
        model_unavailable=False,
        fallback=english_rule_override_count > 0,
        fallback_reason="english_rule_override" if english_rule_override_count > 0 else None,
        error_code=None,
        english_rule_override_count=english_rule_override_count,
        field_count=len(specs),
    )

    return analyzed


def analyze_text_labels(text: str) -> list[str]:
    start = time.monotonic()
    labels: list[str] = []
    sentences = split_sentences(text)
    english_rule_override_count = 0
    try:
        for sentence in sentences:
            _, label_text = predict(sentence)
            if label_text == "정상" and contains_english_profanity(sentence):
                label_text = "비속어"
                english_rule_override_count += 1
            labels.append(label_text)
    except Exception as exc:
        _log_ml_filter_runtime(
            start,
            status=RuntimeStatus.FAILED,
            sentence_count=len(sentences),
            result_count=len(labels),
            model_unavailable=_is_model_unavailable(),
            fallback=True,
            fallback_reason="analyze_text_labels_failed",
            error_code=type(exc).__name__,
            english_rule_override_count=english_rule_override_count,
        )
        raise

    _log_ml_filter_runtime(
        start,
        status=RuntimeStatus.SUCCESS,
        sentence_count=len(sentences),
        result_count=len(labels),
        model_unavailable=False,
        fallback=english_rule_override_count > 0,
        fallback_reason="english_rule_override" if english_rule_override_count > 0 else None,
        error_code=None,
        english_rule_override_count=english_rule_override_count,
    )
    return labels
