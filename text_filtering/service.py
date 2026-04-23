import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from transformers import ElectraForSequenceClassification, ElectraTokenizer


MODEL_PATH = Path("model/my_electra_finetuned")
LOG_PATH = Path("data/bad_text_sample.txt")
ENGLISH_BAD_WORDS_PATH = Path("data/eng_bad_text.txt")

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
        print(f"[ERROR] 영어 비속어 사전 파일을 찾을 수 없습니다: {file_path}")
    return bad_words


ENGLISH_BAD_WORDS = load_english_bad_words(ENGLISH_BAD_WORDS_PATH)


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


def analyze_field(field_name: str, text: str, should_log: bool = False) -> dict[str, Any]:
    results: list[dict[str, str]] = []
    has_profanity = False
    log_lines: list[str] = []

    for sentence in split_sentences(text):
        label_num, label_text = predict(sentence)
        if label_text == "정상" and contains_english_profanity(sentence):
            label_text = "비속어"
            label_num = 1

        results.append({"sentence": sentence, "label": label_text})
        if should_log:
            log_lines.append(f"{sentence}|{label_num}\n")
        if label_text == "비속어":
            has_profanity = True

    return {
        "field": field_name,
        "has_profanity": has_profanity,
        "results": results,
        "log_lines": log_lines,
    }


def analyze_fields(specs: list[tuple[str, str, bool]]) -> dict[str, dict[str, Any]]:
    analyzed: dict[str, dict[str, Any]] = {}
    pending_logs: list[str] = []

    for field_name, text, should_log in specs:
        result = analyze_field(field_name, text, should_log=should_log)
        pending_logs.extend(result.pop("log_lines"))
        analyzed[field_name] = result

    if pending_logs:
        with LOG_PATH.open("a", encoding="utf-8") as file:
            file.writelines(pending_logs)

    return analyzed


def analyze_text_labels(text: str) -> list[str]:
    labels: list[str] = []
    for sentence in split_sentences(text):
        _, label_text = predict(sentence)
        if label_text == "정상" and contains_english_profanity(sentence):
            label_text = "비속어"
        labels.append(label_text)
    return labels
