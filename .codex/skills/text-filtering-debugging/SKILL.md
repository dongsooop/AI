---
name: text-filtering-debugging
description: Debug Dongsooop text filtering behavior. Use when investigating Korean or English profanity detection, ELECTRA model inference, sentence splitting, rule-based text filter endpoints, false positives, false negatives, training sample logging, model artifact loading, or regressions in text_filtering/* and finetuning/text_filter/*.
---

# 텍스트 필터링 디버깅

## 개요

Dongsooop 텍스트 필터링 문제를 디버깅할 때 이 스킬을 사용합니다. 이 기능은 `main.py` 서비스의 일부이며, 시간표 OCR과 같은 서비스에 있지만 챗봇 OSS 서비스와는 분리해서 봅니다. ELECTRA 기반 한국어 비속어 판별, 영어 비속어 사전 보정, 문장 분리, 필드별 응답 형식, 학습 데이터 로깅, 모델 산출물 로딩을 중심으로 확인합니다.

## 필수 확인 문맥

코드를 수정하기 전에 다음 파일을 먼저 읽습니다.

1. `docs/AGENTS.md`: 저장소 가드레일
2. `docs/PLANS.md`: 두 FastAPI 서비스 구조와 텍스트 필터링 영역 설명
3. 텍스트 필터링 관련 파일:
   - `main.py`에서 라우터 연결이 관련될 때
   - `text_filtering/service.py`
   - `text_filtering/text_filtering.py`
   - `text_filtering/text_filtering_rule.py`
   - 학습 또는 모델 재생성이 관련될 때 `text_filtering/bad_text_learning_model.py`
   - 추가 파인튜닝이 관련될 때 `finetuning/text_filter/bad_text_learning_finetuning.py`

환경, 모델 로딩, 배포 문제가 관련될 때만 `Dockerfile`, `requirements.txt`, `.env.example`을 확인합니다.

## 가드레일

- 챗봇 OSS 서비스(`app_oss_main.py`, `LLM/`)와 섞지 않습니다. 챗봇이 텍스트 필터 API를 호출하는 문제라면 호출부와 필터 서비스 경계를 분리해서 봅니다.
- 한국어 처리 특성을 유지합니다. 문장 분리, KoELECTRA tokenizer, Okt 기반 학습 전처리, 영어 비속어 보정 로직은 오탐/미탐에 직접 영향을 줍니다.
- `model/my_electra_finetuned`는 모델 산출물입니다. 모델 파일을 직접 고치는 대신 학습/재학습 로직과 데이터 흐름을 확인합니다.
- `data/bad_text_sample.txt`, `data/new_bad_text_sample.txt`, `data/eng_bad_text.txt`는 비속어 데이터와 사용자 입력 로그가 섞일 수 있으므로 공개 가능 여부를 반드시 확인합니다.
- 비밀값, `.env`, JWT, 서버 URL, 운영 데이터는 커밋하지 않습니다.
- 사용자가 명시적으로 요청하지 않는 한 `Dockerfile`의 모델/데이터 복사 전제를 제거하지 않습니다.

## 디버깅 흐름

1. 실패를 재현합니다. 가능한 경우 입력 문장, 호출 엔드포인트, 기대 라벨, 실제 라벨, 응답 상태코드를 함께 확인합니다.
2. 실패 유형을 분류합니다.
   - 한국어 비속어 미탐
   - 정상 문장 오탐
   - 영어 비속어 미탐
   - 문장 분리 오류
   - 필드 구분자 `|` 파싱 오류
   - 인증이 필요한 엔드포인트의 JWT 오류
   - 모델 파일 로딩 실패
   - 학습 데이터 로깅 또는 재학습 문제
   - Docker/배포 환경에서 모델 또는 사전 파일 누락
3. 요청 흐름을 따라갑니다.
   - 라우터 등록: `main.py`
   - ML 기반 엔드포인트: `text_filtering/text_filtering.py`
   - 규칙/콘텐츠 엔드포인트: `text_filtering/text_filtering_rule.py`
   - 공통 분석 로직: `text_filtering/service.py`
4. 모델과 데이터 의존성을 먼저 확인합니다.
   - `model/my_electra_finetuned`
   - `data/bad_text_sample.txt`
   - `data/eng_bad_text.txt`
   - `torch`
   - `transformers`
   - 인증이 필요한 엔드포인트의 `python-jose[cryptography]`
5. 코드 변경은 실패 유형에 맞춰 가장 좁게 적용합니다.
6. 가능한 경우 직접 함수 호출, `py_compile`, 또는 API 요청으로 검증합니다.

```bash
python -m py_compile text_filtering/service.py text_filtering/text_filtering.py text_filtering/text_filtering_rule.py
```

## 엔드포인트 점검

입력 형식이나 응답 상태가 이상할 때 확인합니다.

- `/text_filter_board`는 `자기소개 | 지원동기` 형식을 기대하고 JWT 인증이 필요합니다.
- `/text_filter_market`은 `제목 | 내용` 형식을 기대하고 JWT 인증이 필요합니다.
- `/text_filter_rule`은 `제목 | 태그 | 본문` 형식을 기대하고 JWT 인증이 필요합니다.
- `/text_filter_single`은 단일 텍스트를 받아 문장별 라벨 리스트를 반환합니다.
- `/text_filter_content`는 단일 content 필드를 분석하고 비속어가 있으면 400을 반환합니다.
- `|` 구분자를 바꾸면 클라이언트 계약과 기존 오류 코드가 함께 영향을 받습니다.

## 모델 추론 점검

오탐/미탐이나 모델 로딩 문제가 있을 때 확인합니다.

- `get_text_filter_model()`은 `local_files_only=True`로 `model/my_electra_finetuned`에서 tokenizer와 model을 로드합니다.
- 추론 `max_length=64`, `padding="max_length"`, `truncation=True` 변경은 긴 문장 판별에 영향을 줍니다.
- `get_device()`는 MPS가 가능하면 `mps`, 아니면 `cpu`를 사용합니다. CUDA 사용을 추가할 때는 Apple Silicon 환경과 Docker 환경을 함께 고려합니다.
- `predict()`는 label `1`을 `비속어`, label `0`을 `정상`으로 해석합니다. 학습 데이터 라벨 정의와 반드시 맞아야 합니다.
- 모델 캐시는 `lru_cache(maxsize=1)`로 유지되므로 런타임 중 모델 파일 교체는 바로 반영되지 않을 수 있습니다.

## 문장 분리와 영어 비속어 점검

문장 단위 결과가 이상할 때 확인합니다.

- `split_sentences()`는 `.`, `!`, `?`와 일부 한국어 종결 표현을 기준으로 문장을 나눕니다.
- `SENTENCE_ENDINGS` 변경은 정상 문장의 과분리 또는 긴 문장 미분리에 영향을 줍니다.
- 영어 비속어는 `data/eng_bad_text.txt`를 소문자로 읽고 `contains_english_profanity()`에서 단순 substring으로 검사합니다.
- 영어 사전 보정은 모델이 `정상`이라고 판단한 문장도 영어 비속어가 있으면 `비속어`로 바꿉니다.
- substring 방식은 단어 경계를 보지 않으므로 짧은 영어 단어를 추가할 때 오탐 가능성을 반드시 확인합니다.

## 학습과 데이터 점검

학습 데이터나 모델 재생성이 관련될 때 확인합니다.

- `analyze_fields()`는 `should_log=True`인 필드의 문장과 라벨을 `data/bad_text_sample.txt`에 append합니다.
- 태그 필드는 현재 `text_filter_rule.py`에서 `should_log=False`입니다.
- `text_filtering/bad_text_learning_model.py`는 `data/bad_text_sample.txt`를 읽어 KoELECTRA 모델을 학습하고 `model/my_electra_finetuned`에 저장합니다.
- `finetuning/text_filter/bad_text_learning_finetuning.py`는 `data/new_bad_text_sample.txt`를 사용해 기존 모델을 추가 학습합니다.
- 학습 스크립트는 KoNLPy/Okt와 Java 환경 영향을 받을 수 있습니다.
- 데이터 파일에 사용자 문장이나 민감 표현이 포함될 수 있으므로 공개/커밋 대상인지 반드시 확인합니다.

## Docker와 배포 점검

배포 환경에서만 실패할 때 확인합니다.

- `Dockerfile`은 `text_filtering/`, `model/my_electra_finetuned`, `data/bad_text_sample.txt`, `data/eng_bad_text.txt`를 복사합니다.
- `requirements.txt`에는 `torch`, `transformers`, `pytesseract`, `opencv-python-headless`, `Pillow`, `numpy`, `python-multipart`, `python-jose[cryptography]`, `python-dotenv`가 포함됩니다.
- 모델 또는 영어 사전 파일이 Docker 이미지에 없으면 로컬과 배포 결과가 달라질 수 있습니다.
- `requirements.txt`가 현재 `.gitignore`에서 무시되는 전제를 바꾸려면 의존성 공개 전략도 함께 확인합니다.

## 출력 형식

결과를 보고할 때 다음 내용을 포함합니다.

1. 실패 유형과 원인
2. 확인하거나 변경한 파일
3. 실행한 검증과 정확한 명령
4. 남은 리스크, 특히 모델 파일, 학습 데이터, 영어 사전, Docker 이미지 안 파일 존재 여부 중 검증하지 못한 부분

코드 변경이 필요 없다면 필요한 데이터 수정, 모델 재학습, Docker 이미지 재빌드, 또는 클라이언트 입력 형식 수정 단계를 명확히 설명합니다.
