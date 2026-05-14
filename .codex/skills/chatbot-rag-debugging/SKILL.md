---
name: chatbot-rag-debugging
description: Debug Dongsooop OSS chatbot RAG behavior. Use when investigating chatbot answer quality, retrieval misses, source grounding, hallucination, contact lookup, academic schedule answers, artifact/index issues, Ollama/OpenAI-compatible client behavior, or regressions in LLM/OSS/Open_AI_OSS.py and LLM/sub_model/*.
---

# 챗봇 RAG 디버깅

## 개요

Dongsooop 챗봇 서비스를 디버깅할 때 이 스킬을 사용합니다. 독립된 텍스트 필터링/시간표 서비스는 건드리지 않고, 검색 품질, 출처 기반 응답, 한국어 질의 처리, 생성된 인덱스 아티팩트, Ollama 런타임 설정을 중심으로 확인합니다.

## 필수 확인 문맥

코드를 수정하기 전에 다음 파일을 먼저 읽습니다.

1. `docs/AGENTS.md`: 저장소 가드레일
2. `docs/PLANS.md`: 서비스 경계와 챗봇 아키텍처
3. 관련 챗봇 파일:
   - `app_oss_main.py`
   - `LLM/OSS/Open_AI_OSS.py`
   - `LLM/sub_model/query_index.py`
   - `LLM/sub_model/schedule_index.py`
   - `LLM/sub_model/index_utils.py`
   - 아티팩트 생성이 관련될 때 `LLM/sub_model/build_index.py`

설정, 배포, 런타임 시작 문제가 관련될 때만 `.env.example`, `Dockerfiles_oss`, `docker-compose-oss.yml`을 확인합니다.

## 가드레일

- 사용자가 명시적으로 공통 서비스 작업을 요청하지 않는 한, 챗봇 변경은 OSS 챗봇 서비스 안에 머물게 합니다.
- `LLM/OSS/Open_AI_OSS.py`는 고위험 파일로 취급합니다. 주변 흐름을 먼저 이해하고, 수정은 좁게 합니다.
- `model/artifacts/` 아래 생성 산출물은 직접 수정하지 않습니다. 생성 로직을 바꾸거나 사용자에게 아티팩트 재생성을 안내합니다.
- 한국어 NLP 동작을 유지합니다. Okt 토큰화, 한국어 정규화, stopword, 연락처 추출, 학사일정/이벤트 태깅을 바꿀 때 특히 조심합니다.
- 비밀값, `.env` 값, 서버 URL, 토큰, 비공개 배포 정보를 커밋하지 않습니다.
- 사용자가 명시적으로 요청하지 않는 한, 챗봇 Docker 이미지가 `.env`를 복사할 수 있다는 OCI 배포 전제를 유지합니다.

## 디버깅 흐름

1. 사용자 프롬프트, 로그, 또는 구체적인 실패 질의로 문제를 재현합니다.
2. 실패 유형을 분류합니다.
   - 검색 누락
   - 잘못된 랭킹
   - 검색 문맥에 근거하지 않은 답변
   - 인사/의도 라우팅 오류
   - 연락처 조회 오류
   - 학사일정/날짜 파싱 오류
   - Ollama/client/runtime 오류
   - 오래되었거나 누락된 아티팩트
3. 요청 흐름을 `app_oss_main.py`에서 `LLM/OSS/Open_AI_OSS.py`로, 그다음 관련 `LLM/sub_model/*` 모듈로 따라갑니다.
4. 코드를 바꾸기 전에 환경변수 기반 경로와 플래그를 확인합니다.
   - `OSS_BASE_URL`
   - `OSS_MODEL`
   - `DATA_JSON`
   - `SEARCH_DF_PATH`
   - `EMB_PATH`
   - `BM25_PATH`
   - `SCHEDULE_CSV_PATH`
   - `STRICT_SOURCE_DATA_ONLY`
5. 아티팩트가 존재하는지, 기대 스키마와 맞는지 확인합니다. 데이터가 바뀐 경우 아티팩트를 직접 고치지 말고 `LLM/sub_model/build_index.py`로 재생성하는 쪽을 우선합니다.
6. 분류한 실패를 고치는 가장 작은 코드 변경을 적용합니다.
7. 집중 질의와 가능한 경우 챗봇 회귀 스크립트로 검증합니다.

```bash
python debug/regression/run_chatbot_regression.py
```

## 검색 점검

검색 누락이나 오답이 있을 때 확인합니다.

- 랭킹 가중치를 바꾸기 전에 질의 정규화와 토큰화를 확인합니다.
- 원본 코퍼스와 생성된 `search_df`에 근거 문서가 있는지 확인합니다.
- BM25와 embedding 결과가 모두 가능하면 둘을 비교합니다.
- 한국어 띄어쓰기, 약어, 동의어, 학과명, 캠퍼스 특화 표현 불일치를 찾습니다.
- 연락처 질의는 프롬프트 문구만 보지 말고 `contact_docs.csv` 생성과 연락처 추출 로직을 확인합니다.
- 학사일정 질의는 최종 답변 프롬프트를 조정하기 전에 일정 파싱과 이벤트 태깅을 먼저 봅니다.

## 근거성 점검

환각이나 근거 없는 답변이 있을 때 확인합니다.

- `STRICT_SOURCE_DATA_ONLY`가 켜져 있는지 확인합니다.
- 모델에 전달되는 검색 문맥을 확인합니다.
- 넓은 프롬프트 문구를 추가하기 전에 출처 필터링이나 답변 조립 로직을 강화합니다.
- 신뢰할 수 있는 출처가 검색되지 않으면 명시적인 fallback 동작을 우선합니다.
- 최종 답변 포맷을 바꾸더라도 한국어 사용자 응답이 자연스럽게 유지되는지 확인합니다.

## 런타임 점검

시작 실패나 모델 호출 실패가 있을 때 확인합니다.

- 챗봇 서비스가 포트 `8010`의 OSS 서비스인지 확인합니다. 포트 `8888`의 텍스트 필터링/시간표 서비스와 혼동하지 않습니다.
- Ollama 또는 설정된 OpenAI 호환 엔드포인트가 런타임 환경에서 접근 가능한지 확인합니다.
- OpenAI SDK 호환 client가 기대하는 경우 `OSS_BASE_URL`에 `/v1` 경로가 맞게 들어가는지 확인합니다.
- 배포된 Ollama 모델과 모델 이름이 호환되는지 확인합니다. 환경변수 override가 없으면 보통 `gpt-oss:20b`입니다.
- KoNLPy, JPype1, pandas, numpy, sklearn, rank-bm25, OpenAI SDK 관련 의존성 변경은 회귀 리스크가 큰 변경으로 취급합니다.

## 출력 형식

결과를 보고할 때 다음 내용을 포함합니다.

1. 실패 유형과 원인
2. 변경하거나 확인한 파일
3. 실행한 검증과 정확한 명령
4. 남은 리스크, 특히 아티팩트/데이터/Ollama 의존 동작 중 검증하지 못한 부분

코드 변경이 필요 없다면 운영상 조치나 데이터 재생성 단계를 명확히 설명합니다.
