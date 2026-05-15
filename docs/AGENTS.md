# AGENTS.md

이 문서는 이 저장소에서 작업하는 GPT 기반 코딩 에이전트를 위한 공통 작업 지침입니다.

## 프로젝트 개요

**동숲 (Dongsooop)** 은 동양미래대학교 학생을 위한 AI 백엔드 서비스입니다. 주요 기능은 다음 3가지입니다.

1. **텍스트 필터링** — 파인튜닝된 ELECTRA 기반 한글/영문 비속어 판별
2. **시간표 분석** — OCR 기반 업로드 이미지 시간표 추출
3. **챗봇** — 오픈소스 LLM(Ollama) 기반 학교 정보 RAG 질의응답

## 작업 원칙

- 두 개의 서비스 구조를 유지합니다. 명시적인 요구가 없으면 챗봇 관련 내용을 텍스트 필터링/시간표 서비스에 섞지 않습니다.
- 넓은 범위의 리팩토링보다 필요한 부분만 국소적으로 수정하는 것을 우선합니다. 특히 `LLM/OSS/Open_AI_OSS.py`는 크고 핵심적인 파일이므로 더 신중하게 다룹니다.
- 한국어 처리 특성을 유지해야 합니다. 토크나이징, OCR 전처리, 비속어 규칙, RAG 검색 로직을 바꿀 때 특히 주의합니다.
- 비밀값, 토큰, `.env` 값, SSH 키, 서버 접속 정보는 소스 파일에 커밋하지 않습니다.
- 이 프로젝트의 OCI 배포는 Docker 이미지 안의 `.env` 파일을 사용하는 전제를 포함합니다. 명시적인 요구가 없으면 `Dockerfile` 또는 `Dockerfiles_oss`에서 `COPY .env ./.env`를 제거하지 않습니다.
- 루트의 `AGENTS.md`는 로컬 에이전트 진입점입니다. 추가 로컬 보조 문서는 `.local-agent-docs/`에 두며, 공개용 문서는 `docs/` 아래 파일을 사용합니다.
- Codex에서 반복적으로 수행하는 브랜치 리뷰 작업은 `.codex/skills/reviewing-dongsooop-branch/SKILL.md`의 프로젝트 skill을 사용합니다.
- `model/artifacts/` 아래 파일은 생성 산출물로 간주합니다. 직접 수정하지 말고 생성 로직을 수정합니다.
- 검색 품질이나 응답 포맷에 영향을 주는 변경은 가능하면 기존 디버그/회귀 스크립트로 검증합니다.

## 문서 사용 순서

저장소 문서는 다음 순서로 참고합니다.

1. `docs/AGENTS.md` — 공통 작업 제약과 아키텍처 가드레일
2. `docs/PLANS.md` — 현재 코드 구조와 변경 방향
3. `docs/BRANCH_REVIEW_GUIDE.md` — 브랜치 리뷰 기준
4. `docs/CHECKLIST.md` — 실제 점검 순서

브랜치 리뷰처럼 반복되는 검토 작업은 가능하면 `$reviewing-dongsooop-branch` skill을 먼저 사용합니다.

사용자가 "지금 브랜치 작업 검토해줘", "현재 변경 검토해줘", "푸시 전 점검해줘", "브랜치 리뷰해줘"처럼 현재 작업물 검토를 요청하면 제안만 하지 말고 실제 점검을 진행합니다. 이때 `docs/PLANS.md` → `docs/BRANCH_REVIEW_GUIDE.md` → `docs/CHECKLIST.md` 순서로 확인하고, 그 기준에 따라 변경 범위, 보안, 기능, 공개 가능 여부, Docker/의존성, 환경변수 정합성, 남은 리스크를 빠짐없이 검토합니다.

브랜치 검토 응답에는 `.github/pull_request_template.md` 형식에 맞춘 PR 본문 초안도 함께 작성합니다. 현재 브랜치 변경 내용을 기준으로 `관련 이슈`, `🎯 배경`, `🔍 주요 내용` 섹션을 채우며, 이슈 번호를 알 수 없으면 `Closes #이슈번호` 자리표시자를 유지합니다. 마지막에는 현재 변경에 어울리는 커밋 메시지 후보 3개를 함께 제안합니다.

커밋 메시지 후보는 `type: english summary` 형식을 따릅니다. type은 변경 성격에 맞게 `feat`, `fix`, `refactor`, `chore`, `docs` 중에서 우선 선택하고, summary는 짧은 영어 소문장으로 작성합니다. 예시는 `feat: chatbot model version upgrade`, `refactor: exception handling and logging standardization`, `chore: sensitive information error log ignore`, `docs: harness engineering file ignore`처럼 작성합니다.

## 서비스 실행

```bash
# 텍스트 필터링 + 시간표 분석 서비스 (포트 8888)
uvicorn main:app --reload --port 8888

# 챗봇 OSS 서비스 (포트 8010)
uvicorn app_oss_main:app --reload --port 8010
```

## Docker 빌드 및 배포

```bash
# ARM64 (Oracle/Apple Silicon)
docker buildx build --platform linux/arm64 -t yjs0120/spring-boot-app:text_filter_arm . --push
docker buildx build --platform linux/arm64 -f Dockerfiles_oss -t yjs0120/spring-boot-app:chatbot_oss_arm . --push

# AMD64
docker buildx build --platform linux/amd64 -t yjs0120/spring-boot-app:text_filter . --push

# 실행
docker-compose up -d text_filter_api
docker compose -f docker-compose-oss.yml up -d
```

## 아키텍처

### 독립된 두 개의 FastAPI 서비스

**서비스 1: `main.py`** (포트 8888)

- `text_filtering/text_filtering.py` — ML 기반 비속어 필터
- `text_filtering/text_filtering_rule.py` — 규칙 기반 필터
- `image_analysis/timetable_analysis.py` — Tesseract OCR + OpenCV 격자 탐지 기반 시간표 분석

**서비스 2: `app_oss_main.py`** (포트 8010)

- `LLM/OSS/Open_AI_OSS.py` — 챗봇 메인 로직, OpenAI SDK 형태로 Ollama 래핑
- `LLM/sub_model/query_index.py` — 학교 문서 대상 Hybrid BM25 + 의미 검색
- `LLM/sub_model/schedule_index.py` — 학사일정 파싱 및 이벤트 태깅
- `LLM/sub_model/index_utils.py` — 한국어 NLP 유틸(KoNLPy/Okt), 청킹, 연락처 추출
- `LLM/sub_model/build_index.py` — 원본 학교 데이터로 검색 아티팩트 생성

### 챗봇 질의 흐름

```text
User Query → Greeting/Intent Detection → Hybrid Search (BM25 + embeddings)
→ RAG Context Retrieval → Ollama LLM (gpt-oss:20b) → Safety Filters → Response
```

### 생성된 모델 아티팩트 (`model/artifacts/`)

- `search_df.parquet` — 검색용 문서 데이터프레임
- `embeddings.npy` — Dense 임베딩 벡터
- `bm25.pkl` — BM25 인덱스
- `tokenized_corpus.json.gz` — BM25용 토크나이즈 코퍼스
- `contact_docs.csv` — 교직원 연락처 데이터

데이터가 바뀌면 `LLM/sub_model/build_index.py`로 아티팩트를 다시 생성합니다.

## 주요 환경변수 (`.env`)

| Variable                                  | Purpose                                         |
| ----------------------------------------- | ----------------------------------------------- |
| `OSS_BASE_URL`                            | Ollama server URL (`http://localhost:11434/v1`) |
| `OSS_MODEL`                               | LLM model name (`gpt-oss:20b`)                  |
| `SECRET_KEY` / `ALGORITHM`                | JWT auth (HS512)                                |
| `DB_NAME/USER/PASSWORD`                   | PostgreSQL credentials                          |
| `DATA_JSON`                               | School info corpus path                         |
| `SEARCH_DF_PATH`, `EMB_PATH`, `BM25_PATH` | Artifact paths                                  |
| `SCHEDULE_CSV_PATH`                       | Academic calendar CSV                           |
| `STRICT_SOURCE_DATA_ONLY`                 | Restrict chatbot to known sources               |

## 외부 의존성 (저장소 외부)

- **Ollama** — 챗봇 서비스 실행 시 로컬 또는 지정 주소에서 동작해야 함
- **Tesseract OCR** + 한국어 언어팩 (`tesseract-ocr-kor`) — 시간표 분석에 필요
- **PostgreSQL 17** — `database/` 스크립트 관련
- **Java JDK** — 챗봇의 KoNLPy/JPype1 사용에 필요

## 인증

모든 API 엔드포인트는 JWT Bearer 토큰이 필요합니다. 토큰은 `.env`의 base64 인코딩 비밀키와 HS512 알고리즘을 사용합니다.

## 테스트 및 디버깅

```bash
# 챗봇 회귀 테스트
python debug/regression/run_chatbot_regression.py

# 시간표 디버그
python debug/timetable/debug_timetable.py
```

## 데이터 파이프라인

크롤링 → 가공 → 인덱스 생성 → 배포 순서로 진행됩니다.

- `crawling/` — 동양미래대학교 사이트 크롤링 스크립트
- `database/aws_insert_database.py` — AWS RDS PostgreSQL 적재 스크립트
- `finetuning/text_filter/` — ELECTRA 비속어 모델 재학습 관련 코드
