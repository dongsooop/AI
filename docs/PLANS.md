# PLANS.md

이 문서는 이 저장소에서 작업하는 GPT 기반 에이전트를 위한 아키텍처 계획서이자 코드 구조 안내서입니다.

이 문서는 고정된 최종 폴더 구조를 선언하는 문서가 아닙니다.

대신 다음 3가지를 추적합니다.

1. 현재 실제로 동작하는 구조
2. 앞으로 지향하는 목표 구조
3. 현재 구조에서 더 나은 구조로 안전하게 옮기기 위한 전환 규칙

## 문서 목적

**동숲 (Dongsooop)** 백엔드를 다음 3가지 핵심 기능 중심으로 유지보수하고 확장하는 것이 목적입니다.

1. 텍스트 필터링
2. 시간표 OCR 분석
3. 학교 정보 챗봇

## 이 문서를 보는 방법

- 현재 폴더 구조는 문서가 업데이트된 시점의 스냅샷으로 이해합니다.
- 현재 구조가 최종 목표 구조라고 가정하지 않습니다.
- 브랜치에서 모듈 경계, 엔트리포인트, 책임 구분이 바뀌면 같은 브랜치에서 `docs/PLANS.md`도 함께 수정합니다.
- 이 문서는 리팩토링을 막는 규칙 문서가 아니라, 현재 상태와 앞으로의 방향을 이해하기 위한 문서로 사용합니다.

## 현재 아키텍처

현재 저장소는 **서로 독립된 두 개의 FastAPI 서비스**를 중심으로 구성되어 있습니다.

### 서비스 A: 텍스트 필터링 + 시간표 분석

- 진입점: `main.py`
- 기본 포트: `8888`
- 역할:
  - 비속어 필터링
  - 시간표 이미지 분석

### 서비스 B: OSS 챗봇

- 진입점: `app_oss_main.py`
- 기본 포트: `8010`
- 역할:
  - RAG 기반 질의응답
  - 검색/인덱스 활용
  - 응답 포맷팅 및 후처리

## 현재 구조 스냅샷

아래 구조는 **현재 실제 저장소 상태**를 설명하는 것이며, 최종 목표 구조를 뜻하지는 않습니다.

```text
.
├── main.py                         # 텍스트 필터링 + 시간표 분석 FastAPI 진입점
├── app_oss_main.py                 # 챗봇 OSS FastAPI 진입점
├── core/
│   ├── auth.py                     # JWT 인증 처리
│   ├── exceptions/                # 공통 예외 타입 및 FastAPI 예외 핸들러
│   ├── logging/                   # 공통 로깅 설정, 요청 ID, 미들웨어
│   ├── settings.py                 # 애플리케이션 설정 로딩
│   └── __init__.py
├── text_filtering/
│   ├── service.py                  # 필터링 기능 API/서비스 연결부
│   ├── text_filtering.py           # ML 기반 비속어 판별
│   ├── text_filtering_rule.py      # 규칙 기반 비속어 필터링
│   └── bad_text_learning_model.py  # 모델 관련 보조 로직
├── image_analysis/
│   ├── service.py                  # 이미지 분석 API/서비스 연결부
│   ├── ocr_engine.py               # OCR 보조 로직
│   └── timetable_analysis.py       # 시간표 추출 파이프라인
├── LLM/
│   ├── OSS/
│   │   ├── Open_AI_OSS.py          # 챗봇 메인 오케스트레이션
│   │   ├── service.py              # 챗봇 서비스 연결부
│   │   ├── formatter.py            # 응답 포맷팅
│   │   ├── modes.py                # 챗봇 모드 정의
│   │   └── postprocess/
│   │       ├── engine.py           # 후처리 파이프라인 엔진
│   │       ├── context.py          # 후처리 컨텍스트 모델
│   │       ├── registry.py         # 후처리 레지스트리
│   │       ├── message_table.py    # 메시지 정규화 데이터
│   │       ├── rules_table.py      # 후처리 규칙
│   │       ├── synonym_table.py    # 동의어 처리 데이터
│   │       └── __init__.py
│   ├── sub_model/
│   │   ├── query_index.py          # 하이브리드 검색
│   │   ├── schedule_index.py       # 학사일정 인덱싱
│   │   ├── schedule_rules.py       # 학사일정 태그/검색 키워드 규칙
│   │   ├── index_utils.py          # 한국어 NLP/청킹 유틸
│   │   └── build_index.py          # 아티팩트 생성
│   ├── patterns.py                 # 챗봇 검색/포맷팅 공통 정규식
│   └── rule_book/
│       ├── index.py                # 규칙 진입점
│       ├── graph.py                # 규칙 관계 / 흐름
│       ├── logger.py               # 규칙 로깅
│       └── __init__.py
├── model/                          # 모델 파일 및 생성 아티팩트
├── data/                           # 원본 및 가공된 학교 데이터
├── crawling/                       # 크롤링 파이프라인
├── database/                       # 데이터베이스 스크립트
├── debug/                          # 회귀 테스트 및 디버깅 스크립트
├── Dockerfile
├── Dockerfiles_oss
├── README.md
├── .codex/
│   └── skills/
│       └── reviewing-dongsooop-branch/
│           ├── SKILL.md            # Codex 브랜치 리뷰 skill
│           └── agents/
│               └── openai.yaml     # Codex UI 메타데이터
├── docs/
│   ├── AGENTS.md                   # GPT 에이전트 공통 작업 지침
│   ├── PLANS.md                    # 현재 구조와 목표 구조 계획서
│   ├── BRANCH_REVIEW_GUIDE.md      # 브랜치 리뷰 기준 문서
│   └── CHECKLIST.md                # 실제 점검 순서 체크리스트
└── CLAUDE.md                       # Claude 호환용 저장소 참고 문서
```

## 현재 핵심 영역

### API 진입점

- `main.py`
- `app_oss_main.py`

### 공통 코어

- `core/settings.py`
- `core/auth.py`
- `core/logging/`
- `core/exceptions/`

### 텍스트 필터링

- `text_filtering/text_filtering.py`
- `text_filtering/text_filtering_rule.py`
- `text_filtering/service.py`

### 시간표 OCR

- `image_analysis/timetable_analysis.py`
- `image_analysis/ocr_engine.py`
- `image_analysis/service.py`

### 챗봇 코어

- `LLM/OSS/Open_AI_OSS.py`
- `LLM/OSS/service.py`
- `LLM/OSS/formatter.py`
- `LLM/OSS/modes.py`

### 챗봇 후처리

- `LLM/OSS/postprocess/engine.py`
- `LLM/OSS/postprocess/context.py`
- `LLM/OSS/postprocess/registry.py`

### 검색 및 인덱싱

- `LLM/sub_model/query_index.py`
- `LLM/sub_model/schedule_index.py`
- `LLM/sub_model/index_utils.py`
- `LLM/sub_model/build_index.py`

현재 학교 정보 RAG 인덱스는 `dmu_documents_cleaned.json`의 납작한 홈페이지 텍스트를 그대로 고정 크기 청크로 자르지 않고, 다음 전처리 단계를 거쳐 생성합니다.

- `index_utils.py`에서 편집모드 꼬리말, `fnctId/fnctNo`, 다운로드 UI 문구 등 검색 노이즈를 제거합니다.
- 문서는 `intro`, `department`, `policy`, `history`, `privacy`, `contact`, `table_like`, `page` 같은 검색용 타입으로 분류합니다.
- `build_index.py`는 `chunk_document()` 결과를 사용해 `text_for_embedding`, `text_for_bm25`, `text_for_answer`를 분리 저장합니다.
- `search_df.parquet`에는 `breadcrumb`, `leaf_title`, `section_title`, `has_phone`, `has_date`, `has_credit`, `has_policy_keyword`, `is_privacy_old` 같은 메타데이터를 함께 저장합니다.
- `query_index.py`는 기존 `build_answer()` 인터페이스를 유지하되, 새 컬럼이 있으면 메타데이터 기반 가중치와 답변용 본문을 우선 사용하고 구버전 아티팩트에는 `text` 기반으로 fallback 합니다.
- `model/artifacts/` 아래 파일은 생성 산출물이므로 직접 편집하지 않고 `LLM/sub_model/build_index.py`로 재생성합니다.

## 목표 구조 방향

이 저장소는 계속 리팩토링되며 발전할 예정입니다. 목표는 현재 폴더 구조를 영구히 유지하는 것이 아니라, 더 명확하고 유지보수하기 좋은 구조로 점진적으로 이동하는 것입니다.

현재 지향점은 다음과 같습니다.

- 모듈이 이동하더라도 두 FastAPI 서비스의 경계를 분명히 유지
- 특히 챗봇 오케스트레이션처럼 지나치게 큰 파일을 점진적으로 분리
- API 계층, 도메인 로직, 인프라, 생성 산출물의 책임 구분을 더 명확하게 정리
- 런타임 코드와 데이터, 디버그 도구, 백업, 생성 산출물을 더 깔끔하게 분리
- 검색, 포맷팅, 후처리, OCR, 필터링 책임을 더 찾기 쉬운 구조로 정리

향후 가능한 구조 방향 예시는 다음과 같습니다.

- 서비스 중심으로 더 명확해진 폴더 구분
- 공통 모듈과 서비스 전용 모듈의 더 분명한 경계
- 큰 혼합 모듈보다 도메인 중심의 작은 패키지 구조
- 애플리케이션 코드, 인프라 설정, 오프라인 인덱스/빌드 스크립트의 분리

다만 아직 특정 폴더 이름 체계 하나로 고정한 상태는 아닙니다.

## 전환 규칙

구조 리팩토링 시에는 다음 원칙을 따릅니다.

1. 완벽한 폴더 구조보다 실제로 동작하는 소프트웨어를 우선합니다.
2. 아키텍처 자체를 바꾸는 것이 아니라면 `main.py`와 `app_oss_main.py`의 두 서비스 경계를 유지합니다.
3. 한 번에 대규모로 바꾸기보다 점진적으로 이동합니다.
4. 꼭 필요하지 않다면 기능 변경과 대규모 구조 변경을 같은 브랜치에 섞지 않습니다.
5. 파일 이동 시 import, 런타임 엔트리포인트, Docker 경로 가정, 환경변수 경로를 함께 점검합니다.
6. 책임 구분, 경계, 실행 경로가 바뀌면 `docs/PLANS.md`를 함께 수정하고 `docs/BRANCH_REVIEW_GUIDE.md`, `docs/CHECKLIST.md` 기준으로 리뷰합니다.
7. 생성 산출물과 프라이빗 데이터는 리팩토링 중에도 애플리케이션 소스와 분리된 상태를 유지합니다.

## 작업 우선순위

1. 두 서비스의 경계를 명확히 유지합니다.
2. OCR, 검색, 텍스트 필터링에서 한국어 처리 특성을 유지합니다.
3. 강한 이유가 없다면 챗봇 코어의 광범위한 리팩토링은 피합니다.
4. 생성 산출물 직접 수정 대신 생성 로직이나 원본 로직 수정을 우선합니다.
5. 위험도가 있는 변경은 가능하면 디버그/회귀 스크립트로 검증합니다.

## 운영상 알아둘 점

- 챗봇은 `OSS_BASE_URL`을 통한 Ollama 호환 API 접근에 의존합니다.
- 시간표 분석은 한국어 지원이 포함된 Tesseract OCR에 의존합니다.
- 일부 챗봇 인덱싱 유틸은 KoNLPy/JPype1 사용을 위해 Java가 필요합니다.
- 인증은 `.env`의 JWT 관련 값을 사용합니다.

## 권장 변경 전략

### 구조 리팩토링 작업 시

- 지금 변경이 현재 상태 문서화인지, 목표 구조로 이동하는 작업인지, 또는 둘 다인지 먼저 구분합니다.
- 폴더 미관보다 동작 안정성을 우선합니다.
- 엔트리포인트, 서비스 연결부, 도메인 로직, 보조 유틸 중 한 경계씩 나누어 이동합니다.
- 디렉터리 이동 후에는 Docker 파일, import, 환경변수 경로를 다시 확인합니다.
- 실제 구조가 의미 있게 바뀌었다면 이 문서도 함께 갱신합니다.

### 텍스트 필터링 작업 시

- `text_filtering/service.py`부터 확인합니다.
- 이후 `text_filtering.py` 또는 `text_filtering_rule.py`로 흐름을 따라갑니다.
- 변경이 ML 기반인지, 규칙 기반인지, 둘 다인지 확인합니다.

### 시간표 OCR 작업 시

- `image_analysis/service.py`부터 확인합니다.
- 다음으로 `timetable_analysis.py`를 봅니다.
- OCR 계층 자체가 문제일 때만 `ocr_engine.py`를 수정합니다.

### 챗봇 응답 품질 작업 시

- `LLM/OSS/Open_AI_OSS.py`부터 확인합니다.
- 그다음 `LLM/OSS/formatter.py`를 봅니다.
- 이후 `LLM/OSS/postprocess/`를 확인합니다.

### 검색 품질 작업 시

- `LLM/sub_model/query_index.py`부터 확인합니다.
- 토크나이징/청킹 영향은 `index_utils.py`를 봅니다.
- 코퍼스 로직이 바뀌면 `build_index.py`로 아티팩트를 다시 생성합니다.
- 인덱스 스키마가 바뀌면 `search_df.parquet`, `embeddings.npy`, `bm25.pkl`의 row 수가 서로 맞는지 확인합니다.
- 회귀 케이스는 `debug/regression/chatbot_regression_cases.json`에 추가하고, 실제 API 회귀 테스트는 OSS FastAPI 서버와 JWT 토큰이 준비된 상태에서 실행합니다.

## 실행 명령

```bash
uvicorn main:app --reload --port 8888
uvicorn app_oss_main:app --reload --port 8010
```

## 검증 참고 명령

```bash
# 챗봇 회귀 테스트는 먼저 OSS FastAPI 서버를 띄운 뒤 실행합니다.
uvicorn app_oss_main:app --reload --port 8010
python debug/regression/run_chatbot_regression.py

python debug/timetable/debug_timetable.py
```

## 문서 역할 관계

- `docs/AGENTS.md`: 공통 작업 원칙
- `docs/PLANS.md`: 현재 구조 스냅샷 + 목표 구조 방향 + 전환 규칙
- `docs/BRANCH_REVIEW_GUIDE.md`: 브랜치 리뷰 기준
- `docs/CHECKLIST.md`: 실제 점검 순서 체크리스트
- `CLAUDE.md`: Claude 호환을 위한 참고 문서
