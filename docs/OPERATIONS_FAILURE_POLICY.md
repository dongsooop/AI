# 장애 기준 정책

이 문서는 동숲 백엔드에 SLO를 적용하기 전에 먼저 합의해야 하는 장애 판단 기준을 정리합니다.

현재 단계의 목적은 장애 감지 시스템이나 SLO 수치를 바로 구현하는 것이 아닙니다. `/health`, `/ready`, request log, runtime log, 회귀 리포트가 각각 어떤 운영 신호로 쓰이는지 구분하고, 이후 SLI/SLO 정의가 같은 언어를 사용하도록 기준을 고정합니다.

## 적용 범위

동숲은 두 개의 FastAPI 서비스를 독립적으로 운영합니다.

| 서비스 | 진입점 | 주요 기능 | 기본 포트 |
| --- | --- | --- | --- |
| main-api | `main.py` | 텍스트 필터링, 시간표 OCR | `8888` |
| chatbot-api | `app_oss_main.py` | OSS LLM 기반 학교 정보 RAG 챗봇 | `8010` |

장애 기준은 두 서비스를 섞어서 판단하지 않습니다. 챗봇 장애는 기본적으로 `chatbot-api` 기준으로 보고, 텍스트 필터링 또는 OCR 장애는 `main-api` 안에서도 컴포넌트를 분리해서 봅니다.

## 상태 구분

운영 상태는 다음 세 가지로 구분합니다.

| 상태 | 의미 | 예시 | 운영 판단 |
| --- | --- | --- | --- |
| 장애 | 필수 기능이 정상 요청을 처리할 수 없음 | `/ready`가 `not_ready`, 필수 모델 또는 인덱스 unavailable, 지속적인 5xx | 장애 감지 및 복구 대상 |
| degraded | 핵심 요청은 처리되지만 보조 의존성 또는 fallback이 발생함 | 챗봇 로그 DB best-effort 실패, BM25 runtime fallback, OCR fallback 증가 | 경고 및 추세 관찰 대상 |
| 품질 회귀 | 서비스는 응답하지만 AI 결과 품질이 기준보다 후퇴함 | RAG 출처 정확도 하락, OCR 인식률 하락, 텍스트 필터 false negative 증가 | 릴리즈 게이트 또는 회귀 분석 대상 |

SLO 1차 적용 대상은 장애와 degraded입니다. 품질 회귀는 운영 장애 알림과 분리하고, 배포 전 회귀 리포트와 품질 게이트에서 관리합니다.

## 헬스체크 기준

### `/health`

`/health`는 프로세스가 살아 있고 FastAPI 라우팅이 가능한지 확인하는 liveness 성격의 엔드포인트입니다.

- `200 OK`이면 프로세스는 살아 있다고 봅니다.
- 의존성 준비 상태를 보장하지 않습니다.
- SLO의 availability 판단에는 단독으로 사용하지 않습니다.

### `/ready`

`/ready`는 요청을 받을 준비가 되었는지 확인하는 readiness 성격의 엔드포인트입니다.

- `200 OK`와 `status: ready`이면 필수 의존성이 준비된 상태입니다.
- `503` 또는 `status: not_ready`이면 장애 감지 대상입니다.
- 컴포넌트별 `status`, `required`, `error_code`가 있으면 장애 원인 분류에 사용합니다.
- `degraded` 컴포넌트는 즉시 장애로 보지 않고 경고 또는 추세 관찰 대상으로 둡니다.

## 서비스별 장애 기준

### main-api 공통

`main-api`는 텍스트 필터링과 시간표 OCR을 함께 제공하지만, 두 기능의 장애 원인은 분리해서 봅니다.

| 컴포넌트 | 장애 기준 | degraded 기준 | 비고 |
| --- | --- | --- | --- |
| request layer | 지속적인 5xx, 인증/예외 핸들러의 비정상 동작 | p95 지연 증가 | request log의 `status_code`, `duration_ms`를 사용 |
| text_filter | 모델 설정 파일, 모델 weight, 영어 사전 누락으로 `not_ready` | 규칙 fallback 또는 override 증가 | ML 판정 품질은 회귀 리포트에서 별도 관리 |
| timetable | Spring URL 미설정, queue worker 부족, OCR 처리 불가 | fallback cell 증가, queue size 증가 | OCR 성능 기준은 OCI 환경 측정값과 분리 |

텍스트 필터링의 false positive/false negative, OCR 인식률 하락은 품질 회귀로 분류합니다. 서비스가 요청을 정상 처리하는 한 운영 장애로 바로 보지 않습니다.

### chatbot-api

`chatbot-api`는 RAG 검색, 규칙/도구 라우팅, Ollama 호환 LLM 호출, 로그 DB 저장을 함께 사용합니다.

| 컴포넌트 | 장애 기준 | degraded 기준 | 비고 |
| --- | --- | --- | --- |
| rule_book | startup 실패 또는 readiness `not_ready` | 해당 없음 | 요청 라우팅에 필요한 필수 컴포넌트 |
| query_index | 검색 문서/임베딩 불일치, 인덱스 로딩 실패 | BM25 pickle 누락 후 tokenized/runtime fallback | 검색 품질 지표는 회귀 리포트에서 별도 관리 |
| LLM/Ollama | LLM 호출 실패, 빈 응답 반복, timeout 증가 | deterministic fallback 또는 grounded fallback 증가 | `fallback_reason`으로 원인 분류 |
| chat_log_db | `CHATBOT_LOG_DB_REQUIRED=1`에서 DB unavailable | 기본 설정에서 DB unavailable | 기본은 best-effort이므로 즉시 장애로 보지 않음 |

챗봇의 `source_url_pass_rate`, hallucination proxy, top-k URL 정확도는 품질 회귀 지표입니다. 운영 중 즉시 장애로 처리하기보다 배포 전 회귀 확인과 원인 분석에 사용합니다.

## 로그 기준

장애 판단에 사용할 기본 로그 신호는 다음과 같습니다.

| 로그 | 주요 필드 | 용도 |
| --- | --- | --- |
| request log | `method`, `path`, `status_code`, `duration_ms`, `request_id` | 공통 availability, error rate, latency 판단 |
| runtime log | `component`, `operation`, `status`, `duration_ms`, `fallback`, `fallback_reason`, `error_code` | 컴포넌트별 실패/fallback 원인 분류 |
| readiness payload | `status`, `components.*.status`, `required`, `error_code` | 장애 감지 시스템의 1차 판정 입력 |
| regression report | `status`, `pass_rate`, `metrics`, `errors` | 품질 회귀와 릴리즈 게이트 판단 |

운영 로그에는 원문 사용자 입력, 토큰, 비밀번호, SSH 정보, 내부 서버 주소를 추가하지 않습니다. 텍스트 필터링 또는 챗봇 요청 분석이 필요하면 hash, 길이, 카운트, category 같은 비식별 필드를 우선 사용합니다.

## SLO 이전 결정 사항

SLO 수치를 선언하기 전에 다음 기준을 먼저 만족해야 합니다.

- 장애 감지 시스템이 `/ready`와 5xx를 서비스별로 분리해서 볼 수 있어야 합니다.
- request log에서 서비스별 error rate와 p95 latency를 집계할 수 있어야 합니다.
- runtime log에서 fallback rate와 `fallback_reason`을 컴포넌트별로 집계할 수 있어야 합니다.
- OCR latency 기준은 로컬 Mac 측정값이 아니라 OCI 또는 OCI-constrained 기준선을 사용해야 합니다.
- AI 품질 지표는 운영 장애 SLO가 아니라 릴리즈 품질 게이트로 분리해야 합니다.

## 다음 단계

이 문서가 1단계 기준입니다. 이후 단계에서는 다음 순서로 확장합니다.

1. 서비스별 SLI 후보와 수집 위치를 `docs/OPERATIONS_SLI_MAP.md`에 매핑합니다.
2. 기존 로그에서 SLI를 집계할 수 있는지 확인합니다.
3. 장애 시스템 구축 후 적용할 1차 운영 SLO 후보를 선언합니다.
4. AI 품질 지표를 릴리즈 품질 게이트 문서로 분리합니다.
5. SLO 위반 시 알림, 점검, 배포 판단 흐름을 연결합니다.
