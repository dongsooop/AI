# 운영 알림과 점검 흐름

이 문서는 SLO 후보 또는 릴리즈 품질 게이트가 실패했을 때 어떤 알림, 점검, 배포 판단 흐름으로 이어질지 정리합니다.

현재 단계에서는 장애 알림 시스템이나 배포 차단 자동화를 구현하지 않습니다. 장애 시스템이 붙은 뒤에도 같은 기준으로 판단할 수 있도록 신호와 액션의 연결 규칙을 먼저 고정합니다.

## 적용 범위

| 구분 | 대상 | 주요 입력 | 기본 액션 |
| --- | --- | --- | --- |
| 운영 장애 | availability, 5xx, readiness `not_ready` | `/ready`, request log | 즉시 점검 |
| degraded | fallback, queue pressure, BM25 fallback tier, latency 증가 | runtime log, readiness payload | 원인 확인 및 추세 관찰 |
| 품질 회귀 | RAG/OCR/텍스트 필터 회귀 지표 후퇴 | regression report | 배포 전 리뷰 또는 수정 |
| 공개 리스크 | raw report, query/answer, 원문, 로컬 경로 포함 | report artifact, PR diff | 공개 전 제거 또는 sanitized summary 대체 |

운영 장애와 품질 회귀를 같은 장애로 표현하지 않습니다. 운영 장애는 현재 서비스 상태를 기준으로 보고, 품질 회귀는 배포 전 검증과 리뷰 기준으로 봅니다.

## 신호별 처리 흐름

| 신호 | 상태 분류 | 1차 확인 | 후속 액션 |
| --- | --- | --- | --- |
| `/ready` `not_ready` | 장애 | 컴포넌트별 `status`, `required`, `error_code` 확인 | 필수 의존성 복구, 배포 보류 |
| request 5xx 증가 | 장애 | `service`, `path`, `request_id`, exception log 확인 | 최근 배포/설정/의존성 변경 확인 |
| request p95 latency 증가 | degraded 또는 장애 | path별 latency, runtime event별 latency 분해 | OCR/LLM/retrieval 등 원인 컴포넌트 확인 |
| `status=failed` runtime event 증가 | 장애 후보 | `event`, `component`, `operation`, `error_code` 확인 | 기능별 복구 또는 rollback 검토 |
| `fallback=true` 또는 `status=fallback` 증가 | degraded | `fallback_reason`별 count 확인 | fallback tier, 모델/아티팩트/의존성 상태 확인 |
| queue pressure 증가 | degraded | `/ready.components.timetable.queue_size` 확인 | OCR worker 상태와 처리 시간 확인 |
| 품질 report `status=failed` | 품질 회귀 | 서비스별 핵심 metrics와 examples 확인 | 배포 보류 또는 승인 리뷰 |
| raw report 공개 위험 | 공개 리스크 | query, answer, 원문, 로컬 경로, error payload 포함 여부 확인 | raw report 제거, sanitized summary만 공유 |

## 운영 장애 점검 순서

운영 장애로 분류되는 신호가 발생하면 다음 순서로 확인합니다.

1. 어떤 FastAPI 서비스인지 먼저 분리합니다: `main-api` 또는 `chatbot-api`.
2. `/ready` payload에서 `status`, `components.*.status`, `required`, `error_code`를 확인합니다.
3. request log에서 `status_code`, `duration_ms`, `path`, `request_id`를 확인합니다.
4. 같은 `request_id` 또는 같은 시간대의 runtime log에서 `event`, `component`, `operation`, `status`, `fallback_reason`, `error_code`를 확인합니다.
5. 최근 배포, 환경변수, 모델/아티팩트, 외부 의존성 변경 여부를 확인합니다.
6. 장애가 사용자 요청 실패로 이어지면 배포를 보류하고 복구 또는 rollback을 검토합니다.

## 서비스별 점검 기준

### main-api

| 영역 | 확인할 신호 | 점검 포인트 |
| --- | --- | --- |
| 공통 API | 5xx, p95 latency, `request_failed` | 인증/예외 핸들러, 업로드 제한, path별 오류 |
| text_filter | `components.text_filter.status`, `text_filter_ml_runtime` | 모델 파일, weight, 영어 사전, `model_unavailable`, fallback/override |
| timetable | `components.timetable.status`, `timetable_job_runtime`, `timetable_ocr_engine_runtime` | Spring URL, worker 수, queue pressure, OCR latency, fallback cell |

OCR latency는 로컬 측정값만으로 장애 판단하지 않습니다. 실제 OCI 또는 OCI-constrained baseline과 비교합니다.

### chatbot-api

| 영역 | 확인할 신호 | 점검 포인트 |
| --- | --- | --- |
| readiness | `rule_book`, `query_index`, `chat_log_db` | 필수 컴포넌트 `not_ready`, log DB best-effort/degraded 구분 |
| LLM | `chatbot_llm_runtime` | Ollama 응답 실패, 빈 응답, timeout, fallback reason |
| retrieval | `chatbot_retrieval_runtime` | retrieval failed, `bm25_fallback_tier`, result_count, requested_top_k |
| request summary | `chatbot_request_summary` | mode, engine, cache_hit, direct_answer_route, total latency |

`chat_log_db`는 기본 best-effort입니다. `CHATBOT_LOG_DB_REQUIRED=1`일 때만 DB logging unavailable을 startup/ready 장애로 봅니다.

## 품질 게이트 점검 순서

품질 게이트 실패 또는 후퇴가 보이면 다음 순서로 확인합니다.

1. 영향 서비스가 chatbot, timetable, text_filtering 중 어디인지 분리합니다.
2. 관련 회귀 리포트의 `status`, `total`, `passed`, `failed`, `skipped`, 핵심 `metrics`를 확인합니다.
3. 이전 baseline 또는 직전 정상 리포트와 비교합니다.
4. 실패 케이스의 공개 가능성을 확인합니다. raw query, answer, 원문, 로컬 경로, error payload는 공개하지 않습니다.
5. 운영 SLO 위반인지, 배포 전 품질 회귀인지 분리해 표현합니다.
6. 품질 회귀가 명확하면 배포 보류, 수정, 또는 명시 승인 중 하나를 선택합니다.

## 배포 판단 기준

| 상태 | 배포 판단 | 비고 |
| --- | --- | --- |
| 운영 장애 진행 중 | 보류 | `/ready not_ready`, 5xx 증가, 필수 runtime failure |
| degraded 증가 | 조건부 보류 | fallback tier, latency, queue pressure 원인 확인 후 판단 |
| 품질 게이트 관측 후퇴 | 리뷰 필요 | baseline 수집 기간이면 자동 차단하지 않음 |
| 품질 게이트 차단 기준 초과 | 보류 | 별도 합의된 threshold가 있을 때만 적용 |
| raw report 공개 위험 | 보류 | sanitized summary로 대체 후 진행 |
| 경미한 문서/집계 기준 변경 | 가능 | secret check, diff check 후 진행 |

자동 차단 기준은 충분한 baseline과 운영 합의가 생긴 뒤에만 도입합니다. 지금 문서 기준에서는 품질 게이트의 기본 강도는 관측입니다.

## 알림 설계 후보

장애 시스템이 구축되면 다음 알림부터 작게 시작합니다.

| 알림 후보 | 입력 | 우선순위 | 비고 |
| --- | --- | --- | --- |
| service readiness not_ready | `/ready` | 높음 | 서비스별, 컴포넌트별 분리 |
| request 5xx rate 증가 | request log | 높음 | 4xx는 제외 |
| request p95 latency 증가 | request log | 중간 | 서비스/path별 baseline 필요 |
| runtime failed rate 증가 | runtime log | 높음 | `component`, `operation`, `error_code` 포함 |
| fallback/degraded rate 증가 | runtime log | 중간 | `fallback_reason`, BM25 tier 포함 |
| OCR queue pressure 증가 | readiness payload | 중간 | worker/queue 상태 확인 |
| 품질 게이트 실패 | regression report | 낮음 또는 수동 | 배포 전 리뷰 중심 |

품질 게이트 알림은 운영 장애 알림 채널과 분리하는 것이 좋습니다. 운영 중 사용자 요청 실패와 배포 전 품질 회귀는 대응 속도와 담당 판단이 다릅니다.

## 수동 실행과 자동화 경계

| 작업 | 기본 운영 방식 | 이유 |
| --- | --- | --- |
| lightweight validation | PR 또는 수동 workflow | 빠르고 항상 실행 가능 |
| full RAG evaluation | manual/self-hosted 또는 로컬 수동 실행 | 모델/아티팩트/runner 의존성이 큼 |
| OCR performance baseline | OCI 또는 OCI-constrained 수동 실행 | 환경별 성능 차이가 큼 |
| text filtering quality report | 로컬/CI 수동 또는 필요 시 strict | 모델/케이스 변화에 따른 해석 필요 |
| incident alerting | 장애 시스템 구축 후 연결 | 현재 단계는 기준 문서화 |

무거운 모델 기반 검증을 항상 자동 실행하는 방식은 피합니다. runner, artifact, 네트워크, OCI 성능 조건이 준비된 뒤 manual 또는 self-hosted 흐름으로 연결합니다.

## 기록 형식

장애 또는 품질 회귀를 기록할 때는 다음 정보를 남깁니다.

| 항목 | 예시 |
| --- | --- |
| 분류 | 운영 장애, degraded, 품질 회귀, 공개 리스크 |
| 서비스 | `main-api`, `chatbot-api`, `text_filtering`, `timetable`, `chatbot` |
| 신호 | `/ready not_ready`, `status=failed`, `fallback_reason=...`, report `status=failed` |
| 영향 | 사용자 요청 실패, 지연 증가, 배포 보류, 품질 후퇴 |
| 근거 | request id, runtime event, report path, sanitized summary |
| 조치 | 복구, rollback, baseline 재측정, 회귀 수정, 승인 후 진행 |

원문 사용자 입력, 토큰, 비밀번호, SSH 정보, 내부 서버 주소는 기록하지 않습니다.

## 다음 단계

이 문서까지가 SLO 적용 전 운영 기준 6단계의 문서화 범위입니다. 실제 장애 시스템을 구축할 때는 이 문서를 기준으로 알림 rule, dashboard, manual workflow, sanitized summary artifact를 작게 연결합니다.
