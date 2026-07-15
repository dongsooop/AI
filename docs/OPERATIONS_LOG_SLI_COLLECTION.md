# 로그 기반 SLI 수집 준비

이 문서는 `docs/OPERATIONS_SLI_MAP.md`에서 정리한 SLI 후보를 현재 로그에서 어떻게 집계할지 정의합니다.

현재 단계에서는 `/metrics` 엔드포인트나 알림 시스템을 추가하지 않습니다. 먼저 request log, runtime log, readiness payload, regression report를 안정적인 SLI 입력으로 사용하는 규칙을 고정합니다.

## 입력 신호

| 입력 | 형식 | 주요 필드 | 사용 목적 |
| --- | --- | --- | --- |
| request log | logging formatter prefix + message | `service`, `request_id`, `method`, `path`, `status_code`, `duration_ms` | 공통 availability, error rate, latency |
| runtime log | key-value message | `event`, `component`, `operation`, `status`, `duration_ms`, `result_count`, `fallback`, `fallback_reason`, `error_code` | 기능별 실패, fallback, degraded 원인 |
| readiness payload | `/ready` JSON | `status`, `components.*.status`, `components.*.required`, `error_code` | 장애 감지 시스템의 1차 판정 |
| regression report | JSON summary | `status`, `pass_rate`, `metrics`, `errors` | 품질 게이트와 배포 전 회귀 판단 |

로그 formatter는 `service`와 `request_id`를 모든 로그 레코드에 붙입니다. request log와 runtime log를 함께 볼 때는 같은 `request_id`를 우선 사용하고, request 밖에서 실행되는 startup 또는 worker 로그는 `service`, `event`, `component`, `operation`으로 묶어 봅니다.

## 공통 집계 규칙

| SLI | 계산식 | 필터 | 판정 범위 |
| --- | --- | --- | --- |
| request count | `request_completed` 수 | `path`별, `service`별 | 운영 SLI |
| request error rate | `status_code >= 500` / `request_completed` | `/health`는 제외 가능, `/ready`는 별도 집계 | 운영 SLI |
| request p95 latency | `request_completed.duration_ms`의 p95 | `service`, `path`별 | 운영 SLI |
| request failure count | `request_failed` 수 | `service`, `path`별 | 운영 SLI |
| readiness success rate | `/ready` 응답이 HTTP 200이고 top-level `status=ready`인 비율 | `service`별 | 운영 SLI |

`/health`는 liveness 확인용입니다. 서비스 availability SLO를 만들 때는 `/health` 단독 성공률보다 `/ready`, 5xx, latency를 함께 봅니다.

## runtime log 집계 규칙

| SLI | 계산식 | 기준 로그 | degraded/장애 해석 |
| --- | --- | --- | --- |
| operation failed rate | `status=failed` / 해당 `event` 전체 | 모든 runtime event | 장애 후보 |
| operation fallback rate | `fallback=true` 또는 `status=fallback` / 해당 `event` 전체 | 모든 runtime event | degraded 후보 |
| operation p95 latency | `duration_ms`의 p95 | 모든 runtime event | 지연 degraded 후보 |
| result empty rate | `result_count=0` / 해당 `event` 전체 | request/retrieval/ocr event | 장애 또는 품질 회귀 후보, event별 해석 필요 |
| error code count | `error_code`별 count | 모든 runtime event | 원인 분류 |
| fallback reason count | `fallback_reason`별 count | 모든 runtime event | degraded 원인 분류 |

`status=failed`는 장애 후보로 보고, `status=fallback` 또는 `fallback=true`는 degraded 후보로 봅니다. 단, fallback이 사용자 요청 실패를 의미하지 않을 수 있으므로 request 5xx와 분리해서 집계합니다.

## 서비스별 집계 규칙

### main-api request layer

| SLI | 입력 | 집계 기준 |
| --- | --- | --- |
| main-api 5xx rate | request log | `service=main-api`, `status_code >= 500` |
| main-api p95 latency | request log | `service=main-api`, `path`별 p95 |
| upload size rejection count | request log | `status_code=413` |

`413`은 클라이언트 입력 제한에 따른 거절이므로 5xx error rate에 포함하지 않습니다. 다만 갑자기 증가하면 사용 패턴 변화 신호로 볼 수 있습니다.

### 텍스트 필터링

| SLI | 입력 | 집계 기준 |
| --- | --- | --- |
| ML filter failed rate | `text_filter_ml_runtime` | `status=failed` 비율 |
| ML filter fallback rate | `text_filter_ml_runtime` | `fallback=true` 또는 `status=fallback` 비율 |
| ML filter p95 latency | `text_filter_ml_runtime.duration_ms` | `sentence_count`, `field_count`와 함께 해석 |
| rule filter failed rate | `text_filter_rule_runtime` | `status=failed` 비율 |
| model unavailable count | `text_filter_ml_runtime.model_unavailable` | `true` count |
| english override count | `english_rule_override_count` | 합계 또는 request당 평균 |

텍스트 원문은 운영 로그 집계 대상으로 삼지 않습니다. 필요한 경우 hash, 길이, 문장 수, match count 같은 비식별 필드만 사용합니다.

### 시간표 OCR

| SLI | 입력 | 집계 기준 |
| --- | --- | --- |
| OCR request failed rate | `timetable_job_runtime` | `status=failed` 비율 |
| OCR engine fallback rate | `timetable_ocr_engine_runtime` | `fallback=true` 또는 `status=fallback` 비율 |
| OCR p95 total latency | `timetable_ocr_engine_runtime.duration_ms` | OCI 또는 OCI-constrained 기준선과 비교 |
| grid detection p95 latency | `grid_detection_duration_ms` | runtime event별 p95 |
| OCR task count | `ocr_task_cell_count` | 평균, p95, case별 추세 |
| skipped empty cell count | `skipped_empty_cell_count` | 평균, p95, 성능 변화 추세 |
| fallback cell count | `ocr_fallback_cell_count` | 평균, p95, degraded 추세 |
| queue pressure | `/ready.components.timetable.queue_size` | `queue_size / queue_max_size` |

OCR 성능 SLI는 로컬 측정값만으로 목표를 정하지 않습니다. 운영 기준은 실제 OCI 또는 OCI-constrained baseline과 비교합니다.

### chatbot-api request layer

| SLI | 입력 | 집계 기준 |
| --- | --- | --- |
| chatbot 5xx rate | request log | `service=chatbot-api`, `status_code >= 500` |
| chatbot p95 request latency | request log | `service=chatbot-api`, `path`별 p95 |
| chatbot summary p95 latency | `chatbot_request_summary.duration_ms` | `mode`, `engine`, `cache_hit`, `direct_answer_route`별 보조 분해 |

request log는 API 계층의 전체 지연을 보고, `chatbot_request_summary`는 챗봇 내부 처리 흐름을 봅니다. 두 값은 목적이 다르므로 같은 그래프에서 비교하되 서로 대체하지 않습니다.

### 챗봇 LLM/RAG

| SLI | 입력 | 집계 기준 |
| --- | --- | --- |
| LLM failed rate | `chatbot_llm_runtime` | `status=failed` 비율 |
| LLM fallback rate | `chatbot_llm_runtime` | `fallback=true` 또는 `status=fallback` 비율 |
| LLM p95 latency | `chatbot_llm_runtime.duration_ms` | `model`, `base_url_configured` 같은 설정 신호와 함께 해석 |
| retrieval failed rate | `chatbot_retrieval_runtime` | `status=failed` 비율 |
| retrieval p95 latency | `chatbot_retrieval_runtime.duration_ms` | `requested_top_k`, `result_count`와 함께 해석 |
| BM25 fallback tier rate | `bm25_fallback_tier` | `pickle`, `tokenized_corpus`, `runtime_tokenize`별 비율 |
| direct answer route rate | `chatbot_request_summary.direct_answer_route` | `true` 비율 |
| cache hit rate | `chatbot_request_summary.cache_hit` | `true` 비율 |

BM25 tier가 `pickle`이 아니면 degraded 후보로 봅니다. 특히 `runtime_tokenize`는 startup 또는 retrieval 성능 리스크가 커질 수 있으므로 별도 집계합니다.

## readiness 집계 규칙

| 서비스 | readiness 컴포넌트 | 장애 후보 | degraded 후보 |
| --- | --- | --- | --- |
| main-api | `text_filter` | `status=not_ready` | 해당 없음 |
| main-api | `timetable` | `status=not_ready` | queue pressure 증가 |
| chatbot-api | `rule_book` | `status=not_ready` | 해당 없음 |
| chatbot-api | `query_index` | `status=not_ready` | BM25 fallback tier로 별도 관찰 |
| chatbot-api | `chat_log_db` | `required=true`에서 `not_ready` | `required=false`에서 `degraded` |

readiness는 장애 감지의 빠른 신호입니다. 실제 사용자 영향은 request error rate, latency, runtime failed/fallback rate로 함께 확인합니다.

## 품질 게이트와 분리할 지표

다음 지표는 운영 SLI 입력으로 참고할 수 있지만, 1차 장애 알림이나 운영 SLO에는 바로 넣지 않습니다.

| 서비스 | 품질 지표 | 기본 처리 |
| --- | --- | --- |
| chatbot | `source_url_pass_rate`, `hallucination_proxy_rate`, `top3_url_accuracy` | 배포 전 RAG 회귀 리포트에서 판단 |
| timetable | `grid_detection_success_rate`, `average_ocr_confidence`, `extracted_course_count` | OCR diagnostics/performance report에서 판단 |
| text_filtering | `false_positive_count`, `false_negative_count`, `shadow_*` metrics | text filtering quality report에서 판단 |

이 지표들은 5단계에서 릴리즈 품질 게이트로 별도 정리합니다.

## 누락 또는 보강 후보

3단계 기준으로 즉시 코드 변경이 필요한 항목은 없습니다. 다만 실제 장애 시스템과 연결하기 전에 다음 보강을 검토합니다.

- request log의 `path` cardinality가 너무 커지지 않도록 path normalization이 필요한지 확인합니다.
- `/ready` JSON을 주기적으로 수집할 위치를 정합니다.
- `request_id`가 worker/background task 로그에도 필요한지 검토합니다.
- `status=fallback`과 `fallback=true`를 같은 degraded로 집계할지, 원인별로 분리할지 결정합니다.
- OCR p95 latency 기준선은 `tests/reports/timetable/timetable_ocr_performance_baseline.json` 또는 OCI 측정 리포트를 기준으로 고정합니다.
- 품질 게이트 리포트의 raw 상세 파일과 공개 가능한 summary 파일을 분리합니다.

## 다음 단계

다음 단계에서는 이 집계 규칙을 바탕으로 장애 시스템 구축 후 적용할 1차 운영 SLO 후보를 `docs/OPERATIONS_SLO_CANDIDATES.md`에 선언합니다. SLO 수치는 아직 임의로 박지 않고, baseline이 필요한 항목은 `TBD after baseline`로 남깁니다.
