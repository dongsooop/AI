# 운영 SLI 후보 맵

이 문서는 장애 시스템 구축 이후 SLO 후보로 확장할 수 있는 SLI(Service Level Indicator)를 정리합니다.

1단계 기준 문서인 `docs/OPERATIONS_FAILURE_POLICY.md`의 상태 구분을 따릅니다. 여기서는 SLO 목표 수치를 선언하지 않고, 어떤 지표를 어디서 얻을 수 있는지만 매핑합니다.

## 분류 원칙

| 분류 | 의미 | 사용 위치 |
| --- | --- | --- |
| 운영 SLI | 장애, degraded, latency, fallback처럼 운영 상태를 직접 설명하는 지표 | 장애 감지, 알림, SLO 후보 |
| 품질 게이트 지표 | AI 결과 품질이나 회귀 여부를 설명하는 지표 | 배포 전 회귀 리포트, 리뷰 |
| 준비 필요 지표 | 현재 로그나 리포트에 일부만 있고 집계 규칙이 더 필요한 지표 | 3단계 로그 기반 수집 준비 |

운영 SLI와 품질 게이트 지표를 섞지 않습니다. API가 정상 응답하더라도 RAG 출처 정확도나 OCR 인식 품질이 떨어질 수 있고, 반대로 품질 리포트가 통과하더라도 운영 중 의존성 장애가 발생할 수 있기 때문입니다.

## 공통 API SLI

| SLI 후보 | 분류 | 수집 위치 | 현재 사용 가능 여부 | 비고 |
| --- | --- | --- | --- | --- |
| readiness success rate | 운영 SLI | `/ready` 응답의 `status`, HTTP status | 가능 | `ready`는 필수 의존성 준비 상태 판단에 사용 |
| request error rate | 운영 SLI | request log의 `status_code` | 가능 | 5xx 비율을 서비스별로 집계 |
| request p95 latency | 운영 SLI | request log의 `duration_ms` | 가능 | `path`와 서비스 기준으로 분리 필요 |
| request failure count | 운영 SLI | `request_failed` log | 가능 | 예외 발생 요청 수 |
| liveness status | 참고 신호 | `/health` | 가능 | 프로세스 생존 확인용이며 availability SLO 단독 근거로 쓰지 않음 |

## main-api SLI

`main-api`는 텍스트 필터링과 시간표 OCR을 함께 제공하므로 기능별 SLI를 분리합니다.

### 텍스트 필터링

| SLI 후보 | 분류 | 수집 위치 | 현재 사용 가능 여부 | 비고 |
| --- | --- | --- | --- | --- |
| text filter readiness | 운영 SLI | `/ready`의 `components.text_filter.status` | 가능 | 모델 설정 파일, weight, 영어 사전 준비 여부 |
| ML filter failed rate | 운영 SLI | `text_filter_ml_runtime`의 `status=failed` | 가능 | `error_code`로 원인 분류 |
| ML filter p95 latency | 운영 SLI | `text_filter_ml_runtime.duration_ms` | 가능 | 요청 단위와 문장/필드 수를 함께 봄 |
| fallback/override rate | 운영 SLI | `text_filter_ml_runtime.fallback`, `fallback_reason`, `english_rule_override_count` | 가능 | fallback은 degraded 후보 |
| model unavailable count | 운영 SLI | `text_filter_ml_runtime.model_unavailable` | 가능 | readiness와 함께 원인 확인 |
| false positive/false negative | 품질 게이트 지표 | `tests/regression/text_filtering/check_text_filter_quality_report.py` 리포트 | 가능 | 운영 알림보다 배포 전 회귀 판단에 사용 |
| shadow detector coverage | 품질 게이트 지표 | text filtering quality report의 `shadow_*` metrics | 가능 | 운영 판정 변경 전 검토용 |

### 시간표 OCR

| SLI 후보 | 분류 | 수집 위치 | 현재 사용 가능 여부 | 비고 |
| --- | --- | --- | --- | --- |
| timetable readiness | 운영 SLI | `/ready`의 `components.timetable.status` | 가능 | Spring URL과 queue worker 상태 |
| OCR request failed rate | 운영 SLI | `timetable_job_runtime`, `timetable_ocr_engine_runtime.status` | 가능 | worker exception과 OCR failure 구분 |
| OCR p95 total latency | 운영 SLI | `timetable_ocr_engine_runtime.duration_ms` 또는 baseline report | 준비 필요 | 운영 기준은 OCI 또는 OCI-constrained 기준선 사용 |
| OCR fallback rate | 운영 SLI | `timetable_ocr_engine_runtime.fallback`, `ocr_fallback_cell_count` | 가능 | fallback cell 증가를 degraded 신호로 사용 |
| queue pressure | 운영 SLI | `/ready`의 `queue_size`, `queue_max_size` | 가능 | queue 포화 추세 확인 |
| skipped empty cell count | 운영 SLI | `skipped_empty_cell_count` | 가능 | 성능 최적화 효과 추적 |
| grid detection success rate | 품질 게이트 지표 | timetable diagnostics report | 가능 | OCR 품질 회귀 판단 |
| average OCR confidence | 품질 게이트 지표 | timetable diagnostics report | 가능 | 운영 알림보다 회귀 리포트 기준 |

## chatbot-api SLI

| SLI 후보 | 분류 | 수집 위치 | 현재 사용 가능 여부 | 비고 |
| --- | --- | --- | --- | --- |
| chatbot readiness | 운영 SLI | `/ready`의 top-level `status` | 가능 | rule book, query index, log DB 상태 포함 |
| rule book readiness | 운영 SLI | `/ready.components.rule_book.status` | 가능 | 필수 컴포넌트 |
| query index readiness | 운영 SLI | `/ready.components.query_index.status` | 가능 | 검색 문서와 임베딩 row 수 확인 |
| chat log DB degraded state | 운영 SLI | `/ready.components.chat_log_db.status` | 가능 | 기본 best-effort에서는 degraded로 취급 |
| chatbot total p95 latency | 운영 SLI | `chatbot_request_summary.duration_ms` | 가능 | request log와 함께 확인 |
| LLM failed/fallback rate | 운영 SLI | `chatbot_llm_runtime.status`, `fallback`, `fallback_reason` | 가능 | Ollama/LLM 장애 또는 빈 응답 추적 |
| retrieval failed rate | 운영 SLI | `chatbot_retrieval_runtime.status` | 가능 | 검색 예외와 empty result를 구분해 볼 수 있음 |
| retrieval p95 latency | 운영 SLI | `chatbot_retrieval_runtime.duration_ms` | 가능 | RAG 성능 추세 확인 |
| BM25 fallback tier rate | 운영 SLI | `chatbot_retrieval_runtime.bm25_fallback_tier` | 가능 | `pickle`, `tokenized_corpus`, `runtime_tokenize` tier별 집계 |
| direct answer route rate | 운영 SLI | `chatbot_request_summary.direct_answer_route` | 가능 | LLM 우회/도구 라우팅 추세 |
| cache hit rate | 운영 SLI | `chatbot_request_summary.cache_hit` | 가능 | 캐시 동작 추세 |
| source URL pass rate | 품질 게이트 지표 | `tests/regression/chatbot/evaluate_rag_retrieval.py` 리포트 | 가능 | 운영 알림보다 배포 전 품질 기준 |
| hallucination proxy rate | 품질 게이트 지표 | chatbot RAG report | 가능 | 출처 없는 답변 회귀 판단 |
| top-k URL accuracy | 품질 게이트 지표 | chatbot RAG report | 가능 | 검색 품질 회귀 판단 |

## 회귀 리포트 기반 품질 게이트

| 서비스 | 주요 지표 | 기본 용도 |
| --- | --- | --- |
| chatbot | `pass_rate`, `top1_url_accuracy`, `top3_url_accuracy`, `source_url_pass_rate`, `hallucination_proxy_rate`, `avg_retrieval_latency_ms`, `p95_retrieval_latency_ms` | RAG 품질 및 검색 성능 회귀 판단 |
| timetable | `grid_detection_success_rate`, `extracted_course_count`, `average_ocr_confidence`, `fallback_cell_count`, `p95_total_duration_ms` | OCR 품질 및 성능 기준선 비교 |
| text_filtering | `pass_rate`, `false_positive_count`, `false_negative_count`, `shadow_detected_false_negative_rate`, `shadow_false_positive_candidate_rate` | 모델/규칙 판정 품질 회귀 판단 |

품질 게이트 지표는 SLO 후보가 될 수 있지만, 1차 운영 SLO에는 바로 넣지 않습니다. 먼저 회귀 리포트 기준으로 안정적인 baseline을 확보한 뒤 배포 판단에 사용합니다.

## 3단계에서 확인할 집계 규칙

다음 항목은 SLI 후보는 정해졌지만, 실제 장애 시스템에서 쓰려면 집계 규칙을 더 정해야 합니다.

- 서비스와 path별 request error rate 집계 방식
- request log와 runtime log를 `request_id`로 함께 보는 방식
- `fallback=true`와 `status=fallback`을 degraded로 집계하는 기준
- OCR p95 latency를 로컬, OCI-constrained, 실제 OCI 중 어느 기준선과 비교할지
- chatbot LLM timeout, empty response, deterministic fallback을 같은 degraded로 볼지 분리할지
- 품질 게이트 리포트의 실패를 배포 차단으로 볼 임계값

이 규칙은 3단계의 로그 기반 SLI 수집 준비에서 구체화합니다.
