# 1차 운영 SLO 후보

이 문서는 장애 시스템 구축 후 동숲 백엔드에 적용할 1차 운영 SLO 후보를 정리합니다.

현재 단계에서는 SLO 목표 수치를 확정하지 않습니다. 실제 운영 로그, OCI 성능 기준선, 장애 감지 시스템의 수집 결과가 준비된 뒤 수치를 확정합니다. baseline이 필요한 항목은 `TBD after baseline`로 남깁니다.

## 적용 원칙

| 원칙 | 내용 |
| --- | --- |
| 작게 시작 | 모든 지표를 SLO로 올리지 않고 availability, latency, error, fallback 중심으로 시작합니다. |
| 서비스 분리 | `main-api`와 `chatbot-api`를 분리하고, `main-api` 안에서도 텍스트 필터링과 OCR을 나눠 봅니다. |
| 품질 분리 | RAG 정확도, OCR 인식 품질, 텍스트 필터 오탐/미탐은 1차 운영 SLO가 아니라 품질 게이트로 둡니다. |
| baseline 우선 | OCR/LLM latency처럼 환경 영향을 크게 받는 지표는 baseline 확보 전 목표 수치를 확정하지 않습니다. |
| degraded 가시화 | fallback, queue pressure, BM25 fallback tier 같은 성능/의존성 저하 신호는 availability와 분리해 봅니다. |

## 1차 SLO 후보 요약

| SLO 후보 | 서비스 | 대표 SLI | 목표 | 우선순위 |
| --- | --- | --- | --- | --- |
| API availability | 공통 | request error rate, readiness success rate | `TBD after baseline` | 높음 |
| API latency | 공통 | request p95 latency | `TBD after baseline` | 높음 |
| Chatbot runtime health | chatbot-api | LLM failed/fallback rate, retrieval failed rate | `TBD after baseline` | 높음 |
| Chatbot degraded mode | chatbot-api | BM25 fallback tier rate, fallback direct answer route rate | `TBD after baseline` | 중간 |
| OCR processing health | main-api/timetable | OCR request failed rate, OCR p95 total latency | `TBD after OCI baseline` | 높음 |
| OCR degraded mode | main-api/timetable | OCR fallback rate, queue pressure | `TBD after baseline` | 중간 |
| Text filter runtime health | main-api/text_filter | ML filter failed rate, model unavailable count | `TBD after baseline` | 중간 |

## 공통 API SLO 후보

### API availability

| 항목 | 내용 |
| --- | --- |
| 대상 | `main-api`, `chatbot-api` |
| 목적 | 사용자가 정상 API 요청을 보냈을 때 서버 장애 없이 응답하는지 확인 |
| SLI | `request error rate`, `readiness success rate` |
| 수집 위치 | request log, `/ready` |
| 목표 | `TBD after baseline` |
| 제외 | `/health` 단독 성공률, 4xx 사용자 입력 오류 |

`/health`는 프로세스 생존 확인에만 사용합니다. availability SLO는 `/ready`, 5xx, request failure를 함께 봅니다.

### API latency

| 항목 | 내용 |
| --- | --- |
| 대상 | `main-api`, `chatbot-api` |
| 목적 | API 계층의 사용자 체감 지연을 추적 |
| SLI | `request p95 latency` |
| 수집 위치 | request log의 `duration_ms` |
| 목표 | `TBD after baseline` |
| 제외 | 배치성 회귀 테스트 지표, 품질 리포트 pass rate |

서비스와 path별로 p95를 분리합니다. 챗봇과 OCR은 처리 특성이 다르므로 하나의 latency 목표로 묶지 않습니다.

## chatbot-api SLO 후보

### Chatbot runtime health

| 항목 | 내용 |
| --- | --- |
| 대상 | 챗봇 RAG 요청 |
| 목적 | LLM/Ollama, retrieval, rule book/query index 의존성 장애를 조기에 감지 |
| SLI | `LLM failed rate`, `LLM fallback rate`, `retrieval failed rate`, `chatbot readiness` |
| 수집 위치 | `chatbot_llm_runtime`, `chatbot_retrieval_runtime`, `/ready` |
| 목표 | `TBD after baseline` |
| 제외 | RAG 답변 품질 지표 |

LLM 빈 응답, timeout, exception은 request 5xx와 별도로 봅니다. fallback으로 사용자 응답을 복구했더라도 degraded 신호로 집계합니다.

### Chatbot degraded mode

| 항목 | 내용 |
| --- | --- |
| 대상 | 챗봇 검색/응답 경로 |
| 목적 | 장애는 아니지만 성능 또는 품질 리스크가 커지는 상태를 추적 |
| SLI | `BM25 fallback tier rate`, `fallback direct answer route rate` |
| 보조 신호 | `cache hit rate` |
| 수집 위치 | `chatbot_retrieval_runtime`, `chatbot_request_summary` |
| 목표 | `TBD after baseline` |
| 제외 | `source_url_pass_rate`, `hallucination_proxy_rate`, `top-k URL accuracy` |

`bm25_fallback_tier=runtime_tokenize`는 startup 또는 retrieval 성능 리스크가 크므로 `tokenized_corpus`와 분리해서 봅니다.

`direct_answer_route=true`만으로는 degraded로 판정하지 않습니다. 정상적인 의도 기반 라우팅을 제외하기 위해 `fallback=true`가 함께 기록된 direct answer route만 `mode`별 기준선과 비교합니다.

`cache hit rate`는 요청 유형과 cold cache 상태에 따라 달라질 수 있으므로 단독 degraded 판정에서 제외합니다. `mode`별 기준선과 비교하고 latency 또는 fallback 증가가 함께 나타나는지 확인하는 보조 신호로만 사용합니다.

## main-api SLO 후보

### OCR processing health

| 항목 | 내용 |
| --- | --- |
| 대상 | 시간표 OCR 요청 |
| 목적 | OCR 처리 실패와 사용자 체감 처리 시간을 추적 |
| SLI | `OCR request failed rate`, `OCR p95 total latency`, `timetable readiness` |
| 수집 위치 | `timetable_job_runtime`, `timetable_ocr_engine_runtime`, `/ready` |
| 목표 | `TBD after OCI baseline` |
| 제외 | 로컬 Mac 단독 측정값 |

OCR latency는 로컬 환경 기준으로 확정하지 않습니다. 실제 OCI 또는 OCI-constrained 기준선이 있어야 목표 수치를 정합니다.

### OCR degraded mode

| 항목 | 내용 |
| --- | --- |
| 대상 | 시간표 OCR 내부 처리 경로 |
| 목적 | fallback 증가와 queue 압박처럼 성능 저하 신호를 추적 |
| SLI | `OCR fallback rate`, `fallback cell rate`, `queue pressure` |
| workload 참고 신호 | `OCR task workload` (`ocr_task_cell_count`) |
| 수집 위치 | `timetable_ocr_engine_runtime`, `/ready.components.timetable` |
| 목표 | `TBD after baseline` |
| 제외 | `grid_detection_success_rate`, `average_ocr_confidence` |

fallback 증가는 즉시 장애가 아닐 수 있지만, 처리 시간이 길어지거나 인식 품질 리스크가 커질 수 있어 degraded 후보로 봅니다.

`OCR task workload`는 요청량 또는 이미지 복잡도를 설명하는 값입니다. 요청별 task 수와 함께 fallback·queue pressure·latency를 정규화하거나 분해할 때만 사용하며, task 수 증가만으로 degraded를 판정하지 않습니다.

`fallback cell rate`는 `sum(ocr_fallback_cell_count) / sum(ocr_task_cell_count)`로 집계합니다. 절대 fallback cell count는 workload 참고값으로만 남기며 단독 degraded 판정에 사용하지 않습니다.

### Text filter runtime health

| 항목 | 내용 |
| --- | --- |
| 대상 | ML 기반 텍스트 필터링 |
| 목적 | 모델 로드, ML 판정 경로, fallback/override 이상을 추적 |
| SLI | `ML filter failed rate`, `ML filter fallback rate`, `model unavailable count`, `text filter readiness` |
| 수집 위치 | `text_filter_ml_runtime`, `/ready.components.text_filter` |
| 목표 | `TBD after baseline` |
| 제외 | false positive/false negative, shadow detector coverage |

오탐/미탐은 1차 운영 SLO가 아니라 릴리즈 품질 게이트에서 다룹니다.

## 1차 SLO에서 제외하는 지표

| 지표 | 제외 이유 | 대체 관리 위치 |
| --- | --- | --- |
| `source_url_pass_rate` | 운영 장애보다 RAG 답변 품질 회귀에 가까움 | 챗봇 품질 게이트 |
| `hallucination_proxy_rate` | 실시간 장애 판단보다 배포 전 검증에 적합 | 챗봇 품질 게이트 |
| `top3_url_accuracy` | 검색 품질 회귀 지표 | 챗봇 품질 게이트 |
| `grid_detection_success_rate` | OCR 품질 회귀 지표 | OCR 품질 게이트 |
| `average_ocr_confidence` | 운영 latency/availability와 다른 품질 지표 | OCR 품질 게이트 |
| `false_positive_count` | 텍스트 필터 품질 회귀 지표 | 텍스트 필터 품질 게이트 |
| `false_negative_count` | 텍스트 필터 품질 회귀 지표 | 텍스트 필터 품질 게이트 |
| `shadow_*` metrics | 운영 판정에 아직 연결하지 않은 shadow 지표 | 텍스트 필터 품질 게이트 |

## SLO 확정 전 필요한 baseline

| 항목 | 필요한 기준선 | 이유 |
| --- | --- | --- |
| API request p95 latency | 서비스/path별 운영 로그 | 챗봇, OCR, 필터링 처리 시간이 서로 다름 |
| Chatbot LLM latency/fallback | 실제 Ollama 운영 조건 | 모델 응답 시간과 빈 응답 빈도는 환경 의존적 |
| Retrieval latency | 현재 검색 아티팩트와 BM25 tier | `pickle`과 fallback tier 성능 차이가 큼 |
| OCR p95 total latency | OCI 또는 OCI-constrained baseline | 로컬 Mac 측정값과 운영 성능 차이가 큼 |
| OCR fallback/queue pressure | 실제 업로드 이미지와 queue 사용량 | fixture만으로 운영 압박을 판단하기 어려움 |
| Text filter latency/fallback | 실제 요청 크기와 문장 수 분포 | 문장 수, field 수에 따라 처리 시간이 달라짐 |

## 다음 단계

다음 단계에서는 RAG, OCR, 텍스트 필터링의 AI 품질 지표를 운영 SLO와 분리해 `docs/OPERATIONS_QUALITY_GATES.md`의 릴리즈 품질 게이트로 정리합니다. 품질 게이트는 배포 전 회귀 리포트와 리뷰 판단에 사용하고, 1차 운영 장애 알림에는 바로 연결하지 않습니다.
