# 운영 릴리즈 품질 게이트

이 문서는 AI 품질 지표를 운영 SLO와 분리해 배포 전 릴리즈 품질 게이트로 관리하는 기준을 정리합니다.

`docs/operations/OPERATIONS_SLO_CANDIDATES.md`의 1차 운영 SLO는 availability, latency, error, fallback 중심으로 둡니다. RAG 답변 품질, OCR 인식 품질, 텍스트 필터 오탐/미탐은 사용자 경험에 중요하지만 실시간 장애 알림보다 회귀 리포트와 리뷰 판단에 더 적합하므로 이 문서에서 별도로 다룹니다.

## 적용 원칙

| 원칙 | 내용 |
| --- | --- |
| 운영 SLO와 분리 | API가 정상 응답하더라도 AI 품질은 후퇴할 수 있으므로 별도 게이트로 봅니다. |
| 리포트 기반 판단 | 품질 게이트는 `tests/regression/*` 스크립트의 JSON summary와 report를 기준으로 판단합니다. |
| 기본은 관측 | 새 지표를 추가하더라도 곧바로 배포 차단 조건으로 만들지 않고 baseline을 먼저 봅니다. |
| 상세 리포트 로컬 유지 | raw report는 `tests/reports/<feature>/` 아래 실행 산출물로 두고, 공개가 필요하면 sanitized summary만 사용합니다. |
| 서비스 경계 유지 | chatbot, timetable, text_filtering 품질 게이트를 한 기준으로 섞지 않습니다. |

## 품질 게이트 요약

| 서비스 | 게이트 대상 | 대표 리포트 | 기본 처리 |
| --- | --- | --- | --- |
| chatbot | RAG 검색/답변 품질, 출처 URL, hallucination proxy | `tests/reports/chatbot/rag_eval_report.json` | 배포 전 회귀 확인 |
| timetable | 격자 검출, OCR confidence, 추출 항목, OCR 성능 baseline | `tests/reports/timetable/*.json` | OCR 변경 전후 비교 |
| text_filtering | 오탐/미탐, 기존 모델 판정, shadow detector coverage | `tests/reports/text_filtering/text_filter_quality_report.json` | 운영 판정 변경 전 검토 |

## 챗봇 품질 게이트

| 지표 | 출처 | 의미 | 게이트 판단 |
| --- | --- | --- | --- |
| `pass_rate` | RAG eval summary | 전체 케이스 기대 조건 통과율 | 기존 baseline 대비 후퇴 여부 확인 |
| `top1_url_accuracy` | RAG eval summary | 검색 1위 URL 정합성 | 검색 ranking 회귀 판단 |
| `top3_url_accuracy` | RAG eval summary | 상위 3개 URL 정합성 | 주요 recall 회귀 판단 |
| `source_url_pass_rate` | RAG eval summary | 답변 출처 URL 포함 비율 | 출처 요구 케이스에서 중요 |
| `hallucination_proxy_rate` | RAG eval summary | 근거 없는 외부 URL 포함 proxy | 증가 시 배포 전 원인 분석 |
| `answer_keyword_pass_rate` | RAG eval summary | 답변 핵심 키워드 포함 여부 | 답변 내용 회귀 판단 |
| `date_pass_rate` | RAG eval summary | 학사일정 날짜 정합성 | 일정 답변 회귀 판단 |
| `avg_retrieval_latency_ms`, `p95_retrieval_latency_ms` | RAG eval summary | 검색 평가 성능 | 운영 latency SLO 보조 지표 |

챗봇 품질 게이트는 `scripts/check_rag_eval.sh` 또는 `tests/regression/chatbot/evaluate_rag_retrieval.py`의 report를 기준으로 봅니다. raw report에는 query, answer, URL, 로컬 경로, error payload가 들어갈 수 있으므로 공개 artifact에는 sanitized summary만 사용합니다.

### 최종 답변 회귀 검증

`python tests/regression/chatbot/evaluate_chatbot_answers.py --strict`는
`answer_quality_cases.json`의 골든 사례를 실제 서비스 함수와 로컬 검색 인덱스로 실행합니다.
검색 결과만 평가하는 기존 RAG 회귀와 별도로, 최종 `text`·`url`과 내부 처리 경로를 확인합니다.

- 설명: 적용 대상, 학제, 연도와 정답 학점 문구를 확인하고 예상하지 않은 학점 숫자를 거부합니다.
- 출처: 기대 URL과 버튼 URL을 비교하고, `single_source` 사례는 본문에 표시된 모든 URL도 같아야 합니다. `layout` 파라미터와 fragment만 URL 비교에서 제외합니다.
- 되묻기: 필요한 조건을 묻는 문구와 처리 경로를 확인하고, 학점 단정이나 불필요한 링크를 금지합니다.
- 안내: 휴학·장학금·기숙사처럼 개괄 안내가 가능한 질문은 기대 공식 안내 URL을 확인합니다.

`run_chatbot_regression.py`도 URL 정책, 필수/금지 정규식, 허용 학점 숫자 검사를 지원합니다.
`expected_tools`는 내부 metadata에 접근 가능한 최종 서비스 평가 전용입니다.
`behavior`는 골든 사례의 기대 행동 분류이며 자동 의미 판정 모델이 아닙니다.
골든 조건의 통과율은 표본 회귀 지표이고, 전체 질문에 대한 의미 정확도나 환각률을 보장하지 않습니다.

최종 서비스 평가는 비속어 HTTP 검사, DB 로그 기록, LLM 호출을 테스트 대체물로 바꿉니다.
검색·인덱스·분류·답변 조립은 실제 코드를 사용하며, 예상하지 않은 LLM 호출은 실패로 기록합니다.
따라서 JWT/HTTP 인증, 운영 네트워크와 LLM 생성 품질은 별도의 실제 API 검증이 필요합니다.
인덱스와 모델이 없으면 평가를 성공 처리하지 않고 실행이 실패합니다.

상세 결과는 `tests/reports/chatbot/answer_quality_report.json`에 로컬 저장하고,
stdout에는 `answer`·`clarification`·`guidance`별 집계만 출력합니다.
기본 실행은 관측용이며 `--strict`에서는 한 사례라도 실패하면 종료 코드 2를 반환합니다.
수동 Full Evaluation 워크플로는 이 명시적 골든 검사를 `--strict`로 실행합니다.
Light Check는 데이터·모델 없이 검사기 테스트와 1~4단계 보호 로직의 회귀 테스트를 실행합니다.

## 시간표 OCR 품질 게이트

| 지표 | 출처 | 의미 | 게이트 판단 |
| --- | --- | --- | --- |
| `grid_detection_success_rate` | timetable diagnostics | 격자 검출 성공률 | OCR 구조 회귀 판단 |
| `extracted_course_count` | timetable diagnostics | 추출된 시간표 항목 수 | 과소 추출 회귀 판단 |
| `average_ocr_confidence` | timetable diagnostics | OCR confidence 평균 | 인식 품질 추세 확인 |
| `average_empty_cell_ratio` | timetable diagnostics | 빈 셀 비율 | 셀 분할/인식 이상 탐지 |
| `fallback_cell_count` | diagnostics/performance | fallback OCR 선택 셀 수 | 품질 및 성능 리스크 |
| `p95_total_duration_ms` | performance baseline | 전체 처리 시간 p95 | OCI 기준 성능 회귀 판단 |
| `ocr_task_cell_count`, `skipped_empty_cell_count` | performance/empty-cell analysis | OCR 부하와 빈 셀 스킵 효과 | 성능 최적화 회귀 판단 |

OCR 품질 게이트는 로컬 측정값만으로 운영 품질을 확정하지 않습니다. 성능 판단은 실제 OCI `production` 프로필 또는 `oci-constrained` 보조 기준선을 함께 봅니다. OCR 로직을 바꾸는 브랜치는 품질 지표와 성능 지표를 함께 봅니다.

## 텍스트 필터링 품질 게이트

| 지표 | 출처 | 의미 | 게이트 판단 |
| --- | --- | --- | --- |
| `pass_rate` | text filtering quality report | 전체 golden case 통과율 | 기본 회귀 판단 |
| `false_positive_count` | quality report | 정상 문장 오탐 수 | 사용자 경험 리스크 |
| `false_negative_count` | quality report | 비속어 미탐 수 | 안전성 리스크 |
| `model_false_positive_count` | quality report | 기존 모델 기준 오탐 수 | 모델 기준선 추적 |
| `model_false_negative_count` | quality report | 기존 모델 기준 미탐 수 | 모델 기준선 추적 |
| `shadow_detected_false_negative_rate` | quality report | shadow detector가 미탐을 보완한 비율 | 후속 detector rollout 후보 |
| `shadow_false_positive_candidate_rate` | quality report | shadow detector 오탐 후보 비율 | rollout 위험 평가 |
| `shadow_strong_rule_*` | quality report | 강한 규칙 후보 효과와 오탐 위험 | default-off 보호막 검토 |
| `by_category`, `by_pattern_id` | quality report | 카테고리/패턴별 영향 | 특정 패턴 회귀 분석 |

텍스트 필터링 품질 게이트는 기존 운영 `True/False`, `has_profanity` 판정을 즉시 바꾸기 위한 기준이 아닙니다. normalization, word matcher, strong rule 후보는 먼저 shadow 지표로 검토하고, 운영 반영은 별도 feature flag 또는 후속 브랜치에서 판단합니다.

## 게이트 강도

| 단계 | 의미 | 사용 시점 |
| --- | --- | --- |
| 관측 | report를 생성하고 summary를 리뷰에 첨부하지만 배포를 차단하지 않음 | 새 지표 추가 직후, baseline 수집 기간 |
| 경고 | baseline 대비 후퇴하면 원인 분석을 요구하지만 자동 차단은 하지 않음 | 지표 안정화 이후 |
| 차단 | 명시된 임계값을 넘으면 배포 전 수정 또는 승인 필요 | 충분한 baseline과 운영 합의 이후 |

현재 기본값은 관측입니다. `--strict`를 지원하는 스크립트는 품질 실패를 차단 조건으로 사용할 수 있지만, 이를 기본 CI나 배포 게이트로 올리기 전에 baseline과 false positive 위험을 먼저 확인합니다.

## 리포트 공개 정책

| 리포트 종류 | 기본 위치 | 공개 정책 |
| --- | --- | --- |
| raw RAG report | `tests/reports/chatbot/` | 로컬/CI 산출물로 유지, 공개 artifact는 sanitized summary 사용 |
| OCR diagnostics/performance report | `tests/reports/timetable/` | 실행 산출물로 유지, 샘플 이미지/환경 정보 포함 여부 확인 |
| text filtering quality report | `tests/reports/text_filtering/` | 실행 산출물로 유지, 사용자 원문 포함 여부 확인 |
| sanitized summary | 필요한 경우 별도 파일 | 공개 가능한 aggregate metrics만 포함 |

`tests/reports/**/*.json` 파일은 실행 산출물이므로 기본적으로 커밋하지 않습니다. 공개가 필요한 기준 리포트는 raw 상세 파일 대신 aggregate metrics 중심의 summary를 별도로 만듭니다.

## 리뷰 체크포인트

품질 게이트를 사용하는 브랜치 리뷰에서는 다음을 확인합니다.

- 어떤 서비스의 품질 지표가 영향을 받는지 명시합니다.
- 관련 회귀 스크립트를 실행했는지, 실행하지 못했다면 이유를 남깁니다.
- report의 `status`, `total`, `passed`, `failed`, `skipped`, 핵심 `metrics`를 요약합니다.
- 이전 baseline이 있으면 후퇴 여부를 비교합니다.
- raw report 또는 샘플에 공개하면 안 되는 query, answer, 원문, 로컬 경로, error payload가 포함되지 않았는지 확인합니다.
- 운영 SLO 위반과 품질 게이트 실패를 같은 장애로 표현하지 않습니다.

## 다음 단계

다음 단계에서는 SLO 또는 품질 게이트 위반 시 어떤 알림, 점검, 배포 판단 흐름으로 이어질지 `docs/operations/OPERATIONS_INCIDENT_WORKFLOW.md`의 incident workflow로 연결합니다.
