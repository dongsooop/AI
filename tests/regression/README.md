# Regression Tests

회귀 테스트는 서비스 영역별로 나눠 둡니다.

- `chatbot/`: OSS 챗봇, RAG 검색 품질, query-index 호환성, tool routing 회귀 테스트
- `timetable/`: 시간표 OCR, 이미지 품질 진단, 셀/그리드 분석 회귀 테스트
- `text_filtering/`: 텍스트 필터링 품질, 정상 문장 오탐/비속어 미탐 회귀 테스트

대표 실행 명령:

```bash
python tests/regression/chatbot/evaluate_rag_retrieval.py --validate-only
python tests/regression/chatbot/evaluate_rag_retrieval.py
python tests/regression/chatbot/check_chatbot_tool_routing.py
python tests/regression/timetable/check_timetable_ocr_diagnostics.py
python tests/regression/timetable/measure_timetable_ocr_baseline.py
python tests/regression/timetable/analyze_timetable_empty_cell_skip.py
python tests/regression/text_filtering/check_text_filter_quality_report.py
python tests/regression/text_filtering/check_text_filter_normalization.py
python tests/regression/text_filtering/check_text_filter_word_matcher.py
python tests/regression/text_filtering/check_text_filter_match_details_response.py
```

## Evaluation Metrics Policy

회귀 스크립트는 로컬과 CI에서 평가 지표를 JSON summary로 출력하고, 상세 리포트를 기능별 디렉터리에 저장합니다. 이 기반 단계에서는 새 지표를 실패 조건으로 강제하지 않고, 기존 테스트의 통과/실패 판정만 유지합니다.

리포트 기본 저장 위치:

- `tests/reports/chatbot/`: 챗봇, RAG 검색, query-index 리포트
- `tests/reports/timetable/`: 시간표 OCR, 이미지 품질 진단 리포트
- `tests/reports/text_filtering/`: 텍스트 필터링 리포트

`tests/reports/**/*.json` 파일은 실행 산출물이므로 커밋하지 않습니다. 공유가 필요한 기준 리포트나 샘플은 별도 문서로 의도를 설명한 뒤 추가합니다.

### Chatbot RAG Product Metrics

`tests/regression/chatbot/evaluate_rag_retrieval.py`는 챗봇 RAG 품질을 제품 관점 지표로 함께 출력합니다.

- `recall_at_1`: 검색 결과 1위 URL이 케이스의 `expected_url_contains`와 일치한 비율
- `recall_at_3`: 검색 결과 상위 3개 URL 중 하나가 `expected_url_contains`와 일치한 비율
- `source_url_pass_rate`: 출처 URL이 필요한 search 케이스에서 답변이 기대 출처 URL을 포함한 비율
- `unofficial_url_hallucination_rate`: 답변에 공식 학교 도메인 또는 검색 근거에 없는 외부 URL이 포함된 비율
- `refusal_rate`: 평가 답변이 거절/fallback 문구로 판정된 비율
- `refusal_expectation_pass_rate`: `expects_refusal` 케이스 설정과 실제 거절 여부가 일치한 비율

카테고리별 결과는 `summary.by_category`에 `passed`, `failed`, `pass_rate`와 함께 같은 제품 지표 일부를 저장합니다. 현재 RAG 평가는 `LLM.sub_model.query_index.build_answer()`와 `schedule_search()`를 직접 호출하므로, API/LLM 라우팅 전체의 거절 정책 평가는 별도 API 회귀 테스트에서 다룹니다.

### Timetable OCR Diagnostics Metrics

`tests/regression/timetable/check_timetable_ocr_diagnostics.py`는 `tests/regression/timetable/timetable_ocr_diagnostic_cases.json`의 케이스를 읽어 시간표 OCR 진단 리포트를 생성합니다.

- `grid_detection_success_rate`: 케이스별 최소 격자 라인 기준을 통과한 비율
- `extracted_course_count`: 추출된 시간표 항목 수
- `average_empty_cell_ratio`: OCR 대상 셀 중 텍스트가 없는 셀 비율 평균
- `average_ocr_confidence`: 진단 OCR confidence 평균
- `fallback_cell_count`: fallback OCR 경로가 선택된 셀 수
- `skipped_empty_cell_count`: OCR 전에 빈 셀 후보로 건너뛴 셀 수

`cv2`, `pytesseract`, Tesseract 런타임이 없는 환경에서는 실패 대신 `status: "skipped"` 리포트를 기본 경로에 기록합니다.

### Timetable OCR Performance Baseline

`tests/regression/timetable/measure_timetable_ocr_baseline.py`는 `tests/regression/timetable/timetable_ocr_performance_cases.json`의 실제 시간표 이미지 샘플을 읽어 현재 OCR 런타임 기준선을 기록합니다. 운영 경로와 가깝게 측정하기 위해 `extract_schedule_runtime_report()`를 호출하며, OCR 인식 로직은 변경하지 않습니다.

- `average_total_duration_ms`, `max_total_duration_ms`, `p95_total_duration_ms`: 전체 처리 시간 기준선
- `average_grid_detection_duration_ms`, `max_grid_detection_duration_ms`: 격자 검출 시간 기준선
- `average_ocr_duration_ms`, `max_ocr_duration_ms`: Tesseract OCR 시간 기준선
- `total_extracted_schedule_count`: 추출된 시간표 항목 수 합계
- `ocr_task_cell_count`, `skipped_empty_cell_count`: 실제 Tesseract 대상 셀 수와 빈 셀 스킵 수
- `failure_count`, `failure_reasons`: 실패 사유 관측치

기본 리포트는 `tests/reports/timetable/timetable_ocr_performance_baseline.json`에 저장됩니다. `data/`가 로컬/운영 샘플 영역이라 이미지 파일이 없는 케이스는 전체 실패 대신 개별 `skipped` 케이스로 기록합니다. OCI에서는 부하를 낮추기 위해 기본 `--repeat 1`을 권장하고, 안정적인 비교가 필요할 때만 반복 횟수를 늘립니다.

OCI 운영 서버와 비교할 때는 먼저 서버에서 기본 운영 프로필을 그대로 측정합니다.

```bash
python tests/regression/timetable/measure_timetable_ocr_baseline.py --profile production --repeat 1
```

로컬에서 OCI 제약에 가까운 보수적 기준선을 보고 싶을 때는 셀 OCR worker 수를 줄인 프로필을 사용합니다. 이 프로필은 현재 프로세스 안에서만 `image_analysis.ocr_engine.OCR_THREAD_WORKERS`를 2로 낮춰 측정하고, 운영 코드는 수정하지 않습니다.

```bash
python tests/regression/timetable/measure_timetable_ocr_baseline.py --profile oci-constrained --repeat 1
```

정확한 OCI 기준선은 실제 OCI 서버에서 측정한 `production` 프로필 결과를 우선합니다. `oci-constrained`는 로컬에서 성능 회귀 위험을 보수적으로 보는 보조 지표입니다.

### Timetable OCR Empty-Cell Skip Analysis

`tests/regression/timetable/analyze_timetable_empty_cell_skip.py`는 빈 셀 OCR 스킵을 운영에 적용하기 전에 셀 foreground density threshold와 text-presence detector 후보를 dry-run으로 분석합니다. 현재 격자/ROI/OCR 판정 함수를 그대로 사용하되 런타임 동작은 바꾸지 않고, threshold별로 OCR 전에 스킵했을 셀 수와 이미 accepted로 판정되는 셀이 스킵 후보에 포함되는지 기록합니다.

- `would_skip_cell_count`: threshold 적용 시 OCR 전에 건너뛰었을 셀 수
- `would_skip_accepted_cell_count`: 현재 OCR에서 accepted인 셀이 스킵 후보에 들어간 수
- `would_skip_text_cell_count`: 현재 OCR에서 비어 있지 않은 셀이 스킵 후보에 들어간 수
- `best_safe_threshold_no_accepted_loss`: 샘플 기준 accepted 셀 손실이 없는 가장 높은 threshold
- `best_safe_threshold_no_text_loss`: 샘플 기준 OCR 텍스트 셀 손실이 없는 가장 높은 threshold
- `text_presence_estimated_ocr_task_count`: OpenCV text-presence detector 적용 시 예상 OCR 대상 셀 수

기본 리포트는 `tests/reports/timetable/timetable_ocr_empty_cell_skip_analysis.json`에 저장됩니다. 이 리포트에서 `would_skip_accepted_cell_count`가 0인 보수적 threshold를 확인한 뒤에만 운영 OCR 스킵 로직을 적용합니다.

### Text Filtering Quality Metrics

`tests/regression/text_filtering/check_text_filter_quality_report.py`는 기본적으로 `tests/regression/text_filtering/cases/*.json`의 케이스를 읽어 기존 텍스트 필터 판정 로직을 평가합니다. `--cases`로 기존 단일 JSON 파일도 지정할 수 있습니다.
현재 케이스 파일의 1차 목적은 단어 단위 shadow 탐지 작업 전에 운영 True/False 및 `has_profanity` 판정 기준선을 고정하는 것입니다.
`tests/regression/text_filtering/check_text_filter_normalization.py`는 운영 판정과 연결하지 않은 정규화 후보 생성 helper만 검증합니다.
`tests/regression/text_filtering/check_text_filter_word_matcher.py`는 정규화 후보 기반 단어 단위 match 근거 생성을 검증하며, 운영 API 판정에는 연결하지 않습니다.
`tests/regression/text_filtering/check_text_filter_match_details_response.py`는 필드별 응답에 `matches`가 추가되어도 기존 `has_profanity`와 `results` 계약이 유지되는지 검증합니다.
`TEXT_FILTER_STRONG_RULE_OVERRIDE=1`이면 strong rule candidate match를 재학습 전 임시 보호막으로 `has_profanity` 판정에 반영합니다. 기본값은 `0`입니다.
품질 리포트의 pass/fail은 `analyze_text_labels()`의 기존 모델 라벨 기준입니다. Strong rule override 효과는 shadow 지표와 `check_text_filter_match_details_response.py`에서 확인합니다.

- `false_positive_count`: 정상 문장을 비속어로 판정한 케이스 수
- `false_negative_count`: 비속어 문장을 정상으로 판정한 케이스 수
- `model_false_positive_count`, `model_false_negative_count`: 기존 모델 판정 기준 오탐/미탐 케이스 수
- `pass_rate`: 전체 golden case 기대값과 실제 결과가 일치한 비율
- `ml_filter_pass_rate`: 현재 ML 기반 판정 경로 기준 통과율
- `rule_endpoint_pass_rate`: 현재 `/text_filter_rule` API 계약의 공유 판정 경로 기준 통과율
- `shadow_match_count`: 운영 판정에 연결하지 않은 단어 단위 detector match 수
- `shadow_matched_case_count`: shadow detector가 하나 이상의 match 근거를 만든 케이스 수
- `shadow_detected_false_negative_count`: 기존 모델이 놓친 비속어 케이스 중 shadow detector가 match 근거를 만든 수
- `shadow_detected_false_negative_rate`: 기존 모델 미탐 중 shadow detector가 보완 근거를 만든 비율
- `shadow_false_positive_candidate_count`: 정상 기대 케이스 중 shadow detector가 match 근거를 만든 오탐 후보 수
- `shadow_false_positive_candidate_rate`: 정상 기대 케이스 중 shadow 오탐 후보 비율
- `shadow_strong_rule_candidate_match_count`: feature flag 기반 운영 반영 후보로 분리된 강한 규칙 match 수
- `shadow_strong_rule_detected_false_negative_count`: 기존 모델이 놓친 케이스 중 강한 규칙 후보가 match 근거를 만든 수
- `shadow_strong_rule_false_positive_candidate_count`: 정상 기대 케이스 중 강한 규칙 후보가 match 근거를 만든 수
- `by_category`, `shadow_by_category`, `by_pattern_id`: 카테고리와 pattern id별로 모델 미탐, shadow 보완, 오탐 후보를 나눠 보는 집계

리포트의 `examples` 섹션은 모델 기준 false negative/false positive 샘플과, 그중 shadow detector가 match 근거를 만든 후보를 최대 5개씩 보여줍니다. 샘플 개수는 `--example-limit`로 조정할 수 있습니다. 각 샘플은 `shadow_patterns`와 `strong_rule_candidate_patterns`를 함께 기록해서 default-off strong rule 후보 판단에 활용할 수 있게 합니다.

이 스크립트는 `analyze_text_labels()`만 호출하므로 테스트 문장을 `data/bad_text_sample.txt`에 append하지 않습니다. 모델 파일 또는 의존성이 없는 환경에서는 실패 대신 `status: "skipped"` 리포트를 기본 경로에 기록합니다.
기본 실행은 품질 리포트를 기록하고 종료 코드는 0으로 유지합니다. 품질 실패를 게이트로 쓰려면 `--strict`를 함께 사용합니다.

## Common JSON Summary

각 리포트는 가능한 한 아래 summary 형식을 따릅니다. 상세 지표는 suite마다 다를 수 있으므로 `metrics` 안에 둡니다.

```json
{
  "schema_version": 1,
  "suite": "timetable_ocr_diagnostics",
  "service": "timetable",
  "status": "passed",
  "total": 1,
  "passed": 1,
  "failed": 0,
  "skipped": 0,
  "metrics": {
    "grid_detection_success_rate": 1.0,
    "extracted_course_count": 4,
    "average_empty_cell_ratio": 0.0,
    "average_ocr_confidence": 87.33,
    "fallback_cell_count": 78
  },
  "errors": []
}
```

CI에서는 summary 한 줄과 `report=<path>` 출력을 수집합니다. 리포트 파일은 기본 경로를 쓰되, 필요한 경우 각 스크립트의 `--out` 옵션으로 저장 위치를 바꿀 수 있습니다.
