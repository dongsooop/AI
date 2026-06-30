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
python tests/regression/text_filtering/check_text_filter_quality_report.py
python tests/regression/text_filtering/check_text_filter_normalization.py
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

`cv2`, `pytesseract`, Tesseract 런타임이 없는 환경에서는 실패 대신 `status: "skipped"` 리포트를 기본 경로에 기록합니다.

### Text Filtering Quality Metrics

`tests/regression/text_filtering/check_text_filter_quality_report.py`는 `tests/regression/text_filtering/text_filter_quality_cases.json`의 케이스를 읽어 기존 텍스트 필터 판정 로직을 평가합니다.
현재 케이스 파일의 1차 목적은 단어 단위 shadow 탐지 작업 전에 운영 True/False 및 `has_profanity` 판정 기준선을 고정하는 것입니다.
`tests/regression/text_filtering/check_text_filter_normalization.py`는 운영 판정과 연결하지 않은 정규화 후보 생성 helper만 검증합니다.

- `false_positive_count`: 정상 문장을 비속어로 판정한 케이스 수
- `false_negative_count`: 비속어 문장을 정상으로 판정한 케이스 수
- `pass_rate`: 전체 golden case 기대값과 실제 결과가 일치한 비율
- `ml_filter_pass_rate`: 현재 ML 기반 판정 경로 기준 통과율
- `rule_endpoint_pass_rate`: 현재 `/text_filter_rule` API 계약의 공유 판정 경로 기준 통과율

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
