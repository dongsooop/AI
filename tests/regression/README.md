# Regression Tests

회귀 테스트는 서비스 영역별로 나눠 둡니다.

- `chatbot/`: OSS 챗봇, RAG 검색 품질, query-index 호환성, tool routing 회귀 테스트
- `timetable/`: 시간표 OCR, 이미지 품질 진단, 셀/그리드 분석 회귀 테스트

대표 실행 명령:

```bash
python tests/regression/chatbot/evaluate_rag_retrieval.py --validate-only
python tests/regression/chatbot/evaluate_rag_retrieval.py
python tests/regression/chatbot/check_chatbot_tool_routing.py
python tests/regression/timetable/check_timetable_ocr_diagnostics.py
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
    "average_confidence": 87.33,
    "fallback_cells": 1
  },
  "errors": []
}
```

CI에서는 summary 한 줄과 `report=<path>` 출력을 수집합니다. 리포트 파일은 기본 경로를 쓰되, 필요한 경우 각 스크립트의 `--out` 옵션으로 저장 위치를 바꿀 수 있습니다.
