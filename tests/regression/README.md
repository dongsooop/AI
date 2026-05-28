# Regression Tests

회귀 테스트는 서비스 영역별로 나눠 둡니다.

- `chatbot/`: OSS 챗봇, RAG 검색 품질, query-index 호환성, tool routing 회귀 테스트
- `timetable/`: 시간표 OCR, 이미지 품질 진단, 셀/그리드 분석 회귀 테스트

대표 실행 명령:

```bash
python tests/regression/chatbot/evaluate_rag_retrieval.py --validate-only
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
