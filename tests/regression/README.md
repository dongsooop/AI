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
