# OCI AI 서비스 성능 기준선 실행 가이드

이 문서는 OCR, RAG 챗봇, 텍스트 필터의 현재 성능과 품질을 같은 형식으로 기록하기 위한 1단계 실행 가이드입니다.

운영 동작이나 임계값을 바꾸지 않습니다. 측정 결과는 `tests/reports/<service>/` 아래 로컬 JSON으로 저장하며, 실제 OCI에서 얻은 원본 로그와 리포트에는 운영 정보가 포함될 수 있으므로 그대로 커밋하지 않습니다.

## 완료 상태 구분

| 항목 | 저장소 준비 | 로컬 대체 측정 | 실제 OCI 측정 |
| --- | --- | --- | --- |
| OCI CPU/OCPU/RAM/배치 구조 | 수집기 제공 | CPU/RAM smoke test 가능 | 서버에서 `--ocpus`와 topology 지정 필요 |
| 프로세스 CPU/peak RSS | PID sampling 제공 | 임의 PID로 검증 가능 | main-api/chatbot-api/Ollama PID 지정 필요 |
| 요청 p50/p95/max, error/fallback | runtime log 집계 제공 | 샘플 로그로 검증 가능 | 동일 부하 구간 로그 필요 |
| 동시 요청 수, OCR queue | request log/readiness 집계 제공 | 서비스 실행 시 검증 가능 | 부하 구간에서 `/ready` sampling 필요 |
| OCR 성능·workload | 기존 리포트 확장 | `oci-constrained` 실행 가능 | 같은 케이스를 OCI에서 재실행 필요 |
| RAG 검색 품질·검색 latency | 기존 리포트 확장 | 로컬 아티팩트로 실행 가능 | OCI 아티팩트에서 재실행 필요 |
| 챗봇 LLM/전체 요청 latency | runtime log 집계 제공 | Ollama 연결 시 가능 | 실제 Ollama 부하 구간 필요 |
| 텍스트 필터 load/inference/품질 | 전용 리포트 제공 | 로컬 모델로 실행 가능 | OCI 컨테이너/프로세스에서 재실행 필요 |

“현재 운영 성능을 숫자로 설명할 수 있는 상태”는 마지막 열까지 채워졌을 때 달성한 것으로 봅니다. 로컬 또는 `oci-constrained` 결과는 운영값의 대체물이 아니라 비교용 사전 점검값입니다.

## 1. 서버와 런타임 구간 수집

OCI 콘솔에서 shape의 OCPU 수를 확인하고, main API와 챗봇 및 Ollama 프로세스 PID를 확인합니다. 실제 부하를 보내는 동안 아래 수집기를 별도 셸에서 실행합니다.

```bash
python scripts/collect_oci_ai_runtime_baseline.py \
  --duration 300 \
  --interval 1 \
  --ocpus 2 \
  --main-chatbot-topology same \
  --ollama-topology same \
  --process main-api=MAIN_API_PID \
  --process chatbot-api=CHATBOT_API_PID \
  --process ollama=OLLAMA_PID \
  --readiness main-api=http://127.0.0.1:8888/ready \
  --readiness chatbot-api=http://127.0.0.1:8010/ready \
  --runtime-log main-api=/path/to/measurement-main.log \
  --runtime-log chatbot-api=/path/to/measurement-chatbot.log
```

`--runtime-log`에는 같은 300초 측정 구간만 포함한 로그 파일을 전달합니다. 전체 운영 로그를 넣으면 측정 구간과 무관한 요청이 섞입니다. 수집 결과는 기본적으로 `tests/reports/operations/oci_ai_runtime_baseline.json`에 기록됩니다.

주요 결과 위치는 다음과 같습니다.

- `host`: CPU architecture/model, logical CPU, OCI OCPU, RAM
- `topology`: main/chatbot 및 Ollama 동일 서버 여부
- `processes`: 프로세스별 CPU p50/p95/max와 peak RSS
- `readiness`: readiness 성공률, OCR queue p50/p95/max 및 최대 사용률
- `logs.*.request_metrics`: path별 요청 p50/p95/max와 error rate
- `logs.*.runtime_events`: retrieval/LLM/OCR/filter latency와 failed/fallback rate
- `logs.*.runtime_events.chatbot_retrieval_runtime.bm25_fallback_tiers`: BM25 tier 분포
- `logs.*.max_concurrent_requests`: 로그 구간 최대 동시 요청 수

## 2. OCR 기준선

```bash
conda run -n dl_study python tests/regression/timetable/measure_timetable_ocr_baseline.py \
  --profile oci-constrained \
  --repeat 3 \
  --warmup 1
```

실제 OCI에서도 같은 cases, repeat, warmup을 사용합니다. 리포트에는 전체/격자/OCR p50·p95·max, OCR 대상 셀, 빈 셀 스킵, fallback 셀, 추출 강의 수, process CPU, peak RSS가 기록됩니다.

## 3. RAG 챗봇 기준선

검색 품질과 검색/답변 조립 파이프라인은 다음 명령으로 측정합니다.

```bash
conda run -n dl_study python tests/regression/chatbot/evaluate_rag_retrieval.py
```

이 리포트의 `answer_latency_ms`는 `build_answer()` 처리시간이며 Ollama LLM latency가 아닙니다. 실제 LLM 및 전체 챗봇 요청 p50/p95/max, fallback 비율은 1번의 `chatbot_llm_runtime`과 `chatbot_request_summary` 로그 집계를 사용합니다. 품질 리포트에서는 top1/top3 URL 정확도를 확인합니다.

## 4. 텍스트 필터 기준선

```bash
conda run -n dl_study python tests/regression/text_filtering/measure_text_filter_baseline.py \
  --repeat 5 \
  --warmup 1
```

리포트에는 모델 최초 로딩시간, 문장당 평균 추론시간, 요청 p50/p95/max, process CPU, peak RSS, false positive/negative가 기록됩니다. 모델 판정 로직이나 runtime `has_profanity` 계약은 변경하지 않습니다.

## 5. 비교 시 고정할 조건

- 동일한 commit과 모델/검색 아티팩트
- 동일한 cases 파일과 repeat/warmup
- 동일한 OCI shape, OCPU, RAM
- 동일한 OCR worker 설정
- 동일한 Ollama 모델과 배치 구조
- 동일한 동시 요청 수와 부하 시간
- cache cold/warm 여부를 별도 기록

한 조건이라도 다르면 절대값을 직접 전후 비교하지 말고 별도 baseline으로 보관합니다.

## 완료 체크리스트

- [ ] OCI CPU 종류, OCPU, RAM이 채워졌다.
- [ ] main API, chatbot API, Ollama 배치 구조가 채워졌다.
- [ ] 서비스/path별 요청 p50/p95/max가 채워졌다.
- [ ] 프로세스별 CPU와 peak RSS가 채워졌다.
- [ ] failed/fallback 비율과 BM25 tier가 채워졌다.
- [ ] 최대 동시 요청 수와 OCR queue 사용량이 채워졌다.
- [ ] OCR 리포트를 실제 OCI에서 저장했다.
- [ ] RAG 품질 리포트와 실제 LLM runtime 지표를 저장했다.
- [ ] 텍스트 필터 성능·품질 리포트를 실제 OCI에서 저장했다.

