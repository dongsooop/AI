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

## 현재 운영 프로필

사용자 확인을 바탕으로 다음 프로필을 사용합니다. 정확한 OCI 콘솔 shape 이름은 아직 확인 전이므로 `VM.Standard.A1.Flex`는 추정값으로 표시합니다.

| 항목 | 값 |
| --- | --- |
| profile | `oci-a1-flex-shared` |
| architecture | ARM64, Ampere 추정 |
| OCPU / RAM | 8 OCPU / 24GB |
| 인스턴스 | 1대 |
| 컨테이너 | main API, chatbot API, Ollama |
| Ollama 모델 | `gpt-oss:20b` |
| 시간표 queue worker | 2 |
| OCR thread worker 비교값 | 2, 8 |
| Python | 3.11.x |

세 컨테이너는 분리되어 있지만 CPU와 RAM을 공유합니다. 따라서 단독 서비스 결과와 chatbot+OCR+text-filter 혼합 부하 결과를 모두 기록합니다.

## 1. 서버와 런타임 구간 수집

OCI 콘솔에서 shape의 OCPU 수를 확인하고, main API와 챗봇 및 Ollama 프로세스 PID를 확인합니다. 실제 부하를 보내는 동안 아래 수집기를 별도 셸에서 실행합니다.

```bash
python scripts/collect_oci_ai_runtime_baseline.py \
  --profile oci-a1-flex-shared \
  --duration 300 \
  --interval 1 \
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

`oci-a1-flex-shared` 프로필은 8 OCPU, 24GB RAM, 동일 인스턴스, Docker, `gpt-oss:20b`, queue worker 2를 기대값으로 기록합니다. `host.ram_mb`는 수집기가 관찰한 호스트 값이고 `deployment_profile.expected_ram_gb`는 OCI 구성값이므로 구분해서 봅니다.

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

## 5. 단독·혼합 HTTP 부하 실행

부하 실행기는 `tests/regression/operations/run_ai_service_load.py`입니다. 기본 결과에는 요청 본문, JWT, base URL을 저장하지 않습니다.

인증이 필요한 chatbot은 토큰을 환경변수로 전달합니다. 토큰 값을 명령행에 직접 붙이지 말고 운영 셸의 안전한 방식으로 `AI_BASELINE_BEARER_TOKEN`을 미리 설정합니다.

```bash
python tests/regression/operations/run_ai_service_load.py \
  --scenario chatbot \
  --concurrency 1 \
  --requests-per-service 10 \
  --chatbot-query-file tests/regression/operations/chatbot_load_queries.json \
  --cache-state cold
```

같은 chatbot 질의를 반복하면 서비스 캐시 성능을 측정하게 됩니다. Ollama와 전체 chatbot 처리시간을 측정할 때는 `chatbot_load_queries.json`처럼 서로 다른 비민감 질의 세트를 사용합니다. 리포트에는 질의 본문을 저장하지 않고 사용한 variant 수만 기록합니다. 의도적으로 cache hit 성능을 측정할 때만 단일 `--chatbot-query`를 반복하고 `--cache-state warm`으로 표시합니다.

텍스트 필터 단독 부하는 인증이 필요 없는 `/text_filter_single`을 사용하며 학습 샘플 로그를 변경하지 않습니다.

```bash
python tests/regression/operations/run_ai_service_load.py \
  --scenario text-filter \
  --concurrency 4 \
  --requests-per-service 20
```

### OCR API 실행 주의

`/timetable_analysis`는 OCR 결과를 실제 Spring 서버로 전송할 수 있습니다. 이 때문에 OCR과 mixed 시나리오는 `--allow-ocr-api-side-effects`를 명시하지 않으면 실행되지 않습니다.

또한 서버는 HTTP 202 응답 이후에도 OCR 작업이 끝날 때까지 같은 JWT subject의 추가 시간표 작업을 막습니다. 작업 완료를 조회하는 API가 없으므로 OCR과 mixed 시나리오는 duration 방식 대신 고정 요청 수만 지원하며, 요청마다 서로 다른 subject의 토큰이 필요합니다. `--requests-per-service 4`라면 토큰도 4개가 필요합니다. 토큰 파일은 저장소 밖의 임시 경로에 두고 한 줄에 하나씩 기록합니다.

```bash
python tests/regression/operations/run_ai_service_load.py \
  --scenario ocr \
  --concurrency 2 \
  --requests-per-service 4 \
  --token-file /private/tmp/oci-baseline-tokens.txt \
  --ocr-image data/image/3.jpeg \
  --allow-ocr-api-side-effects
```

혼합 부하의 `--concurrency`는 선택된 각 서비스의 worker 수입니다. 예를 들어 `mixed --concurrency 2`는 chatbot 2, OCR 2, text-filter 2를 동시에 실행합니다.

```bash
python tests/regression/operations/run_ai_service_load.py \
  --scenario mixed \
  --concurrency 2 \
  --requests-per-service 4 \
  --token-file /private/tmp/oci-baseline-tokens.txt \
  --ocr-image data/image/3.jpeg \
  --allow-ocr-api-side-effects \
  --cache-state warm
```

안전 중단 조건은 readiness 연속 실패, 서비스별 최근 요청 error rate 초과, OCR queue 사용률 초과, Linux 호스트 메모리 사용률 95% 이상 지속입니다. 초기 순서는 concurrency 1 → 2 → 4 → 8이며, 앞 단계가 안정적일 때만 다음 단계로 진행합니다. 이 값들은 부하 테스트 보호용이며 운영 SLO가 아닙니다.

## 6. 공개 안전 요약 생성

원본 리포트를 모은 뒤 다음 명령으로 JSON과 Markdown 요약을 생성합니다.

```bash
python scripts/summarize_oci_ai_baseline.py \
  --load-report tests/reports/operations/ai_service_load_report.json
```

기본 출력:

- `tests/reports/operations/oci_ai_baseline_summary.json`
- `tests/reports/operations/oci_ai_baseline_summary.md`

요약에는 prompt, token, base URL, PID, 원본 로그 줄을 포함하지 않습니다. RAG 리포트가 임베딩 모델 또는 아티팩트 로딩 실패로 생성된 경우 0%를 기준선으로 사용하지 않고 `environment_invalid`로 표시합니다.

## 7. 비교 시 고정할 조건

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
