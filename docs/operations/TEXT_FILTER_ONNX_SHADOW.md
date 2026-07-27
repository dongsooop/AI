# 텍스트 필터 ONNX shadow 실험

이 실험은 현재 PyTorch ELECTRA 판정을 ONNX FP32 및 dynamic INT8과 비교합니다.
운영 API, `has_profanity`, `True/False` 계약과 `text_filtering/service.py`의
실행 backend는 변경하지 않습니다.

## 설치

운영 requirements와 분리된 실험용 의존성을 사용합니다.

```bash
python -m pip install -r requirements-onnx-shadow.txt
```

## 모델 생성

```bash
python scripts/export_text_filter_onnx.py
```

생성 파일은 `model/generated/text_filter_onnx/`에 저장되며 Git에 커밋하지 않습니다.

## 비교 실행

```bash
python tests/regression/text_filtering/benchmark_text_filter_onnx_shadow.py \
  --repeat 5 \
  --warmup 1
```

PyTorch, ONNX FP32, ONNX INT8은 서로 다른 프로세스에서 실행됩니다. 각 backend의
pass rate, false positive/negative, cold start, p50/p95/max latency, peak RSS와
케이스별 판정을 `tests/reports/text_filtering/text_filter_onnx_shadow_report.json`
과 같은 이름의 Markdown 요약에 기록합니다.

## 운영 전환 조건

- ONNX FP32와 INT8의 케이스별 판정이 PyTorch와 동일
- false positive와 false negative 증가 없음
- 충분한 반복 측정에서 p95 latency 감소
- peak RSS 감소
- OCI와 동일한 ARM64 Docker 이미지에서 정상 실행

이 조건을 모두 확인하기 전에는 `TEXT_FILTER_INFERENCE_BACKEND` feature flag나
ONNX runtime 경로를 운영 서비스에 추가하지 않습니다.

## 실험 결과

Linux ARM64 컨테이너에서 backend별 독립 프로세스 측정을 3회 실행하고 중앙값을
비교했습니다. 각 케이스는 warmup 1회 후 5회 반복 측정했습니다.

| Backend | Pass rate | FP | FN | Cold start | p95 latency | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch | 0.5714 | 0 | 6 | 1330.46 ms | 130.52 ms | 662.96 MB |
| ONNX FP32 | 0.5714 | 0 | 6 | 697.00 ms | 65.45 ms | 880.48 MB |
| ONNX INT8 | 0.5000 | 0 | 7 | 571.40 ms | 21.49 ms | 326.78 MB |

FP32는 PyTorch와 케이스별 판정이 같고 p95 latency가 약 49.85% 감소했지만,
peak RSS가 약 32.81% 증가했습니다. INT8은 p95 latency가 약 83.54%,
peak RSS가 약 50.71% 감소했지만 `korean_initial_consonant_01` 케이스에서
false negative가 1건 증가했습니다.

현재 golden case는 14건으로 운영 전환을 판단하기에 충분하지 않고, 실제 OCI
인스턴스에서도 같은 결과를 확인하지 않았습니다. 정확도 회귀 또는 RSS 증가가
없는 후보가 없으므로 운영 backend는 PyTorch를 유지합니다. ONNX feature flag와
운영 추론 경로는 추가하지 않습니다.
