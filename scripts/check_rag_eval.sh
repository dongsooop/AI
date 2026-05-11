#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REPORT_PATH="${RAG_EVAL_REPORT_PATH:-/tmp/rag_eval_report.json}"
CONDA_ENV="${RAG_EVAL_CONDA_ENV:-dl_study}"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
conda run -n "$CONDA_ENV" python tests/regression/evaluate_rag_retrieval.py \
  --out "$REPORT_PATH" \
  --fail-on-fail

conda run -n "$CONDA_ENV" python tests/regression/check_query_index_compat.py
conda run -n "$CONDA_ENV" python tests/regression/check_query_index_metadata_quality.py

echo "rag_eval_report=$REPORT_PATH"
