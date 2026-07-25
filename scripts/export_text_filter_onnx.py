#!/usr/bin/env python3
"""Export the local ELECTRA text filter to ONNX FP32 and dynamic INT8."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT_DIR / "model" / "my_electra_finetuned"
DEFAULT_OUT_DIR = ROOT_DIR / "model" / "generated" / "text_filter_onnx"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export text-filter ELECTRA ONNX shadow artifacts")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    try:
        import onnx
        import torch
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from transformers import ElectraForSequenceClassification, ElectraTokenizer
    except ModuleNotFoundError as exc:
        print(
            f"[ERROR] missing dependency: {exc.name}; "
            "install requirements-onnx-shadow.txt",
            file=sys.stderr,
        )
        return 2

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    fp32_path = out_dir / "text_filter_fp32.onnx"
    int8_path = out_dir / "text_filter_int8.onnx"
    manifest_path = out_dir / "manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    tokenizer = ElectraTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = ElectraForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.to("cpu")
    model.eval()

    class LogitsWrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_ids, attention_mask):
            return self.wrapped(input_ids=input_ids, attention_mask=attention_mask).logits

    encoded = tokenizer(
        "텍스트 필터 ONNX export",
        add_special_tokens=True,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    torch.onnx.export(
        LogitsWrapper(model),
        (encoded["input_ids"], encoded["attention_mask"]),
        fp32_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(fp32_path))

    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
    )
    onnx.checker.check_model(onnx.load(int8_path))

    manifest = {
        "schema_version": 1,
        "suite": "text_filter_onnx_export",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "opset": args.opset,
        "quantization": {
            "method": "dynamic",
            "weight_type": "QInt8",
            "per_channel": False,
        },
        "source_model_dir": str(model_dir.relative_to(ROOT_DIR)) if model_dir.is_relative_to(ROOT_DIR) else None,
        "artifacts": {
            "fp32": {"filename": fp32_path.name, "size_bytes": fp32_path.stat().st_size},
            "int8": {"filename": int8_path.name, "size_bytes": int8_path.stat().st_size},
        },
        "export_duration_ms": round((time.perf_counter() - started) * 1000, 2),
        "generated_artifacts_tracked": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] text-filter ONNX shadow export")
    print(json.dumps(manifest, ensure_ascii=False))
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
