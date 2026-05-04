#!/usr/bin/env python3
import argparse
import gzip
import importlib
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from LLM.sub_model.query_index_schema import (  # noqa: E402
    METADATA_BOOL_COLS,
    METADATA_TEXT_COLS,
    TEXT_FALLBACK_COLS,
    normalize_search_df_schema,
)


def make_legacy_search_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "title": "컴퓨터공학부",
                "url": "https://www.dongyang.ac.kr/example/computer",
                "text": "컴퓨터공학부 소개와 교육과정 안내입니다.",
                "doc_type": "Department",
            },
            {
                "title": "졸업 학점 안내",
                "url": "https://www.dongyang.ac.kr/example/grad",
                "text": "3년제 졸업이수 학점은 총 120학점입니다.",
                "doc_type": "policy",
            },
            {
                "title": "학생성공지원팀",
                "url": "https://www.dongyang.ac.kr/example/contact",
                "text": "학생성공지원팀 전화번호 02-2610-1234",
                "doc_type": "contact",
            },
        ]
    )


def check_normalized_schema(df: pd.DataFrame) -> list[str]:
    errors = []
    required = [
        "doc_type",
        "title",
        "text",
        "unit",
        "phone",
        "email",
        "url",
        *TEXT_FALLBACK_COLS,
        *METADATA_TEXT_COLS,
        *METADATA_BOOL_COLS,
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"missing_columns:{missing}")

    for c in TEXT_FALLBACK_COLS:
        if not df[c].equals(df["text"]):
            errors.append(f"text_fallback_mismatch:{c}")

    for c in METADATA_TEXT_COLS:
        if df[c].isna().any():
            errors.append(f"metadata_text_has_na:{c}")

    for c in METADATA_BOOL_COLS:
        if str(df[c].dtype) != "bool":
            errors.append(f"metadata_bool_not_bool:{c}:{df[c].dtype}")
        if df[c].any():
            errors.append(f"metadata_bool_default_not_false:{c}")

    if df["doc_type"].tolist() != ["department", "policy", "contact"]:
        errors.append(f"doc_type_not_normalized:{df['doc_type'].tolist()}")

    return errors


def install_fake_sentence_transformer() -> None:
    module = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def encode(self, texts, normalize_embeddings=True):
            vectors = []
            for text in texts:
                seed = sum(ord(ch) for ch in str(text))
                vec = np.array(
                    [
                        1.0,
                        float(seed % 7) / 7.0,
                        float(seed % 11) / 11.0,
                        float(seed % 13) / 13.0,
                    ],
                    dtype=np.float32,
                )
                if normalize_embeddings:
                    norm = np.linalg.norm(vec)
                    if norm:
                        vec = vec / norm
                vectors.append(vec)
            return np.vstack(vectors)

    module.SentenceTransformer = FakeSentenceTransformer
    sys.modules["sentence_transformers"] = module


def install_fake_rank_bm25() -> None:
    module = types.ModuleType("rank_bm25")

    class FakeBM25Okapi:
        def __init__(self, corpus):
            self.corpus = [list(doc) for doc in corpus]

        def get_scores(self, query_tokens):
            query_set = set(query_tokens)
            return np.array(
                [float(len(query_set.intersection(doc))) for doc in self.corpus],
                dtype=np.float32,
            )

    module.BM25Okapi = FakeBM25Okapi
    sys.modules["rank_bm25"] = module


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def check_query_index_runtime_import() -> list[str]:
    errors = []
    legacy_df = make_legacy_search_df()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        search_df_path = tmp / "search_df.parquet"
        emb_path = tmp / "embeddings.npy"
        bm25_path = tmp / "missing_bm25.pkl"
        tok_path = tmp / "tokenized_corpus.json.gz"
        contact_kws_path = tmp / "contact_keywords.txt"

        legacy_df.to_parquet(search_df_path, index=False)
        embeddings = np.array(
            [
                [1.0, 0.1, 0.0, 0.0],
                [1.0, 0.0, 0.2, 0.0],
                [1.0, 0.0, 0.0, 0.3],
            ],
            dtype=np.float32,
        )
        np.save(emb_path, embeddings)

        tokenized = [
            ["컴퓨터공학부", "소개", "교육과정"],
            ["졸업", "학점", "3년제", "120학점"],
            ["학생성공지원팀", "전화번호", "연락처"],
        ]
        with gzip.open(tok_path, "wt", encoding="utf-8") as f:
            json.dump(tokenized, f, ensure_ascii=False)

        write_text(contact_kws_path, "전화번호\n연락처\n담당자\n")
        write_text(tmp / "data.json", "[]")

        env = {
            "DATA_JSON": str(tmp / "data.json"),
            "ART_DIR": str(tmp),
            "SEARCH_DF_PATH": str(search_df_path),
            "EMB_PATH": str(emb_path),
            "BM25_PATH": str(bm25_path),
            "TOK_PATH": str(tok_path),
            "CONTACTS_CSV": str(tmp / "contacts.csv"),
            "META_PATH": str(tmp / "meta.json"),
            "CONTACT_KWS_PATH": str(contact_kws_path),
        }
        old_env = {k: os.environ.get(k) for k in env}
        os.environ.update(env)

        old_modules = {
            name: sys.modules.get(name)
            for name in ["sentence_transformers", "rank_bm25", "LLM.sub_model.query_index"]
        }

        try:
            sys.modules.pop("LLM.sub_model.query_index", None)
            install_fake_sentence_transformer()
            install_fake_rank_bm25()
            query_index = importlib.import_module("LLM.sub_model.query_index")

            hits = query_index.hybrid_search("졸업학점", top_k=2)
            if hits.empty:
                errors.append("runtime_hybrid_search_empty")

            answer = query_index.build_answer("학생성공지원팀 전화번호", top_k=2)
            if not str(answer.get("answer", "")).strip():
                errors.append("runtime_build_answer_empty")
        except Exception as exc:
            errors.append(f"runtime_import_failed:{type(exc).__name__}:{exc}")
        finally:
            for name, module in old_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="Check query-index schema compatibility")
    ap.add_argument(
        "--out",
        default="",
        help="optional JSON report path; defaults to a temp file",
    )
    args = ap.parse_args()

    legacy_df = make_legacy_search_df()
    normalized = normalize_search_df_schema(legacy_df)
    errors = check_normalized_schema(normalized)
    errors.extend(check_query_index_runtime_import())

    with tempfile.TemporaryDirectory() as td:
        parquet_path = Path(td) / "legacy_search_df.parquet"
        legacy_df.to_parquet(parquet_path, index=False)
        reloaded = normalize_search_df_schema(pd.read_parquet(parquet_path))
        errors.extend(f"parquet_reload:{e}" for e in check_normalized_schema(reloaded))

    report = {
        "check": "query_index_legacy_schema",
        "passed": not errors,
        "legacy_columns": legacy_df.columns.tolist(),
        "normalized_columns": normalized.columns.tolist(),
        "n_rows": int(normalized.shape[0]),
        "errors": errors,
    }

    out_path = Path(args.out) if args.out else Path("/tmp/query_index_compat_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"passed": report["passed"], "errors": errors}, ensure_ascii=False))
    print(f"report={out_path}")

    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
