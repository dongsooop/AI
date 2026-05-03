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


def install_fake_sentence_transformer() -> None:
    module = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def encode(self, texts, normalize_embeddings=True):
            vectors = []
            for _ in texts:
                vec = np.ones(4, dtype=np.float32)
                if normalize_embeddings:
                    vec = vec / np.linalg.norm(vec)
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


def make_metadata_rich_df() -> pd.DataFrame:
    rows = [
        {
            "doc_id": 1,
            "chunk_id": "1-0",
            "parent_id": 1,
            "title": "졸업 학점 안내",
            "url": "https://www.dongyang.ac.kr/example/grad",
            "source": "main",
            "text": "3년제 졸업학점 졸업 학점 졸업이수 학점 총 120학점 전공최저이수 학점 안내",
            "text_for_embedding": "졸업학점 3년제 총 120학점",
            "text_for_bm25": "3년제 졸업학점 졸업 학점 120학점",
            "text_for_answer": "3년제 졸업이수 학점은 총 120학점입니다.",
            "doc_type": "policy",
            "chunk_type": "policy",
            "breadcrumb": "학사안내/졸업",
            "leaf_title": "졸업 학점 안내",
            "section_title": "졸업",
            "has_phone": False,
            "has_email": False,
            "has_date": False,
            "has_credit": True,
            "has_policy_keyword": True,
            "is_privacy_old": False,
            "unit": "",
            "phone": "",
            "email": "",
        },
        {
            "doc_id": 2,
            "chunk_id": "2-0",
            "parent_id": 2,
            "title": "학생성공지원팀 연락처",
            "url": "https://www.dongyang.ac.kr/example/contact",
            "source": "main",
            "text": "학생성공지원팀 전화번호 연락처 담당자 02-2610-1234 이메일 success@example.ac.kr",
            "text_for_embedding": "학생성공지원팀 전화번호 연락처",
            "text_for_bm25": "학생성공지원팀 전화번호 연락처 담당자",
            "text_for_answer": "학생성공지원팀 전화번호 02-2610-1234",
            "doc_type": "contact",
            "chunk_type": "contact",
            "breadcrumb": "대학안내/연락처",
            "leaf_title": "학생성공지원팀 연락처",
            "section_title": "연락처",
            "has_phone": True,
            "has_email": True,
            "has_date": False,
            "has_credit": False,
            "has_policy_keyword": False,
            "is_privacy_old": False,
            "unit": "학생성공지원팀",
            "phone": "02-2610-1234",
            "email": "success@example.ac.kr",
        },
        {
            "doc_id": 3,
            "chunk_id": "3-0",
            "parent_id": 3,
            "title": "컴퓨터공학부 소개",
            "url": "https://www.dongyang.ac.kr/example/computer",
            "source": "main",
            "text": "컴퓨터공학부 소개 전공 교육과정 학부 진로 안내",
            "text_for_embedding": "컴퓨터공학부 소개 학부 전공",
            "text_for_bm25": "컴퓨터공학부 소개 학부 전공",
            "text_for_answer": "컴퓨터공학부는 소프트웨어와 컴퓨터 전공 교육과정을 운영합니다.",
            "doc_type": "department",
            "chunk_type": "department",
            "breadcrumb": "학부/컴퓨터공학부",
            "leaf_title": "컴퓨터공학부",
            "section_title": "소개",
            "has_phone": False,
            "has_email": False,
            "has_date": False,
            "has_credit": False,
            "has_policy_keyword": False,
            "is_privacy_old": False,
            "unit": "컴퓨터공학부",
            "phone": "",
            "email": "",
        },
        {
            "doc_id": 4,
            "chunk_id": "4-0",
            "parent_id": 4,
            "title": "개인정보 처리방침",
            "url": "https://www.dongyang.ac.kr/example/privacy",
            "source": "main",
            "text": "개인정보 처리방침 영상정보 이메일무단수집거부 안내",
            "text_for_embedding": "개인정보 처리방침",
            "text_for_bm25": "개인정보 처리방침 영상정보",
            "text_for_answer": "개인정보 처리방침 안내입니다.",
            "doc_type": "privacy",
            "chunk_type": "privacy",
            "breadcrumb": "이용안내/개인정보",
            "leaf_title": "개인정보 처리방침",
            "section_title": "개인정보",
            "has_phone": False,
            "has_email": False,
            "has_date": False,
            "has_credit": False,
            "has_policy_keyword": False,
            "is_privacy_old": False,
            "unit": "",
            "phone": "",
            "email": "",
        },
        {
            "doc_id": 5,
            "chunk_id": "5-0",
            "parent_id": 5,
            "title": "장학 신청 안내",
            "url": "https://www.dongyang.ac.kr/example/scholarship",
            "source": "main",
            "text": "장학 신청 성적 학점 기준 안내",
            "text_for_embedding": "장학 신청 학점",
            "text_for_bm25": "장학 신청 성적 학점",
            "text_for_answer": "장학 신청 기준 안내입니다.",
            "doc_type": "policy",
            "chunk_type": "policy",
            "breadcrumb": "학사안내/장학",
            "leaf_title": "장학 신청 안내",
            "section_title": "장학",
            "has_phone": False,
            "has_email": False,
            "has_date": False,
            "has_credit": False,
            "has_policy_keyword": True,
            "is_privacy_old": False,
            "unit": "",
            "phone": "",
            "email": "",
        },
    ]
    return pd.DataFrame(rows)


def make_sparse_metadata_df(df: pd.DataFrame) -> pd.DataFrame:
    sparse = df.copy()
    sparse = sparse.drop(
        columns=[
            "chunk_type",
            "breadcrumb",
            "leaf_title",
            "section_title",
            "has_phone",
            "has_email",
            "has_date",
            "has_credit",
            "has_policy_keyword",
            "is_privacy_old",
        ]
    )
    sparse["unit"] = ""
    sparse["phone"] = ""
    sparse["email"] = ""
    return sparse


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def import_query_index(df: pd.DataFrame, tmp: Path):
    search_df_path = tmp / "search_df.parquet"
    emb_path = tmp / "embeddings.npy"
    tok_path = tmp / "tokenized_corpus.json.gz"
    contact_kws_path = tmp / "contact_keywords.txt"

    df.to_parquet(search_df_path, index=False)
    np.save(emb_path, np.ones((len(df), 4), dtype=np.float32))

    tokenized = [str(text).split() for text in df["text_for_bm25"].fillna(df["text"]).astype(str)]
    with gzip.open(tok_path, "wt", encoding="utf-8") as f:
        json.dump(tokenized, f, ensure_ascii=False)

    write_text(contact_kws_path, "전화번호\n연락처\n담당자\n")
    write_text(tmp / "data.json", "[]")

    env = {
        "DATA_JSON": str(tmp / "data.json"),
        "ART_DIR": str(tmp),
        "SEARCH_DF_PATH": str(search_df_path),
        "EMB_PATH": str(emb_path),
        "BM25_PATH": str(tmp / "missing_bm25.pkl"),
        "TOK_PATH": str(tok_path),
        "CONTACTS_CSV": str(tmp / "contacts.csv"),
        "META_PATH": str(tmp / "meta.json"),
        "CONTACT_KWS_PATH": str(contact_kws_path),
    }
    os.environ.update(env)
    sys.modules.pop("LLM.sub_model.query_index", None)
    return importlib.import_module("LLM.sub_model.query_index")


def rank_for_url(hits: pd.DataFrame, expected_url: str) -> int | None:
    urls = hits["url"].astype(str).tolist()
    if expected_url not in urls:
        return None
    return urls.index(expected_url) + 1


def run_quality_checks() -> tuple[list[dict], list[str]]:
    cases = [
        {
            "id": "grad_credit",
            "query": "3년제 졸업학점",
            "expected_url": "https://www.dongyang.ac.kr/example/grad",
        },
        {
            "id": "contact_phone",
            "query": "학생성공지원팀 전화번호",
            "expected_url": "https://www.dongyang.ac.kr/example/contact",
        },
        {
            "id": "department_intro",
            "query": "컴퓨터공학부 소개",
            "expected_url": "https://www.dongyang.ac.kr/example/computer",
        },
        {
            "id": "privacy_policy",
            "query": "개인정보 처리방침",
            "expected_url": "https://www.dongyang.ac.kr/example/privacy",
        },
    ]

    errors = []
    results = []
    rich_df = make_metadata_rich_df()
    sparse_df = make_sparse_metadata_df(rich_df)

    old_env = {
        key: os.environ.get(key)
        for key in [
            "DATA_JSON",
            "ART_DIR",
            "SEARCH_DF_PATH",
            "EMB_PATH",
            "BM25_PATH",
            "TOK_PATH",
            "CONTACTS_CSV",
            "META_PATH",
            "CONTACT_KWS_PATH",
        ]
    }
    old_modules = {
        name: sys.modules.get(name)
        for name in ["sentence_transformers", "rank_bm25", "LLM.sub_model.query_index"]
    }

    try:
        install_fake_sentence_transformer()
        install_fake_rank_bm25()

        with tempfile.TemporaryDirectory() as rich_td, tempfile.TemporaryDirectory() as sparse_td:
            rich_index = import_query_index(rich_df, Path(rich_td))
            rich_hits_by_case = {
                case["id"]: rich_index.hybrid_search(case["query"], top_k=5, alpha=0.0)
                for case in cases
            }

            sparse_index = import_query_index(sparse_df, Path(sparse_td))
            sparse_hits_by_case = {
                case["id"]: sparse_index.hybrid_search(case["query"], top_k=5, alpha=0.0)
                for case in cases
            }

            for case in cases:
                rich_hits = rich_hits_by_case[case["id"]]
                sparse_hits = sparse_hits_by_case[case["id"]]
                rich_rank = rank_for_url(rich_hits, case["expected_url"])
                sparse_rank = rank_for_url(sparse_hits, case["expected_url"])

                if rich_rank != 1:
                    errors.append(f"{case['id']}:rich_expected_not_top:{rich_rank}")
                if sparse_rank != 1:
                    errors.append(f"{case['id']}:sparse_expected_not_top:{sparse_rank}")

                results.append(
                    {
                        "id": case["id"],
                        "query": case["query"],
                        "expected_url": case["expected_url"],
                        "rich_rank": rich_rank,
                        "sparse_rank": sparse_rank,
                        "rich_top_url": rich_hits.iloc[0]["url"] if not rich_hits.empty else "",
                        "sparse_top_url": sparse_hits.iloc[0]["url"] if not sparse_hits.empty else "",
                    }
                )
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

    return results, errors


def main() -> int:
    ap = argparse.ArgumentParser(description="Check query-index quality with sparse metadata")
    ap.add_argument("--out", default="", help="optional JSON report path")
    args = ap.parse_args()

    results, errors = run_quality_checks()
    report = {
        "check": "query_index_sparse_metadata_quality",
        "passed": not errors,
        "results": results,
        "errors": errors,
    }

    out_path = Path(args.out) if args.out else Path("/tmp/query_index_metadata_quality_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"passed": report["passed"], "errors": errors}, ensure_ascii=False))
    print(f"report={out_path}")

    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
