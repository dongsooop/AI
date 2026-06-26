import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from LLM.sub_model.index_utils import get_tokenizer, load_json_gz
from LLM.sub_model.query_index_schema import normalize_search_df_schema

load_dotenv()

logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent.parent
DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-base"


@dataclass(frozen=True)
class QueryIndexPaths:
    data_json: Path
    art_dir: Path
    search_df_path: Path
    emb_path: Path
    bm25_path: Path
    tok_path: Path
    contacts_csv: Path
    meta_path: Path
    contact_kws_path: Path


@dataclass(frozen=True)
class QueryIndexResources:
    paths: QueryIndexPaths
    contact_kws: list[str]
    embeddings: np.ndarray
    search_df: pd.DataFrame
    tokenizer: Callable[[str], list[str]]
    bm25: BM25Okapi
    bm25_fallback_tier: str
    model_name: str
    model: SentenceTransformer


def env_path(name: str) -> Path:
    val = os.getenv(name, "").strip()
    if not val:
        raise FileNotFoundError(f"{name} environment empty.")
    s = os.path.expandvars(val)
    p = Path(s)
    if not p.is_absolute():
        p = ROOT_DIR / p
    return p.resolve()


def load_list_from_txt(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_query_index_paths() -> QueryIndexPaths:
    return QueryIndexPaths(
        data_json=env_path("DATA_JSON"),
        art_dir=env_path("ART_DIR"),
        search_df_path=env_path("SEARCH_DF_PATH"),
        emb_path=env_path("EMB_PATH"),
        bm25_path=env_path("BM25_PATH"),
        tok_path=env_path("TOK_PATH"),
        contacts_csv=env_path("CONTACTS_CSV"),
        meta_path=env_path("META_PATH"),
        contact_kws_path=env_path("CONTACT_KWS_PATH"),
    )


def load_embeddings(path: Path) -> np.ndarray:
    embeddings = np.load(path, mmap_mode="r").astype(np.float32)
    row_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    return embeddings / row_norms


def load_bm25(path: Path, tok_path: Path, search_df: pd.DataFrame, tokenizer) -> tuple[BM25Okapi, str]:
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f), "pickle"
        except Exception as exc:
            logger.warning(
                "query_index_bm25_pickle_load_failed fallback=tokenized_or_runtime bm25_file=%s error=%s",
                path.name,
                type(exc).__name__,
            )

    if tok_path.exists():
        try:
            fallback_tier = "tokenized_corpus"
            logger.warning(
                "query_index_bm25_rebuild fallback=tokenized_corpus bm25_file=%s tokenized_file=%s",
                path.name,
                tok_path.name,
            )
            tokenized_corpus = load_json_gz(str(tok_path))
        except Exception as exc:
            fallback_tier = "runtime_tokenize"
            logger.warning(
                "query_index_bm25_tokenized_load_failed fallback=runtime_tokenize bm25_file=%s tokenized_file=%s error=%s",
                path.name,
                tok_path.name,
                type(exc).__name__,
            )
            tokenized_corpus = [
                tokenizer(t)
                for t in search_df["text_for_bm25"].astype(str).tolist()
            ]
    else:
        fallback_tier = "runtime_tokenize"
        logger.warning(
            "query_index_bm25_rebuild fallback=runtime_tokenize bm25_file=%s tokenized_file=%s documents=%s",
            path.name,
            tok_path.name,
            len(search_df),
        )
        tokenized_corpus = [
            tokenizer(t)
            for t in search_df["text_for_bm25"].astype(str).tolist()
        ]
    return BM25Okapi(tokenized_corpus), fallback_tier


def load_query_index_resources(
    paths: QueryIndexPaths | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> QueryIndexResources:
    paths = paths or load_query_index_paths()
    search_df = normalize_search_df_schema(pd.read_parquet(paths.search_df_path))
    tokenizer = get_tokenizer()
    bm25, bm25_fallback_tier = load_bm25(paths.bm25_path, paths.tok_path, search_df, tokenizer)

    return QueryIndexResources(
        paths=paths,
        contact_kws=load_list_from_txt(paths.contact_kws_path),
        embeddings=load_embeddings(paths.emb_path),
        search_df=search_df,
        tokenizer=tokenizer,
        bm25=bm25,
        bm25_fallback_tier=bm25_fallback_tier,
        model_name=model_name,
        model=SentenceTransformer(model_name),
    )
