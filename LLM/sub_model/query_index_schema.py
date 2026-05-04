import numpy as np
import pandas as pd


TEXT_FALLBACK_COLS = ["text_for_embedding", "text_for_bm25", "text_for_answer"]
METADATA_TEXT_COLS = ["chunk_type", "breadcrumb", "leaf_title", "section_title"]
METADATA_BOOL_COLS = [
    "has_phone",
    "has_email",
    "has_date",
    "has_credit",
    "has_policy_keyword",
    "is_privacy_old",
]
BASE_TEXT_COLS = ["doc_type", "title", "text", "unit", "phone", "email", "url"]


def normalize_search_df_schema(search_df: pd.DataFrame) -> pd.DataFrame:
    """Fill newer search-index columns so legacy artifacts remain readable."""
    df = search_df.copy()

    for c in [*BASE_TEXT_COLS, *TEXT_FALLBACK_COLS, *METADATA_TEXT_COLS]:
        if c not in df.columns:
            df[c] = np.nan
    for c in METADATA_BOOL_COLS:
        if c not in df.columns:
            df[c] = False

    df["doc_type"] = df["doc_type"].fillna("").astype(str).str.strip().str.lower()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str)

    for c in TEXT_FALLBACK_COLS:
        df[c] = df[c].fillna(df["text"]).astype(str)
        empty_mask = df[c].str.strip().eq("")
        df.loc[empty_mask, c] = df.loc[empty_mask, "text"]

    for c in METADATA_TEXT_COLS:
        df[c] = df[c].fillna("").astype(str)
    for c in METADATA_BOOL_COLS:
        df[c] = df[c].fillna(False).astype(bool)

    df["unit"] = df["unit"].fillna("").astype(str)
    df["phone"] = df["phone"].fillna("").astype(str).str.replace(r"\s+", "", regex=True)
    df["email"] = df["email"].fillna("").astype(str).str.strip()
    df["url"] = df["url"].fillna("").astype(str).str.strip()

    return df
