"""Load and clean the raw Sephora reviews into one row per (user, product).

In plain words: Loads the raw review CSVs and cleans them: drops broken user ids, removes duplicate
scrapes and keeps one row per (user, product) - the user's latest review.

The cleaning rules mirror dbt_project/models/staging/stg_reviews.sql so the
Python pipeline and the warehouse agree on what a "review" is:

1. drop corrupted author_ids (letters mixed into a numeric id);
2. collapse scrape duplicates that agree on (author_id, product_id,
   submission_time, review_text), keeping the row with the most feedback.

On top of that this module reduces to one *interaction* per (author_id,
product_id), keeping the user's **latest** review of that product. The old
pipeline averaged repeated ratings, which blends a later review into an
earlier timestamp; under a temporal split that would leak. Keeping one row
also means a (user, product) pair can never straddle train and test.
"""

from pathlib import Path

import pandas as pd

REVIEW_FILES = [
    "reviews_0-250.csv",
    "reviews_250-500.csv",
    "reviews_500-750.csv",
    "reviews_750-1250.csv",
    "reviews_1250-end.csv",
]

INTERACTION_COLUMNS = [
    "author_id",
    "product_id",
    "rating",
    "is_recommended",
    "helpfulness",
    "total_feedback_count",
    "submission_time",
    "review_text",
    "skin_tone",
    "skin_type",
    "eye_color",
    "hair_color",
]

_DEDUP_KEY = ["author_id", "product_id", "submission_time", "review_text"]


def load_raw_reviews(data_dir: str | Path) -> pd.DataFrame:
    """Concatenate the five raw review CSVs, keeping only the modelling columns."""
    data_dir = Path(data_dir)
    frames = [
        pd.read_csv(data_dir / f, usecols=INTERACTION_COLUMNS, low_memory=False)
        for f in REVIEW_FILES
    ]
    return pd.concat(frames, ignore_index=True)


def clean_reviews(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the stg_reviews rules, then keep each user's latest review per product.

    Returns a frame sorted by (submission_time, author_id, product_id) with a
    fresh RangeIndex, so row order is deterministic.
    """
    df = raw[INTERACTION_COLUMNS].copy()
    df["author_id"] = df["author_id"].astype(str)
    df["product_id"] = df["product_id"].astype(str)
    df = df.loc[~df["author_id"].str.contains(r"[a-zA-Z]", regex=True, na=False)]

    df["submission_time"] = pd.to_datetime(df["submission_time"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["submission_time", "rating"])
    df = df[df["rating"].between(1, 5)]

    # Scrape duplicates: keep the most complete snapshot (most feedback recorded).
    df = df.sort_values(
        ["total_feedback_count", "author_id", "product_id"],
        ascending=[False, True, True],
        na_position="last",
        kind="stable",
    )
    df = df.drop_duplicates(subset=_DEDUP_KEY, keep="first")

    # One interaction per (user, product): the latest review wins.
    df = df.sort_values(
        ["submission_time", "total_feedback_count"],
        ascending=[True, True],
        na_position="first",
        kind="stable",
    )
    df = df.drop_duplicates(subset=["author_id", "product_id"], keep="last")

    df = df.sort_values(
        ["submission_time", "author_id", "product_id"], kind="stable"
    ).reset_index(drop=True)
    return df


def load_interactions(
    data_dir: str | Path, cache_dir: str | Path | None = None, refresh: bool = False
) -> pd.DataFrame:
    """Cleaned interactions, read from a parquet cache when one exists."""
    cache_path = Path(cache_dir) / "interactions.parquet" if cache_dir else None
    if cache_path is not None and cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    df = clean_reviews(load_raw_reviews(data_dir))
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    return df
