"""Global temporal train/val/test split.

Interactions are ordered by ``submission_time``: the oldest go to train, the
newest to test. Cut-offs fall on a timestamp, never inside one, so rows that
share a timestamp (reviews are dated to the day) always land in the same
split. That makes ``max(train) < min(val) <= max(val) < min(test)`` hold
strictly, which is what lets later steps build "as of" features from strictly
earlier rows without peeking at the future.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitCutoffs:
    val_start: pd.Timestamp
    test_start: pd.Timestamp


def temporal_split(
    df: pd.DataFrame,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    time_col: str = "submission_time",
) -> tuple[pd.Series, SplitCutoffs]:
    """Label every row 'train' / 'val' / 'test'; returns (labels, cutoffs).

    Fractions are targets: because cut-offs snap to a timestamp, realised
    sizes differ slightly. ``labels`` is aligned to ``df.index``.
    """
    if val_frac <= 0 or test_frac <= 0:
        raise ValueError("val_frac and test_frac must be positive")
    if val_frac + test_frac >= 1:
        raise ValueError("val_frac + test_frac must be < 1")

    times = df[time_col].sort_values().to_numpy()
    n = len(times)
    val_start = pd.Timestamp(times[int(n * (1 - val_frac - test_frac))])
    test_start = pd.Timestamp(times[int(n * (1 - test_frac))])
    if not val_start < test_start:
        raise ValueError("Timestamps too coarse for the requested split fractions")

    t = df[time_col]
    labels = pd.Series("train", index=df.index, dtype="object")
    labels[t >= val_start] = "val"
    labels[t >= test_start] = "test"
    return labels.astype("category"), SplitCutoffs(val_start, test_start)


def assert_no_leakage(df: pd.DataFrame, labels: pd.Series, time_col: str = "submission_time") -> None:
    """Raise AssertionError if the split lets the future leak into the past."""
    t = df[time_col]
    train_max = t[labels == "train"].max()
    val_min, val_max = t[labels == "val"].min(), t[labels == "val"].max()
    test_min = t[labels == "test"].min()
    assert train_max < val_min, f"train ({train_max}) overlaps val ({val_min})"
    assert val_max < test_min, f"val ({val_max}) overlaps test ({test_min})"

    pairs = df.assign(_s=labels.to_numpy()).groupby(["author_id", "product_id"])["_s"].nunique()
    assert (pairs == 1).all(), "a (user, product) pair appears in more than one split"


def split_report(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Per-split size, date range and cold-start shares (the numbers to sanity-check).

    ``cold_user`` / ``cold_item`` is the share of a split's rows whose user /
    product has no row in *train*; for train itself it is 0 by definition.
    """
    train = df[labels == "train"]
    known_users = set(train["author_id"])
    known_items = set(train["product_id"])
    rows = []
    for name in SPLITS:
        part = df[labels == name]
        rows.append(
            {
                "split": name,
                "rows": len(part),
                "frac": len(part) / len(df),
                "start": part["submission_time"].min(),
                "end": part["submission_time"].max(),
                "cold_user": float((~part["author_id"].isin(known_users)).mean()),
                "cold_item": float((~part["product_id"].isin(known_items)).mean()),
                "mean_rating": float(part["rating"].mean()),
            }
        )
    return pd.DataFrame(rows).set_index("split")
