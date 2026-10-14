import numpy as np
import pandas as pd
import pytest

from src.data import INTERACTION_COLUMNS, clean_reviews
from src.split import assert_no_leakage, split_report, temporal_split


def _raw(rows: list[dict]) -> pd.DataFrame:
    base = {
        "rating": 5,
        "is_recommended": 1.0,
        "helpfulness": 1.0,
        "total_feedback_count": 0,
        "submission_time": "2020-01-01",
        "review_text": "ok",
        "skin_tone": "fair",
        "skin_type": "dry",
        "eye_color": "blue",
        "hair_color": "black",
    }
    df = pd.DataFrame([{**base, **r} for r in rows])
    return df[INTERACTION_COLUMNS]


def test_drops_corrupted_author_ids():
    raw = _raw(
        [
            {"author_id": "123", "product_id": "P1"},
            {"author_id": "12a3", "product_id": "P1"},
            {"author_id": "dummy", "product_id": "P2"},
        ]
    )
    out = clean_reviews(raw)
    assert out["author_id"].tolist() == ["123"]


def test_scrape_duplicates_keep_row_with_most_feedback():
    raw = _raw(
        [
            {"author_id": "1", "product_id": "P1", "total_feedback_count": 2, "helpfulness": 0.5},
            {"author_id": "1", "product_id": "P1", "total_feedback_count": 9, "helpfulness": 0.9},
        ]
    )
    out = clean_reviews(raw)
    assert len(out) == 1
    assert out.loc[0, "total_feedback_count"] == 9


def test_repeated_pair_keeps_latest_review_not_average():
    raw = _raw(
        [
            {"author_id": "1", "product_id": "P1", "rating": 1, "submission_time": "2019-01-01", "review_text": "bad"},
            {"author_id": "1", "product_id": "P1", "rating": 5, "submission_time": "2021-01-01", "review_text": "great now"},
        ]
    )
    out = clean_reviews(raw)
    assert len(out) == 1
    assert out.loc[0, "rating"] == 5
    assert out.loc[0, "submission_time"] == pd.Timestamp("2021-01-01")


def test_invalid_rating_or_time_rows_are_dropped_and_output_is_sorted():
    raw = _raw(
        [
            {"author_id": "1", "product_id": "P1", "submission_time": "2021-05-01"},
            {"author_id": "2", "product_id": "P1", "submission_time": "not a date"},
            {"author_id": "3", "product_id": "P1", "rating": 9},
            {"author_id": "4", "product_id": "P2", "submission_time": "2020-05-01"},
        ]
    )
    out = clean_reviews(raw)
    assert out["author_id"].tolist() == ["4", "1"]
    assert out["submission_time"].is_monotonic_increasing
    assert out.index.tolist() == [0, 1]


def test_clean_reviews_is_deterministic_under_row_shuffle():
    rows = [
        {"author_id": str(u), "product_id": f"P{p}", "rating": 1 + (u + p) % 5,
         "submission_time": f"2020-0{1 + (u * p) % 9}-01", "review_text": f"t{u}{p}"}
        for u in range(1, 8) for p in range(1, 5)
    ]
    raw = _raw(rows)
    a = clean_reviews(raw)
    b = clean_reviews(raw.sample(frac=1.0, random_state=3))
    pd.testing.assert_frame_equal(a, b)


def _interactions(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
    return pd.DataFrame(
        {
            "author_id": rng.integers(0, 200, n).astype(str),
            "product_id": [f"P{i}" for i in range(n)],  # unique pairs
            "rating": rng.integers(1, 6, n),
            "submission_time": days,
        }
    ).sort_values("submission_time").reset_index(drop=True)


def test_temporal_split_is_strictly_ordered_and_close_to_requested_sizes():
    df = _interactions()
    labels, cut = temporal_split(df, val_frac=0.1, test_frac=0.2)
    assert_no_leakage(df, labels)
    frac = labels.value_counts(normalize=True)
    assert frac["val"] == pytest.approx(0.1, abs=0.03)
    assert frac["test"] == pytest.approx(0.2, abs=0.03)
    assert cut.val_start < cut.test_start


def test_rows_sharing_a_timestamp_never_straddle_a_cutoff():
    df = _interactions()
    labels, _ = temporal_split(df)
    per_day = df.assign(s=labels.to_numpy()).groupby("submission_time")["s"].nunique()
    assert (per_day == 1).all()


def test_split_does_not_depend_on_row_order():
    df = _interactions()
    shuffled = df.sample(frac=1.0, random_state=1)
    a, _ = temporal_split(df)
    b, _ = temporal_split(shuffled)
    assert a.astype(str).sort_index().equals(b.astype(str).sort_index())


def test_assert_no_leakage_catches_overlap_and_pair_in_two_splits():
    df = _interactions()
    labels, _ = temporal_split(df)
    bad = labels.astype(str).copy()
    bad.iloc[0] = "test"  # oldest row moved into test -> train/test overlap in time
    with pytest.raises(AssertionError):
        assert_no_leakage(df, pd.Series(bad, index=df.index))

    dup = pd.concat([df, df.iloc[[0]].assign(submission_time=df["submission_time"].max())], ignore_index=True)
    dup_labels, _ = temporal_split(dup)
    with pytest.raises(AssertionError, match="more than one split"):
        assert_no_leakage(dup, dup_labels)


def test_invalid_fractions_raise():
    df = _interactions()
    with pytest.raises(ValueError):
        temporal_split(df, val_frac=0.6, test_frac=0.5)
    with pytest.raises(ValueError):
        temporal_split(df, val_frac=0.0, test_frac=0.1)


def test_split_report_counts_cold_users_against_train_only():
    df = pd.DataFrame(
        {
            "author_id": ["a", "a", "b", "c"],
            "product_id": ["P1", "P2", "P1", "P3"],
            "rating": [5, 4, 3, 2],
            "submission_time": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
        }
    )
    labels = pd.Series(["train", "train", "val", "test"], index=df.index)
    rep = split_report(df, labels)
    assert rep.loc["val", "cold_user"] == 1.0   # b never seen in train
    assert rep.loc["val", "cold_item"] == 0.0   # P1 seen in train
    assert rep.loc["test", "cold_item"] == 1.0  # P3 never seen
    assert rep.loc["train", "cold_user"] == 0.0
