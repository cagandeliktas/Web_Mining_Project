"""Leakage and correctness tests for the as-of features and the fit-on-train pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.features.asof import asof_prior_sums
from src.features.item import build_item_asof
from src.features.user import build_user_asof, text_columns
from src.pipeline import FeatureSet, build_features
from src.split import temporal_split
from src.transforms import IdVocab, TabularTransformer
from tests.synthetic import make_interactions, make_product_info, small_cfg


# ---------------------------------------------------------------- as-of correctness
def test_asof_prior_sums_matches_brute_force_with_ties():
    df = make_interactions()
    vals = pd.DataFrame({"n": 1.0, "r": df["rating"].astype(float)}, index=df.index)
    got = asof_prior_sums(df["author_id"], df["submission_time"], vals)
    for i in range(0, len(df), 7):
        row = df.iloc[i]
        prior = df[(df["author_id"] == row["author_id"]) & (df["submission_time"] < row["submission_time"])]
        assert got.loc[df.index[i], "n"] == len(prior)
        assert got.loc[df.index[i], "r"] == pytest.approx(prior["rating"].sum())


def test_user_features_match_brute_force_definition():
    df = make_interactions()
    emb = np.random.default_rng(2).normal(size=(len(df), 4)).astype(np.float32)
    feats = build_user_asof(df, emb)
    tcols = text_columns(4)
    for i in range(0, len(df), 5):
        row = df.iloc[i]
        mask = (df["author_id"] == row["author_id"]) & (df["submission_time"] < row["submission_time"])
        prior = df[mask]
        f = feats.iloc[i]
        assert f["u_hist_count"] == len(prior)
        assert f["u_has_history"] == float(len(prior) > 0)
        if len(prior):
            assert f["u_rating_mean"] == pytest.approx(prior["rating"].mean())
            rec = prior["is_recommended"].dropna()
            assert (np.isnan(f["u_rec_mean"]) if rec.empty else f["u_rec_mean"] == pytest.approx(rec.mean()))
        else:
            assert np.isnan(f["u_rating_mean"])
            assert np.isnan(f[tcols].astype(float)).all()


# ---------------------------------------------------------------- leakage (causal) tests
def _scramble(df: pd.DataFrame, emb: np.ndarray, rows: np.ndarray, seed=9):
    """Overwrite label-bearing content of ``rows`` (rating, recommendation, review text/embedding)."""
    rng = np.random.default_rng(seed)
    d2, e2 = df.copy(), emb.copy()
    d2.loc[rows, "rating"] = (d2.loc[rows, "rating"] % 5) + 1
    d2.loc[rows, "is_recommended"] = 1.0 - d2.loc[rows, "is_recommended"].fillna(0.5)
    d2.loc[rows, "review_text"] = "scrambled words here"
    e2[rows] = rng.normal(size=(len(rows), emb.shape[1]))
    return d2, e2


def test_a_rows_own_label_and_text_never_change_its_features():
    df = make_interactions()
    emb = np.random.default_rng(3).normal(size=(len(df), 4)).astype(np.float32)
    base_u, base_i = build_user_asof(df, emb), build_item_asof(df, 3.8, 10)
    for i in [0, 5, 50, 200, len(df) - 1]:
        d2, e2 = _scramble(df, emb, np.array([i]))
        pd.testing.assert_series_equal(build_user_asof(d2, e2).iloc[i], base_u.iloc[i], check_names=False)
        pd.testing.assert_series_equal(build_item_asof(d2, 3.8, 10).iloc[i], base_i.iloc[i], check_names=False)


def test_features_up_to_time_T_ignore_everything_at_or_after_T():
    df = make_interactions()
    emb = np.random.default_rng(4).normal(size=(len(df), 4)).astype(np.float32)
    T = df["submission_time"].quantile(0.5, interpolation="nearest")
    future = np.where(df["submission_time"] >= T)[0]
    d2, e2 = _scramble(df, emb, future)

    a_u, b_u = build_user_asof(df, emb), build_user_asof(d2, e2)
    a_i, b_i = build_item_asof(df, 3.8, 10), build_item_asof(d2, 3.8, 10)
    upto = (df["submission_time"] <= T).to_numpy()
    pd.testing.assert_frame_equal(a_u[upto], b_u[upto])
    pd.testing.assert_frame_equal(a_i[upto], b_i[upto])

    # sanity: the test can fail - later rows really do depend on what was scrambled
    later = (df["submission_time"] > T).to_numpy()
    assert not a_i[later].equals(b_i[later])


def test_everything_fitted_sees_train_rows_only():
    df = make_interactions()
    info = make_product_info()
    labels, _ = temporal_split(df, 0.15, 0.15)
    cfg = small_cfg()
    base = build_features(df, info, labels, cfg)

    nontrain = np.where(labels.to_numpy() != "train")[0]
    emb_dummy = np.zeros((len(df), 1))
    d2, _ = _scramble(df, emb_dummy, nontrain)
    other = build_features(d2, info, labels, cfg)

    assert other.mu == base.mu
    a, b = base.artifacts, other.artifacts
    np.testing.assert_array_equal(a["svd"].components_, b["svd"].components_)
    assert a["tfidf"].vocabulary_ == b["tfidf"].vocabulary_
    for c in a["user_tf"].mean_:
        assert a["user_tf"].mean_[c] == b["user_tf"].mean_[c]
        assert a["user_tf"].fill_[c] == b["user_tf"].fill_[c]
    for c in a["item_tf"].std_:
        assert a["item_tf"].std_[c] == b["item_tf"].std_[c]
    assert a["user_vocab"].index_ == b["user_vocab"].index_
    tr = base.split == "train"
    np.testing.assert_array_equal(base.X_user[tr], other.X_user[tr])
    np.testing.assert_array_equal(base.X_item[tr], other.X_item[tr])


def test_snapshot_catalog_columns_are_not_model_inputs():
    df = make_interactions()
    labels, _ = temporal_split(df)
    fs = build_features(df, make_product_info(), labels, small_cfg())
    names = fs.artifacts["item_tf"].feature_names_
    for banned in ["rating", "reviews", "loves_count", "out_of_stock"]:
        assert banned not in names
    assert not any("helpful" in n or "feedback" in n for n in fs.artifacts["user_tf"].feature_names_)


# ---------------------------------------------------------------- pipeline output shape / transforms
def test_feature_set_shapes_finite_and_roundtrip(tmp_path):
    df = make_interactions()
    labels, _ = temporal_split(df)
    fs = build_features(df, make_product_info(), labels, small_cfg())
    n = len(df)
    assert fs.X_user.shape == (n, len(fs.artifacts["user_tf"].feature_names_))
    assert fs.X_item.shape == (n, len(fs.artifacts["item_tf"].feature_names_))
    assert np.isfinite(fs.X_user).all() and np.isfinite(fs.X_item).all()
    assert fs.X_user.dtype == np.float32 and fs.y.dtype == np.float32
    assert set(fs.split) == {"train", "val", "test"}

    fs.save(tmp_path)
    back = FeatureSet.load(tmp_path)
    np.testing.assert_array_equal(back.X_user, fs.X_user)
    np.testing.assert_array_equal(back.user_idx, fs.user_idx)
    assert back.mu == fs.mu
    assert back.artifacts["user_tf"].feature_names_ == fs.artifacts["user_tf"].feature_names_


def test_cold_ids_map_to_slot_zero():
    vocab = IdVocab().fit(pd.Series(["a", "b", "a"]))
    assert len(vocab) == 3
    assert vocab.transform(pd.Series(["a", "b", "zzz", None])).tolist() == [1, 2, 0, 0]


def test_tabular_transformer_scales_on_fit_data_and_handles_unseen_and_missing():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan], "b": [0, 1, 1, 0], "c": ["u", "v", "u", None]})
    tf = TabularTransformer(numeric=["x"], binary=["b"], categorical=["c"], missing_indicators=["x"]).fit(train)
    out = tf.transform(train)
    assert tf.feature_names_ == ["x", "x__missing", "b", "c=u", "c=v", "c=__other__"]
    assert out[:, 0].mean() == pytest.approx(0.0, abs=1e-6)
    assert out[:, 1].tolist() == [0, 0, 0, 1]
    test = pd.DataFrame({"x": [np.nan], "b": [np.nan], "c": ["never seen"]})
    row = tf.transform(test)[0]
    assert np.isfinite(row).all()
    assert row[-1] == 1.0 and row[3] == 0.0 and row[4] == 0.0  # unseen category -> __other__
