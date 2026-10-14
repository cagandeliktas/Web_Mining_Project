"""Interactions + product catalog -> model-ready, leak-free feature arrays.

In plain words: The glue that turns cleaned reviews plus the product catalogue into model-ready
arrays, in the safe order: split by time first, fit everything on train only.

Order of operations (the leakage contract):
1. the temporal split labels are decided first;
2. everything *fitted* (TF-IDF, SVD, imputation values, scalers, category and
   id vocabularies, the global mean rating) sees train rows only;
3. everything *computed per row* (user history, item popularity) uses only
   strictly earlier interactions, whichever split they fall in - that is what
   a deployed system would know at that moment.
"""

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.item import (
    STATIC_BINARY,
    STATIC_CATEGORICAL,
    build_item_asof,
    build_item_static,
    join_item_static,
)
from src.features.text import embed_texts, fit_text_model
from src.features.user import DEMOGRAPHIC_COLUMNS, build_user_asof, text_columns
from src.transforms import IdVocab, TabularTransformer

META_COLUMNS = ["author_id", "product_id", "submission_time"]


@dataclass
class FeatureSet:
    """Everything the model needs, as arrays: user features, item features, ids, true ratings, which
    split each row belongs to, and the fitted preprocessing objects (which serving must reuse).
    """
    X_user: np.ndarray            # (n, d_user) float32
    X_item: np.ndarray            # (n, d_item) float32
    user_idx: np.ndarray          # (n,) int64, 0 = user unseen in train
    item_idx: np.ndarray          # (n,) int64, 0 = item unseen in train
    y: np.ndarray                 # (n,) float32, raw 1-5 rating (no target scaling)
    split: np.ndarray             # (n,) '<U5' in {train, val, test}
    meta: pd.DataFrame            # author_id, product_id, submission_time
    mu: float                     # train mean rating (bias-head intercept)
    artifacts: dict = field(default_factory=dict)  # tfidf, svd, user_tf, item_tf, vocabs

    def part(self, name: str) -> dict:
        """Return one split ("train", "val" or "test") as a dict of arrays."""
        m = self.split == name
        return {
            "X_user": self.X_user[m],
            "X_item": self.X_item[m],
            "user_idx": self.user_idx[m],
            "item_idx": self.item_idx[m],
            "y": self.y[m],
            "meta": self.meta[m].reset_index(drop=True),
        }

    def save(self, directory) -> None:
        """Write the arrays and fitted objects to a folder (the cache)."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(
            d / "arrays.npz",
            X_user=self.X_user, X_item=self.X_item, user_idx=self.user_idx,
            item_idx=self.item_idx, y=self.y, split=self.split,
        )
        self.meta.to_parquet(d / "meta.parquet", index=False)
        joblib.dump({"mu": self.mu, "artifacts": self.artifacts}, d / "artifacts.joblib")

    @classmethod
    def load(cls, directory) -> "FeatureSet":
        """Read a FeatureSet back from the cache folder."""
        d = Path(directory)
        a = np.load(d / "arrays.npz")
        extra = joblib.load(d / "artifacts.joblib")
        return cls(
            X_user=a["X_user"], X_item=a["X_item"], user_idx=a["user_idx"], item_idx=a["item_idx"],
            y=a["y"], split=a["split"], meta=pd.read_parquet(d / "meta.parquet"),
            mu=extra["mu"], artifacts=extra["artifacts"],
        )


def build_features(
    interactions: pd.DataFrame,
    product_info: pd.DataFrame,
    labels: pd.Series,
    cfg,
) -> FeatureSet:
    """Build the full feature set. ``labels`` is the temporal split (aligned to ``interactions.index``)."""
    df = interactions.reset_index(drop=True)
    is_train = (labels.to_numpy() == "train")
    fc = cfg.features

    mu = float(df.loc[is_train, "rating"].mean())

    tfidf, svd = fit_text_model(
        df.loc[is_train, "review_text"],
        max_features=fc.tfidf_max_features, min_df=fc.tfidf_min_df, max_df=fc.tfidf_max_df,
        n_components=fc.svd_components, seed=cfg.seed,
    )
    text_emb = embed_texts(tfidf, svd, df["review_text"])

    user_raw = build_user_asof(df, text_emb)
    item_raw = pd.concat(
        [
            build_item_asof(df, prior_mean=mu, shrinkage=fc.item_shrinkage),
            join_item_static(df, build_item_static(product_info)),
        ],
        axis=1,
    )

    tcols = text_columns(fc.svd_components)
    user_tf = TabularTransformer(
        numeric=["u_hist_count", "u_rating_mean", "u_rec_mean"] + tcols,
        log1p=["u_hist_count"],
        binary=["u_has_history"],
        categorical=DEMOGRAPHIC_COLUMNS,
        missing_indicators=["u_rec_mean", tcols[0]],
    ).fit(user_raw[is_train])
    item_tf = TabularTransformer(
        numeric=["i_hist_count", "i_rating_mean", "price_usd", "child_count", "child_max_price",
                 "child_min_price", "size_ml", "brand_freq"],
        log1p=["i_hist_count", "price_usd", "child_max_price", "child_min_price", "size_ml", "brand_freq"],
        binary=STATIC_BINARY,
        categorical=STATIC_CATEGORICAL,
        missing_indicators=["size_ml", "price_usd"],
    ).fit(item_raw[is_train])

    user_vocab = IdVocab().fit(df.loc[is_train, "author_id"])
    item_vocab = IdVocab().fit(df.loc[is_train, "product_id"])

    meta = df[META_COLUMNS].copy()
    meta["u_hist_count"] = user_raw["u_hist_count"].to_numpy()  # raw prior counts, for bucketed evaluation
    meta["i_hist_count"] = item_raw["i_hist_count"].to_numpy()

    return FeatureSet(
        X_user=user_tf.transform(user_raw),
        X_item=item_tf.transform(item_raw),
        user_idx=user_vocab.transform(df["author_id"]),
        item_idx=item_vocab.transform(df["product_id"]),
        y=df["rating"].to_numpy(dtype=np.float32),
        split=labels.astype(str).to_numpy(dtype="U5"),
        meta=meta,
        mu=mu,
        artifacts={"tfidf": tfidf, "svd": svd, "user_tf": user_tf, "item_tf": item_tf,
                   "user_vocab": user_vocab, "item_vocab": item_vocab},
    )


def load_or_build_features(cfg, refresh: bool = False, refresh_data: bool = False) -> FeatureSet:
    """Cached end-to-end path: raw CSVs -> clean -> temporal split -> features.

    The cache lives in ``<cache_dir>/features``; pass ``refresh=True`` after
    changing anything under ``features:``/``split:`` in the config.
    """
    import pandas as pd

    from src.data import load_interactions
    from src.split import assert_no_leakage, temporal_split

    cache = Path(cfg.data.cache_dir) / "features"
    if (cache / "arrays.npz").exists() and not refresh:
        return FeatureSet.load(cache)
    df = load_interactions(cfg.data.dir, cfg.data.cache_dir, refresh=refresh_data)
    labels, _ = temporal_split(df, cfg.split.val_frac, cfg.split.test_frac)
    assert_no_leakage(df, labels)
    fs = build_features(df, pd.read_csv(cfg.data.product_info), labels, cfg)
    fs.save(cache)
    return fs
