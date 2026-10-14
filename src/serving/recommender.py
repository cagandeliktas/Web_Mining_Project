"""Request-time scoring that reuses the training feature code.

A request's reviews are treated exactly like a user's earlier history in
training: one extra "query" row, dated after all of them, is appended and run
through ``build_user_asof`` (the same function that built the training
features), then through the train-fitted ``user_tf``. A parity test checks the
result against a real training row. The user bias is the cold slot (0): the
API serves users it does not know by id.
"""

import numpy as np
import pandas as pd
import torch

from src.features.text import embed_texts
from src.features.user import DEMOGRAPHIC_COLUMNS, build_user_asof
from src.serving.bundle import ServingBundle

RATING_MIN, RATING_MAX = 1.0, 5.0
_BASE_TIME = pd.Timestamp("2000-01-01")
_HISTORY_COLUMNS = ["rating", "is_recommended", "review_text", "submission_time", *DEMOGRAPHIC_COLUMNS]


def request_user_features(bundle: ServingBundle, reviews: pd.DataFrame | None, profile: dict | None = None) -> np.ndarray:
    """(1, n_user_features) float32 vector for a user with the given earlier reviews (may be empty)."""
    hist = pd.DataFrame(columns=_HISTORY_COLUMNS) if reviews is None or reviews.empty else reviews.copy()
    for c in _HISTORY_COLUMNS:
        if c not in hist.columns:
            hist[c] = None
    hist = hist[_HISTORY_COLUMNS].reset_index(drop=True)
    hist["rating"] = pd.to_numeric(hist["rating"], errors="coerce")
    hist["is_recommended"] = pd.to_numeric(hist["is_recommended"], errors="coerce")
    times = pd.to_datetime(hist["submission_time"], errors="coerce")
    hist["submission_time"] = times.fillna(times.min() if times.notna().any() else _BASE_TIME)
    query_time = (hist["submission_time"].max() if len(hist) else _BASE_TIME) + pd.Timedelta(days=1)

    # demographics: explicit profile first, else the most recent non-null answer in the history
    latest_first = hist.sort_values("submission_time", ascending=False, kind="stable")
    query = {"rating": np.nan, "is_recommended": np.nan, "review_text": None, "submission_time": query_time}
    for c in DEMOGRAPHIC_COLUMNS:
        given = (profile or {}).get(c)
        fallback = latest_first[c].dropna()
        query[c] = given if given is not None else (fallback.iloc[0] if len(fallback) else None)

    df = pd.concat([hist, pd.DataFrame([query])], ignore_index=True)
    df["submission_time"] = pd.to_datetime(df["submission_time"])
    df.insert(0, "author_id", "__request__")
    emb = embed_texts(bundle.tfidf, bundle.svd, df["review_text"])
    raw = build_user_asof(df, emb).iloc[[-1]]
    return bundle.user_tf.transform(raw)


class Recommender:
    def __init__(self, bundle: ServingBundle):
        self.bundle = bundle
        self.pos = {pid: i for i, pid in enumerate(bundle.item_ids)}

    def user_embedding(self, reviews: pd.DataFrame | None, profile: dict | None = None) -> np.ndarray:
        x = torch.from_numpy(request_user_features(self.bundle, reviews, profile))
        with torch.no_grad():
            return self.bundle.model.user_tower(x).numpy()[0]

    def raw_scores(self, user_emb: np.ndarray) -> np.ndarray:
        """mu + <user, item> + item bias for every catalog product (user bias = cold slot = 0)."""
        b = self.bundle
        return b.mu + b.item_emb @ user_emb + b.item_bias

    def recommend(self, reviews=None, profile=None, top_k: int = 10, exclude=(), min_item_reviews: int = 0) -> pd.DataFrame:
        b = self.bundle
        scores = self.raw_scores(self.user_embedding(reviews, profile))
        allowed = b.item_count >= min_item_reviews
        banned = set(exclude)
        if reviews is not None and "product_id" in getattr(reviews, "columns", []):
            banned |= set(reviews["product_id"].dropna().astype(str))
        if banned:
            allowed &= ~np.isin(b.item_ids, list(banned))
        order = np.argsort(-np.where(allowed, scores, -np.inf), kind="stable")
        order = order[allowed[order]][:top_k]  # rank on the unclipped score; clipping would create ties at 5.0
        out = b.item_meta.iloc[order].reset_index().rename(columns={"index": "product_id"})
        out["predicted_rating"] = np.clip(scores[order], RATING_MIN, RATING_MAX)
        return out

    def predict(self, product_id: str, reviews=None, profile=None) -> float:
        if product_id not in self.pos:
            raise KeyError(product_id)
        score = self.raw_scores(self.user_embedding(reviews, profile))[self.pos[product_id]]
        return float(np.clip(score, RATING_MIN, RATING_MAX))
