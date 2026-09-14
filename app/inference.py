"""Loads the trained two-tower model and its artifacts once, and scores a
user against the full product catalog.

Item features come pre-scaled from artifacts/item_matrix.pkl (built by
scripts/build_item_matrix.py using the same StandardScaler the model was
trained on) - see that script's docstring and README.md for why this
intentionally differs from the notebook's own (buggy, unscaled) example
cells.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.schemas import ProductRecommendation, ReviewIn
from src.user_features import build_user_vector

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "NeuralNetworkApproach"
    / "final_two_tower_model_last.keras"
)


def _l2_normalize(x):
    return tf.math.l2_normalize(x, axis=1)


def _to_float32(t):
    return tf.cast(t, tf.float32)


class RecommenderService:
    """Holds every loaded artifact needed to score recommendations."""

    def __init__(
        self, artifacts_dir: Path = ARTIFACTS_DIR, model_path: Path = MODEL_PATH
    ):
        import keras

        self.tfidf = joblib.load(artifacts_dir / "tfidf.pkl")
        self.svd = joblib.load(artifacts_dir / "svd.pkl")
        self.defaults = joblib.load(artifacts_dir / "defaults.pkl")
        self.scaler_user = joblib.load(artifacts_dir / "scaler_user.pkl")
        self.scaler_target = joblib.load(artifacts_dir / "scaler_target.pkl")
        self.user_cols = joblib.load(artifacts_dir / "user_feature_cols.pkl")
        self.analyzer = SentimentIntensityAnalyzer()

        item_matrix = joblib.load(artifacts_dir / "item_matrix.pkl")
        self.item_ids = item_matrix["product_ids"]
        self.item_matrix = item_matrix["matrix"]

        self.product_info = pd.read_csv(artifacts_dir / "product_info.csv").set_index(
            "product_id"
        )

        self.model = keras.models.load_model(
            model_path,
            custom_objects={"l2_normalize": _l2_normalize, "to_float32": _to_float32},
            compile=False,
            safe_mode=False,
        )

    def _reviews_to_dataframe(self, reviews: list[ReviewIn]) -> pd.DataFrame:
        if not reviews:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "rating": r.rating,
                    "is_recommended": (
                        None if r.is_recommended is None else int(r.is_recommended)
                    ),
                    "helpfulness": r.helpfulness,
                    "review_text": r.review_text,
                    "skin_tone": r.skin_tone,
                    "skin_type": r.skin_type,
                    "eye_color": r.eye_color,
                    "hair_color": r.hair_color,
                }
                for r in reviews
            ]
        )

    def recommend(
        self, reviews: list[ReviewIn], top_k: int
    ) -> list[ProductRecommendation]:
        raw_reviews = self._reviews_to_dataframe(reviews)
        u_vec = build_user_vector(
            raw_reviews,
            user_cols=self.user_cols,
            tfidf=self.tfidf,
            svd=self.svd,
            scaler_user=self.scaler_user,
            defaults=self.defaults,
            analyzer=self.analyzer,
        )

        n_items = self.item_matrix.shape[0]
        batch_user = np.repeat(
            u_vec.to_numpy(dtype=np.float32), repeats=n_items, axis=0
        )
        scores = self.model.predict(
            [batch_user, self.item_matrix], batch_size=4096, verbose=0
        ).reshape(-1)
        scores_raw = self.scaler_target.inverse_transform(
            scores.reshape(-1, 1)
        ).ravel()

        ranked = np.argsort(-scores_raw)[:top_k]

        results = []
        for idx in ranked:
            product_id = str(self.item_ids[idx])
            meta = (
                self.product_info.loc[product_id]
                if product_id in self.product_info.index
                else None
            )
            results.append(
                ProductRecommendation(
                    product_id=product_id,
                    product_name=meta["product_name"] if meta is not None else None,
                    brand_name=meta["brand_name"] if meta is not None else None,
                    price_usd=meta["price_usd"] if meta is not None else None,
                    primary_category=(
                        meta["primary_category"] if meta is not None else None
                    ),
                    pred_score=float(scores_raw[idx]),
                )
            )
        return results
