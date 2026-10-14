"""Small synthetic interactions / catalog shared by the tests."""

import numpy as np
import pandas as pd

from src.config import load_config

WORDS = "soft smooth love hate dry oily cheap pricey great awful glow matte scent".split()


def make_interactions(n=400, n_users=40, n_items=15, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 60, n), unit="D")  # many ties
    df = pd.DataFrame(
        {
            "author_id": rng.integers(0, n_users, n).astype(str),
            "product_id": [f"P{i}" for i in rng.integers(0, n_items, n)],
            "rating": rng.integers(1, 6, n),
            "is_recommended": rng.choice([0.0, 1.0, np.nan], n),
            "helpfulness": rng.random(n),
            "total_feedback_count": rng.integers(0, 5, n),
            "submission_time": times,
            "review_text": [
                " ".join(rng.choice(WORDS, rng.integers(0, 6))) or None for _ in range(n)
            ],
            "skin_tone": rng.choice(["fair", "dark", None], n),
            "skin_type": rng.choice(["dry", "oily", None], n),
            "eye_color": rng.choice(["blue", "brown", None], n),
            "hair_color": rng.choice(["black", "blonde", None], n),
        }
    )
    df = df.drop_duplicates(["author_id", "product_id"])
    return df.sort_values("submission_time", kind="stable").reset_index(drop=True)


def make_product_info(n_items=15) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "product_id": [f"P{i}" for i in range(n_items)],
            "brand_id": rng.integers(0, 4, n_items),
            "size": rng.choice(["30 ml", "1.7 oz", "50 g", None], n_items),
            "price_usd": rng.random(n_items) * 80,
            "child_count": rng.integers(0, 5, n_items),
            "child_max_price": rng.random(n_items) * 90,
            "child_min_price": rng.random(n_items) * 40,
            "limited_edition": rng.integers(0, 2, n_items),
            "new": rng.integers(0, 2, n_items),
            "online_only": rng.integers(0, 2, n_items),
            "sephora_exclusive": rng.integers(0, 2, n_items),
            "primary_category": rng.choice(["Skincare", "Makeup"], n_items),
            "variation_type": rng.choice(["Size", "Color", None], n_items),
            # snapshot columns that must NOT reach the model
            "rating": rng.random(n_items) * 5,
            "reviews": rng.integers(1, 100, n_items),
            "loves_count": rng.integers(1, 1000, n_items),
            "out_of_stock": rng.integers(0, 2, n_items),
        }
    )


def small_cfg():
    return load_config(overrides={"features": {"svd_components": 4, "tfidf_min_df": 1, "tfidf_max_df": 1.0,
                                               "tfidf_max_features": 50}})
