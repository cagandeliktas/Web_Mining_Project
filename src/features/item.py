"""Item-side features: static catalog attributes + as-of interaction stats.

In plain words: Builds the product-side numbers: catalogue facts (price, size, brand size, flags)
plus how many reviews and what average rating the product had BEFORE each review.

Dropped on purpose: product_info's ``rating``, ``reviews``, ``loves_count``
and ``out_of_stock``. They are 2023 snapshots computed from all reviews
(including test-period ones), so they would leak the future. The as-of
popularity / mean-rating columns below replace them.
"""

import pandas as pd

from src.features.asof import asof_prior_sums
from src.preprocessing import normalize_size

STATIC_BINARY = ["limited_edition", "new", "online_only", "sephora_exclusive"]
STATIC_NUMERIC = ["price_usd", "child_count", "child_max_price", "child_min_price", "size_ml", "brand_freq"]
STATIC_CATEGORICAL = ["primary_category", "variation_type"]


def build_item_static(product_info: pd.DataFrame) -> pd.DataFrame:
    """One row per product, indexed by product_id; raw values (scaling happens later)."""
    info = product_info.copy()
    info["product_id"] = info["product_id"].astype(str)
    info = info.drop_duplicates("product_id").set_index("product_id")

    out = pd.DataFrame(index=info.index)
    for c in STATIC_BINARY:
        out[c] = pd.to_numeric(info[c], errors="coerce").fillna(0.0)
    for c in ["price_usd", "child_count", "child_max_price", "child_min_price"]:
        out[c] = pd.to_numeric(info[c], errors="coerce")
    out["size_ml"] = info["size"].map(normalize_size)
    # catalog breadth of the brand: label-free, so safe to compute on the whole catalog
    out["brand_freq"] = info["brand_id"].map(info["brand_id"].value_counts()).astype(float)
    for c in STATIC_CATEGORICAL:
        out[c] = info[c]
    return out


def shrunk_mean(total, n, prior_mean: float, shrinkage: float):
    """(sum + k*prior) / (n + k): pulls sparse items toward the train mean."""
    return (total + shrinkage * prior_mean) / (n + shrinkage)


def build_item_asof(df: pd.DataFrame, prior_mean: float, shrinkage: float) -> pd.DataFrame:
    """As-of popularity and shrunk mean rating for each row's product.

    ``prior_mean`` must come from train only. An item with no earlier ratings
    gets exactly ``prior_mean`` and a count of 0.
    """
    values = pd.DataFrame({"n": 1.0, "rating": df["rating"].to_numpy()}, index=df.index)
    prior = asof_prior_sums(df["product_id"], df["submission_time"], values)
    out = pd.DataFrame(index=df.index)
    out["i_hist_count"] = prior["n"]
    out["i_rating_mean"] = shrunk_mean(prior["rating"], prior["n"], prior_mean, shrinkage)
    return out


def join_item_static(df: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    """Per-interaction copy of the static attributes (NaN for products missing from the catalog)."""
    joined = static.reindex(df["product_id"].to_numpy())
    joined.index = df.index
    return joined
