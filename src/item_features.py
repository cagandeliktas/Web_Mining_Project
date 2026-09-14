"""Build the item-side feature matrix for the two-tower model.

Ported from the item-feature-engineering cells in
NeuralNetworkApproach/NN_w_textFeatures.ipynb (raw product_info.csv ->
one-hot-encoded, cleaned item feature table). Kept here as a plain,
importable function so it can be unit-tested without a live TensorFlow
model or the full Kaggle dataset, and reused by both the one-time
scripts/build_item_matrix.py build step and any future retraining code.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.preprocessing import normalize_size

_COLUMNS_TO_DROP = [
    "product_name",
    "brand_name",
    "value_price_usd",
    "sale_price_usd",
    "variation_value",
    "variation_desc",
    "tertiary_category",
    "secondary_category",
    "highlights",
    "ingredients",
]
_COLUMNS_TO_IMPUTE_MEDIAN = ["rating", "child_max_price", "child_min_price"]


def build_item_feature_table(product_info: pd.DataFrame) -> pd.DataFrame:
    """Turn raw product_info.csv rows into the model's 32-column item feature table.

    Parameters
    ----------
    product_info : DataFrame
        Raw product_info.csv, one row per product.

    Returns
    -------
    DataFrame
        One row per product, with columns ``product_id`` followed by the
        32 numeric/one-hot feature columns the two-tower model's item
        tower was trained on, in the same order pandas produces them in
        (this matches the notebook's pipeline exactly, since it's the
        same sequence of operations on the same input).
    """
    df = product_info.drop(
        columns=[c for c in _COLUMNS_TO_DROP if c in product_info.columns]
    ).copy()

    brand_freq = df["brand_id"].value_counts()
    df["brand_freq"] = df["brand_id"].map(brand_freq)
    df["brand_freq_log"] = np.log1p(df["brand_freq"])
    df["brand_freq_scaled"] = MinMaxScaler().fit_transform(df[["brand_freq_log"]])

    for col in _COLUMNS_TO_IMPUTE_MEDIAN:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())
    df["reviews"] = pd.to_numeric(df["reviews"], errors="coerce").fillna(0)

    df["primary_category"] = df["primary_category"].fillna("Unknown")
    df["variation_type"] = df["variation_type"].fillna("Unknown")
    df = pd.get_dummies(
        df, columns=["variation_type", "primary_category"], prefix=["var_type", "pc"]
    )

    df["size_ml"] = df["size"].apply(normalize_size)
    df["size_ml_log"] = np.log1p(df["size_ml"])
    df["size_missing"] = df["size"].isna().astype(int)
    median_size_log = df["size_ml_log"].median()
    df["size_ml_log"] = df["size_ml_log"].fillna(median_size_log)
    df = df.drop(columns=["size_ml", "size"])

    df = df.astype({c: "int" for c in df.columns if df[c].dtype == bool})
    df = df.drop(columns=["brand_id", "brand_freq", "brand_freq_log"])

    return df
