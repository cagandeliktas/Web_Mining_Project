"""One-time build step: produce artifacts/item_matrix.pkl for the FastAPI service.

Needs the raw Kaggle dataset locally (product_info.csv plus the five
reviews_*.csv files - see README.md for the download link) to replicate the
exact review-filtered product catalog and column order the two-tower model
was trained on. The reviews CSVs are only needed here, once, to work out
which products have at least one review; the resulting artifact is what the
API actually loads at request time, so reviewers of this repo don't need the
(very large, gitignored) raw reviews files themselves.

Item features are scaled with the same StandardScaler used at training time
(artifacts/scaler_item.pkl), matching how the model was actually trained and
evaluated. This intentionally differs from the notebook's own "example
recommendation" cells, which fed the model unscaled item features by
mistake - see README.md for the full explanation.

Usage:
    python scripts/build_item_matrix.py --data-dir /path/to/sephora_datasets
"""

import argparse

import joblib
import numpy as np
import pandas as pd

from src.item_features import build_item_feature_table

REVIEW_FILES = [
    "reviews_0-250.csv",
    "reviews_250-500.csv",
    "reviews_500-750.csv",
    "reviews_750-1250.csv",
    "reviews_1250-end.csv",
]


def reviewed_product_ids(data_dir: str) -> pd.DataFrame:
    """Replicate df_reviews_final's product_id universe from the notebook."""
    df_reviews = pd.concat(
        [pd.read_csv(f"{data_dir}/{f}") for f in REVIEW_FILES], ignore_index=True
    )
    df_reviews_final = df_reviews[["product_id", "author_id", "rating"]].copy()
    df_reviews_final = df_reviews_final.loc[
        ~df_reviews_final["author_id"]
        .astype(str)
        .str.contains(r"[a-zA-Z]", regex=True, na=False),
        :,
    ]
    df_reviews_final = df_reviews_final.groupby(
        ["author_id", "product_id"], as_index=False
    )["rating"].mean()
    return df_reviews_final[["product_id"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing the raw reviews_*.csv files",
    )
    parser.add_argument(
        "--product-info",
        default="artifacts/product_info.csv",
        help="Path to product_info.csv",
    )
    parser.add_argument("--scaler-item", default="artifacts/scaler_item.pkl")
    parser.add_argument("--out", default="artifacts/item_matrix.pkl")
    args = parser.parse_args()

    pro_inf = pd.read_csv(args.product_info)
    item_table = build_item_feature_table(pro_inf)

    catalog = (
        reviewed_product_ids(args.data_dir)
        .merge(item_table, on="product_id", how="left")
        .drop_duplicates(keep="first")
    )

    scaler_item = joblib.load(args.scaler_item)
    cols_to_scale = list(scaler_item.feature_names_in_)

    item_ids = catalog["product_id"].to_numpy()
    item_numeric = catalog.drop(columns=["product_id"])
    scaled = scaler_item.transform(item_numeric[cols_to_scale])
    scaled_df = pd.DataFrame(scaled, columns=cols_to_scale, index=item_numeric.index)
    unchanged_cols = [c for c in item_numeric.columns if c not in cols_to_scale]
    item_matrix_df = pd.concat([scaled_df, item_numeric[unchanged_cols]], axis=1)[
        item_numeric.columns
    ]

    joblib.dump(
        {
            "product_ids": item_ids,
            "matrix": item_matrix_df.to_numpy(dtype=np.float32),
            "columns": list(item_matrix_df.columns),
        },
        args.out,
    )
    print(f"Wrote {args.out}: {len(item_ids)} products x {item_matrix_df.shape[1]} features")


if __name__ == "__main__":
    main()
