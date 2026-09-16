"""A/B test: surprise-based KNNWithMeans (variant A) vs the two-tower NN
(variant B), Basic_Recommender_System.ipynb vs NN_w_textFeatures.ipynb.

Uses a leave-one-out evaluation: for every user with at least 2 ratings,
hold out exactly one (product, rating) pair and predict it from everything
else that user rated. Both variants are scored on the *same* held-out
instances - that pairing is what makes the significance test in
src/ab_testing.py meaningful (a paired test, not two independent samples).

Needs the raw Kaggle review CSVs locally (not committed - see README) plus
the artifacts already built for the FastAPI service (scaler_user.pkl,
item_matrix.pkl, etc. - see scripts/build_item_matrix.py, which must be
run first).

Usage:
    python scripts/run_ab_test.py --data-dir /path/to/sephora_datasets
"""

import argparse
import json
import random

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from surprise import Dataset, KNNWithMeans, Reader
from surprise.model_selection import LeaveOneOut
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.ab_testing import compare_variants, prediction_errors
from src.user_features import build_user_vector

REVIEW_FILES = [
    "reviews_0-250.csv",
    "reviews_250-500.csv",
    "reviews_500-750.csv",
    "reviews_750-1250.csv",
    "reviews_1250-end.csv",
]

SEED = 42


def _l2_normalize(x):
    return tf.math.l2_normalize(x, axis=1)


def _to_float32(t):
    return tf.cast(t, tf.float32)


def load_reviews(data_dir: str) -> pd.DataFrame:
    """Raw reviews, cleaned the same way as dbt_project/models/staging/stg_reviews.sql."""
    df = pd.concat(
        [pd.read_csv(f"{data_dir}/{f}", low_memory=False) for f in REVIEW_FILES],
        ignore_index=True,
    )
    df["author_id"] = df["author_id"].astype(str)
    df = df.loc[~df["author_id"].str.contains(r"[a-zA-Z]", regex=True, na=False), :]

    df = df.sort_values("total_feedback_count", ascending=False, na_position="last")
    df = df.drop_duplicates(
        subset=["author_id", "product_id", "submission_time", "review_text"],
        keep="first",
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument(
        "--model-path", default="NeuralNetworkApproach/final_two_tower_model_last.keras"
    )
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--out", default="ab_test_results.json")
    args = parser.parse_args()

    print("Loading raw reviews...")
    reviews = load_reviews(args.data_dir)
    print(f"{len(reviews)} cleaned review rows")

    reviews_by_author = {aid: g for aid, g in reviews.groupby("author_id")}

    df_aggregated = reviews.groupby(["author_id", "product_id"], as_index=False)[
        "rating"
    ].mean()

    print("Building leave-one-out split...")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        df_aggregated[["author_id", "product_id", "rating"]], reader
    )
    loo = LeaveOneOut(n_splits=1, min_n_ratings=2, random_state=SEED)
    trainset, testset = next(loo.split(data))
    print(f"{trainset.n_ratings} training ratings, {len(testset)} held-out instances")

    if len(testset) > args.sample_size:
        random.seed(SEED)
        testset = random.sample(testset, args.sample_size)
        print(f"Subsampled to {len(testset)} held-out instances")

    # --- Variant A: KNNWithMeans (same sim_options the notebook used) ---
    print("Training KNNWithMeans (variant A)...")
    sim_options = {"name": "pearson", "user_based": False, "min_support": 1}
    algo = KNNWithMeans(sim_options=sim_options)
    algo.fit(trainset)
    predictions_a = algo.test(testset)
    pred_a_lookup = {(p.uid, p.iid): p.est for p in predictions_a}

    # --- Variant B: two-tower NN ---
    print("Loading two-tower model + artifacts...")
    model = keras.models.load_model(
        args.model_path,
        custom_objects={"l2_normalize": _l2_normalize, "to_float32": _to_float32},
        compile=False,
        safe_mode=False,
    )
    scaler_user = joblib.load(f"{args.artifacts_dir}/scaler_user.pkl")
    tfidf = joblib.load(f"{args.artifacts_dir}/tfidf.pkl")
    svd = joblib.load(f"{args.artifacts_dir}/svd.pkl")
    defaults = joblib.load(f"{args.artifacts_dir}/defaults.pkl")
    user_cols = joblib.load(f"{args.artifacts_dir}/user_feature_cols.pkl")
    scaler_target = joblib.load(f"{args.artifacts_dir}/scaler_target.pkl")
    item_matrix = joblib.load(f"{args.artifacts_dir}/item_matrix.pkl")
    item_ids = item_matrix["product_ids"]
    item_mat_all = item_matrix["matrix"]
    item_id_to_row = {pid: i for i, pid in enumerate(item_ids)}
    analyzer = SentimentIntensityAnalyzer()

    review_cols = [
        "rating",
        "is_recommended",
        "helpfulness",
        "review_text",
        "skin_tone",
        "skin_type",
        "eye_color",
        "hair_color",
    ]

    print(f"Building user vectors for variant B ({len(testset)} instances)...")
    user_vecs, item_vecs, true_ratings, kept_pairs = [], [], [], []
    skipped = 0
    for uid, iid, r_ui in testset:
        if iid not in item_id_to_row:
            skipped += 1
            continue
        user_reviews = reviews_by_author.get(uid)
        if user_reviews is not None:
            history = user_reviews.loc[user_reviews["product_id"] != iid, review_cols]
        else:
            history = pd.DataFrame(columns=review_cols)

        u_vec = build_user_vector(
            history,
            user_cols=user_cols,
            tfidf=tfidf,
            svd=svd,
            scaler_user=scaler_user,
            defaults=defaults,
            analyzer=analyzer,
        )
        user_vecs.append(u_vec.to_numpy(dtype=np.float32)[0])
        item_vecs.append(item_mat_all[item_id_to_row[iid]])
        true_ratings.append(r_ui)
        kept_pairs.append((uid, iid))

    print(f"Scoring variant B ({len(user_vecs)} instances, {skipped} skipped)...")
    user_batch = np.array(user_vecs, dtype=np.float32)
    item_batch = np.array(item_vecs, dtype=np.float32)
    scores = model.predict([user_batch, item_batch], batch_size=256, verbose=0).reshape(-1)
    preds_b = scaler_target.inverse_transform(scores.reshape(-1, 1)).ravel()
    preds_b = np.clip(preds_b, 1.0, 5.0)

    true_ratings = np.array(true_ratings)
    preds_a = np.array([pred_a_lookup[pair] for pair in kept_pairs])

    errors_a = prediction_errors(true_ratings, preds_a)
    errors_b = prediction_errors(true_ratings, preds_b)

    print(f"\nVariant A (KNNWithMeans) MAE: {errors_a.mean():.4f}")
    print(f"Variant B (Two-Tower NN)  MAE: {errors_b.mean():.4f}")

    result = compare_variants(errors_a, errors_b)
    print("\n=== Significance test (paired, same held-out instances) ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
