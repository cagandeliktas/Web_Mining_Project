"""Retrain the two-tower model on a leak-free split.

The original notebook (NeuralNetworkApproach/NN_w_textFeatures.ipynb) built
each user's feature profile (numeric aggregates, sentiment, demographics,
TF-IDF/SVD text embedding) from ALL of that user's reviews, THEN split rows
into train/test. That means a user's test-row profile included that same
test review's own rating/text/sentiment - the model could partly see the
answer baked into its own input. The TF-IDF vocabulary and SVD components
were also fit on the full corpus, including test-set review text.

This script splits at the review-row level FIRST, fits TF-IDF/SVD/scalers
on the train portion only, and builds every user's profile from their
train-side reviews only - a test row's profile has never seen that row.

Usage:
    pip install -r requirements-train.txt
    python scripts/train_leakfree_model.py --data-dir /path/to/sephora_datasets \
        [--sample-frac 1.0] [--search full|fixed] [--out-dir artifacts_v2]
"""

import argparse
import json
import os
import re
import time

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import keras_tuner as kt
from keras.layers import Dense, Dot, Input, Lambda
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.item_features import build_item_feature_table
from src.metrics import measures_at_k_nn, precision_recall_f1_nn_vectorized

REVIEW_FILES = [
    "reviews_0-250.csv",
    "reviews_250-500.csv",
    "reviews_500-750.csv",
    "reviews_750-1250.csv",
    "reviews_1250-end.csv",
]
SEED = 1


def load_reviews(data_dir: str) -> pd.DataFrame:
    df = pd.concat(
        [pd.read_csv(f"{data_dir}/{f}", low_memory=False) for f in REVIEW_FILES],
        ignore_index=True,
    )
    df["author_id"] = df["author_id"].astype(str)
    df = df.loc[~df["author_id"].str.contains(r"[a-zA-Z]", regex=True, na=False), :]
    df = df.groupby(["author_id", "product_id"], as_index=False).agg(
        {
            "rating": "mean",
            "is_recommended": "mean",
            "helpfulness": "mean",
            "total_feedback_count": "mean",
            "review_text": "first",
            "skin_tone": "first",
            "skin_type": "first",
            "eye_color": "first",
            "hair_color": "first",
        }
    )
    return df


def build_user_features_from(reviews: pd.DataFrame, tfidf, svd, analyzer) -> pd.DataFrame:
    """Aggregate a per-user profile from exactly the rows given (train-only, at fit time)."""
    reviews = reviews.copy()
    mask_no_feedback = reviews["total_feedback_count"].fillna(0) == 0
    reviews.loc[mask_no_feedback, "helpfulness"] = 0.0
    for col in ["skin_tone", "eye_color", "skin_type", "hair_color"]:
        reviews[col] = reviews[col].fillna("Unknown")

    user_agg = reviews.groupby("author_id").agg(
        {
            "rating": ["mean", "count"],
            "is_recommended": "mean",
            "helpfulness": "mean",
            "skin_tone": "first",
            "skin_type": "first",
            "eye_color": "first",
            "hair_color": "first",
        }
    )
    user_agg.columns = [
        "rating_avg",
        "rating_count",
        "recommend_ratio",
        "helpfulness_avg",
        "skin_tone",
        "skin_type",
        "eye_color",
        "hair_color",
    ]
    user_agg = user_agg.reset_index()

    user_features = pd.get_dummies(
        user_agg,
        columns=["skin_tone", "skin_type", "eye_color", "hair_color"],
        prefix=["tone", "type", "eye", "hair"],
    )
    global_recommend_avg = reviews["is_recommended"].mean()
    user_features["recommend_missing"] = user_features["recommend_ratio"].isna().astype(int)
    user_features["recommend_ratio"] = user_features["recommend_ratio"].fillna(
        global_recommend_avg
    )
    user_features = user_features.astype(
        {c: "int" for c in user_features.columns if user_features[c].dtype == "bool"}
    )

    reviews["sentiment"] = reviews["review_text"].fillna("").apply(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )
    user_sentiment = reviews.groupby("author_id", sort=False)["sentiment"].mean().reset_index()
    user_features = user_features.merge(user_sentiment, on="author_id", how="left")

    x_tfidf = tfidf.transform(reviews["review_text"].fillna(""))
    x_svd = svd.transform(x_tfidf)
    embeddings = pd.DataFrame(x_svd)
    embeddings["author_id"] = reviews["author_id"].values
    user_text = embeddings.groupby("author_id").mean().reset_index()
    user_text.columns = ["author_id"] + [f"user_text_emb_{i}" for i in range(x_svd.shape[1])]

    user_features = user_features.merge(user_text, on="author_id", how="left")
    text_cols = [c for c in user_features.columns if c.startswith("user_text_emb_")]
    user_features["text_emb_missing"] = user_features[text_cols[0]].isna().astype(int)
    user_features[text_cols] = user_features[text_cols].fillna(user_features[text_cols].mean())

    return user_features


def l2_normalize(x):
    return tf.math.l2_normalize(x, axis=1)


def to_float32(t):
    return tf.cast(t, tf.float32)


def build_two_tower(hp, num_user_features, num_item_features):
    embed_dim = hp.Int("embedding_dim", 16, 128, step=16, default=32)

    def tower(prefix, n_features):
        seq = keras.Sequential(name=f"{prefix}_tower")
        units1 = hp.Int(f"{prefix}_dense1", 64, 512, step=64, default=256)
        units2 = hp.Int(f"{prefix}_dense2", 32, 256, step=32, default=128)
        seq.add(Dense(units1, activation="relu"))
        seq.add(Dense(units2, activation="relu"))
        seq.add(Dense(embed_dim))
        return seq

    user_in = Input((num_user_features,), name="user_in")
    item_in = Input((num_item_features,), name="item_in")
    vu = tower("u", num_user_features)(user_in)
    vm = tower("i", num_item_features)(item_in)
    vu = Lambda(l2_normalize)(vu)
    vm = Lambda(l2_normalize)(vm)
    dot = Dot(axes=1)([vu, vm])
    out = Lambda(to_float32)(dot)

    model = keras.Model([user_in, item_in], out)
    lr = hp.Float("lr", 1e-4, 1e-2, sampling="log", default=1e-3)
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--product-info", default="artifacts/product_info.csv")
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--search", choices=["full", "fixed", "none"], default="fixed")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--out-dir", default="artifacts_v2")
    parser.add_argument(
        "--leaky",
        action="store_true",
        help=(
            "Deliberately reproduce the original leak (fit tfidf/svd/user "
            "profiles on ALL rows, train+test, instead of train-only) - for "
            "a controlled comparison isolating the leak as the only "
            "variable. Never use this for a real model."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    print("Loading + cleaning reviews...")
    reviews = load_reviews(args.data_dir)
    if args.sample_frac < 1.0:
        reviews = reviews.sample(frac=args.sample_frac, random_state=SEED).reset_index(drop=True)
    print(f"{len(reviews)} (author_id, product_id) review rows")

    print("Splitting at the review-row level FIRST...")
    train_idx, test_idx = train_test_split(
        np.arange(len(reviews)), train_size=0.8, random_state=SEED, shuffle=True
    )
    reviews_train_rows = reviews.iloc[train_idx]

    # fit_rows is the ONLY thing that changes between the leak-free and
    # deliberately-leaky modes - same data loading, same split, same
    # architecture, same everything else. Isolates the leak as the one
    # variable under test.
    fit_rows = reviews if args.leaky else reviews_train_rows
    if args.leaky:
        print("*** --leaky: fitting on ALL rows (train+test), reproducing the original bug ***")

    print(f"Fitting TF-IDF + SVD on {'ALL' if args.leaky else 'TRAIN-only'} review text...")
    tfidf = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8)
    tfidf.fit(fit_rows["review_text"].fillna(""))
    x_tfidf_train = tfidf.transform(fit_rows["review_text"].fillna(""))
    svd = TruncatedSVD(n_components=128, random_state=SEED)
    svd.fit(x_tfidf_train)

    analyzer = SentimentIntensityAnalyzer()
    print(f"Building user profiles from {'ALL' if args.leaky else 'TRAIN-only'} reviews...")
    user_features = build_user_features_from(fit_rows, tfidf, svd, analyzer)
    user_cols_full = user_features.drop(columns=["author_id"]).columns.tolist()

    text_cols = [c for c in user_features.columns if c.startswith("user_text_emb_")]
    defaults = {
        "rec_avg": fit_rows["is_recommended"].mean(),
        "text_emb_mean": user_features[text_cols].mean().values,
    }

    print("Building item features (product-level, no row-split leakage)...")
    pro_inf = pd.read_csv(args.product_info)
    item_table = build_item_feature_table(pro_inf)

    user_features_ordered = reviews[["author_id"]].merge(user_features, on="author_id", how="left")
    item_features_ordered = reviews[["product_id"]].merge(item_table, on="product_id", how="left")

    # numeric fill for users with zero train-side history (cold start w.r.t. training)
    num_means = user_features[
        [c for c in user_cols_full if c not in text_cols]
    ].mean(numeric_only=True)
    for col in user_cols_full:
        if col in text_cols:
            user_features_ordered[col] = user_features_ordered[col].fillna(defaults["text_emb_mean"][int(col.split("_")[-1])])
        elif col in num_means.index:
            user_features_ordered[col] = user_features_ordered[col].fillna(num_means[col])
        else:
            user_features_ordered[col] = user_features_ordered[col].fillna(0)

    user_train_df = user_features_ordered.iloc[train_idx].reset_index(drop=True)
    user_test_df = user_features_ordered.iloc[test_idx].reset_index(drop=True)
    item_train_df = item_features_ordered.iloc[train_idx].reset_index(drop=True)
    item_test_df = item_features_ordered.iloc[test_idx].reset_index(drop=True)
    y_train = reviews["rating"].to_numpy()[train_idx]
    y_test = reviews["rating"].to_numpy()[test_idx]
    user_test_ids = user_test_df["author_id"].values

    print("Fitting scalers on TRAIN only...")

    def detect_binary(df):
        return [c for c in df.columns[1:] if df[c].dropna().isin([0, 1]).all()]

    user_cols_to_scale = [c for c in user_train_df.columns[1:] if c not in detect_binary(user_train_df)]
    item_cols_to_scale = [c for c in item_train_df.columns[1:] if c not in detect_binary(item_train_df)]

    scaler_user = StandardScaler().fit(user_train_df[user_cols_to_scale])
    scaler_item = StandardScaler().fit(item_train_df[item_cols_to_scale])
    for _df in [user_train_df, user_test_df]:
        _df[user_cols_to_scale] = scaler_user.transform(_df[user_cols_to_scale])
    for _df in [item_train_df, item_test_df]:
        _df[item_cols_to_scale] = scaler_item.transform(_df[item_cols_to_scale])

    user_train = user_train_df.drop(columns="author_id").to_numpy(np.float32)
    user_test = user_test_df.drop(columns="author_id").to_numpy(np.float32)
    item_train = item_train_df.drop(columns="product_id").to_numpy(np.float32)
    item_test = item_test_df.drop(columns="product_id").to_numpy(np.float32)

    scaler_target = MinMaxScaler((-1, 1)).fit(y_train.reshape(-1, 1))
    y_train_scaled = scaler_target.transform(y_train.reshape(-1, 1)).astype(np.float32)
    y_test_scaled = scaler_target.transform(y_test.reshape(-1, 1)).astype(np.float32)

    num_user_features = user_train.shape[1]
    num_item_features = item_train.shape[1]
    print(f"num_user_features={num_user_features}, num_item_features={num_item_features}")
    print(f"train={len(user_train)}, test={len(user_test)}")
    print(f"Setup done in {time.time() - t0:.1f}s")

    user_tr, user_val, item_tr, item_val, y_tr, y_val = train_test_split(
        user_train, item_train, y_train_scaled, train_size=0.8, random_state=SEED, shuffle=True
    )

    stop = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

    if args.search == "none":
        print("Skipping training (setup/benchmark only).")
        return

    if args.search == "full":
        print("Running full Hyperband search...")
        tuner = kt.Hyperband(
            lambda hp: build_two_tower(hp, num_user_features, num_item_features),
            objective=kt.Objective("val_rmse", "min"),
            max_epochs=args.epochs,
            factor=3,
            directory=f"{args.out_dir}/kt_tuner",
            project_name="two_tower_leakfree",
            overwrite=True,
        )
        t_search = time.time()
        tuner.search(
            x=[user_tr, item_tr],
            y=y_tr,
            validation_data=([user_val, item_val], y_val),
            batch_size=256,
            callbacks=[stop],
            verbose=2,
        )
        print(f"Search took {time.time() - t_search:.1f}s")
        best_hp = tuner.get_best_hyperparameters(1)[0]
    else:
        print("Using the notebook's own best-found hyperparameters (no search)...")
        best_hp = kt.HyperParameters()
        best_hp.values = {
            "embedding_dim": 32,
            "u_dense1": 256,
            "u_dense2": 128,
            "i_dense1": 256,
            "i_dense2": 128,
            "lr": 1e-3,
        }

    print("Final fit on train+val...")
    final_model = build_two_tower(best_hp, num_user_features, num_item_features)
    t_fit = time.time()
    final_model.fit(
        [np.concatenate([user_tr, user_val]), np.concatenate([item_tr, item_val])],
        np.concatenate([y_tr, y_val]),
        epochs=args.epochs,
        batch_size=256,
        callbacks=[stop],
        verbose=2,
    )
    print(f"Final fit took {time.time() - t_fit:.1f}s")

    print("Evaluating on the untouched leak-free test split...")
    y_pred_scaled = final_model.predict([user_test, item_test], verbose=0)
    y_pred = scaler_target.inverse_transform(y_pred_scaled)
    y_test_orig = scaler_target.inverse_transform(y_test_scaled)
    rmse = float(np.sqrt(mean_squared_error(y_test_orig, y_pred)))
    mae = float(mean_absolute_error(y_test_orig, y_pred))
    prec, rec, f1, acc = precision_recall_f1_nn_vectorized(
        user_test_ids, y_test_orig, y_pred, threshold=3.5
    )
    map_, p10, ndcg10 = measures_at_k_nn(
        user_test_ids, y_test_orig, y_pred, k=10, threshold=3.5
    )
    print(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  Accuracy: {acc:.4f}")
    print(f"MAP: {map_:.4f}  Precision@10: {p10:.4f}  nDCG@10: {ndcg10:.4f}")

    print(f"Saving artifacts to {args.out_dir}/ ...")
    final_model.save(f"{args.out_dir}/final_two_tower_model_last.keras")
    joblib.dump(tfidf, f"{args.out_dir}/tfidf.pkl")
    joblib.dump(svd, f"{args.out_dir}/svd.pkl")
    joblib.dump(scaler_user, f"{args.out_dir}/scaler_user.pkl")
    joblib.dump(scaler_item, f"{args.out_dir}/scaler_item.pkl")
    joblib.dump(scaler_target, f"{args.out_dir}/scaler_target.pkl")
    joblib.dump(defaults, f"{args.out_dir}/defaults.pkl")
    joblib.dump(user_cols_full, f"{args.out_dir}/user_feature_cols.pkl")

    with open(f"{args.out_dir}/retrain_metrics.json", "w") as f:
        json.dump(
            {
                "rmse": rmse,
                "mae": mae,
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "accuracy": float(acc),
                "map": float(map_),
                "precision_at_10": float(p10),
                "ndcg_at_10": float(ndcg10),
                "n_test": len(user_test),
                "n_train": len(user_train),
                "search": args.search,
                "leaky": args.leaky,
                "best_hp": dict(best_hp.values),
            },
            f,
            indent=2,
        )
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
