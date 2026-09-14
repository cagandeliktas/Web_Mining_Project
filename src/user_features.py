"""Build a scaled user-feature vector for the two-tower model's user tower.

Ported from the `build_user_vector` cell in
NeuralNetworkApproach/NN_w_textFeatures.ipynb. The original notebook
version reads its fitted artifacts (tfidf, svd, scalers, defaults) from
notebook globals; here they're explicit parameters so the function is
testable and reusable outside the notebook (e.g. from the FastAPI
service), following the same pattern as src/similarity.py.
"""

import numpy as np
import pandas as pd

_REQUIRED_COLUMNS = [
    "rating",
    "is_recommended",
    "helpfulness",
    "review_text",
    "skin_tone",
    "skin_type",
    "eye_color",
    "hair_color",
]
_DEMOGRAPHIC_COLUMNS = ["skin_tone", "skin_type", "eye_color", "hair_color"]
_DEMOGRAPHIC_PREFIXES = ["tone", "type", "eye", "hair"]


def build_user_vector(
    raw_reviews: pd.DataFrame,
    *,
    user_cols: list,
    tfidf,
    svd,
    scaler_user,
    defaults: dict,
    analyzer,
) -> pd.DataFrame:
    """Build the full, scaled user-feature vector expected by the two-tower model.

    Parameters
    ----------
    raw_reviews : DataFrame
        One row per review written by the (new) user. Expected columns:
        ['rating', 'is_recommended', 'helpfulness', 'review_text',
         'skin_tone', 'skin_type', 'eye_color', 'hair_color'].
        May be empty (fully cold-start/anonymous user) or contain NaNs.
    user_cols : list
        Column order the model's user tower was trained on
        (``user_feature_cols.pkl``).
    tfidf, svd : fitted sklearn transformers
        The text pipeline used at training time (``tfidf.pkl``, ``svd.pkl``).
    scaler_user : fitted sklearn StandardScaler
        ``scaler_user.pkl``.
    defaults : dict
        Training-time fallback values, with keys ``rec_avg`` and
        ``text_emb_mean`` (``defaults.pkl``).
    analyzer : vaderSentiment.SentimentIntensityAnalyzer

    Returns
    -------
    DataFrame
        Single-row dataframe whose columns match ``user_cols`` and whose
        numeric columns are z-scaled with ``scaler_user``.
    """
    raw_reviews = raw_reviews.copy()

    if raw_reviews.empty:
        raw_reviews = pd.DataFrame(
            [
                {
                    "rating": np.nan,
                    "is_recommended": np.nan,
                    "helpfulness": np.nan,
                    "review_text": np.nan,
                    "skin_tone": "Unknown",
                    "skin_type": "Unknown",
                    "eye_color": "Unknown",
                    "hair_color": "Unknown",
                }
            ]
        )

    agg = raw_reviews.agg(
        {"rating": ["mean", "count"], "is_recommended": "mean", "helpfulness": "mean"}
    )
    row = {
        "rating_avg": agg.loc["mean", "rating"],
        "rating_count": agg.loc["count", "rating"],
        "recommend_ratio": agg.loc["mean", "is_recommended"],
        "helpfulness_avg": agg.loc["mean", "helpfulness"],
    }

    if pd.isna(row["recommend_ratio"]):
        row["recommend_missing"] = 1
        row["recommend_ratio"] = defaults["rec_avg"]
    else:
        row["recommend_missing"] = 0

    raw_reviews["sentiment"] = (
        raw_reviews["review_text"]
        .fillna("")
        .map(lambda t: analyzer.polarity_scores(t)["compound"])
    )
    row["sentiment"] = raw_reviews["sentiment"].mean()

    for c in _DEMOGRAPHIC_COLUMNS:
        raw_reviews[c] = raw_reviews[c].fillna("Unknown")

    demo = pd.get_dummies(
        raw_reviews[_DEMOGRAPHIC_COLUMNS].iloc[[0]],
        columns=_DEMOGRAPHIC_COLUMNS,
        prefix=_DEMOGRAPHIC_PREFIXES,
    )

    if raw_reviews["review_text"].notna().any():
        emb = svd.transform(tfidf.transform(raw_reviews["review_text"].fillna("")))
        text_vec = emb.mean(axis=0)
        text_missing = 0
    else:
        text_vec = defaults["text_emb_mean"]
        text_missing = 1

    row.update({f"user_text_emb_{i}": text_vec[i] for i in range(len(text_vec))})
    row["text_emb_missing"] = text_missing

    user_df = pd.DataFrame([row]).join(demo, how="left")
    user_df = user_df.reindex(columns=user_cols, fill_value=np.nan)

    num_means = dict(zip(scaler_user.feature_names_in_, scaler_user.mean_))
    for col in scaler_user.feature_names_in_:
        if col not in user_df.columns:
            user_df[col] = num_means[col]
        else:
            user_df[col] = user_df[col].fillna(num_means[col])

    user_df = user_df.fillna(0.0)

    cols_to_scale = list(scaler_user.feature_names_in_)
    user_df[cols_to_scale] = scaler_user.transform(user_df[cols_to_scale])

    bool_cols = user_df.columns[user_df.dtypes == bool]
    user_df[bool_cols] = user_df[bool_cols].astype(int)

    return user_df
