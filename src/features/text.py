"""TF-IDF + truncated SVD review-text embeddings (fit on train text only)."""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_text_model(train_texts: pd.Series, *, max_features: int, min_df, max_df, n_components: int, seed: int):
    """Fit TF-IDF then SVD on the given (train) texts; returns (tfidf, svd)."""
    texts = train_texts.fillna("")
    tfidf = TfidfVectorizer(max_features=max_features, min_df=min_df, max_df=max_df)
    x = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    svd.fit(x)
    return tfidf, svd


def has_text(texts: pd.Series) -> np.ndarray:
    return texts.fillna("").str.strip().ne("").to_numpy()


def embed_texts(tfidf, svd, texts: pd.Series) -> np.ndarray:
    """(n, n_components) float32 embeddings; rows with no text are all zeros."""
    emb = svd.transform(tfidf.transform(texts.fillna(""))).astype(np.float32)
    emb[~has_text(texts)] = 0.0
    return emb
