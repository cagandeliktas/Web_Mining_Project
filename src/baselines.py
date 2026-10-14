"""Non-neural rating baselines the two-tower model has to beat.

Static baselines (global / item / user / bias / kNN) are fitted on the train
rows only and then frozen - the textbook setup. ``asof_additive`` is the fair
counterpart to the neural model: it sees exactly the same strictly-earlier
history the as-of features do. All predictions are clipped to [1, 5].
"""

import numpy as np
import pandas as pd
from scipy import sparse

from src.features.asof import asof_prior_sums

LO, HI = 1.0, 5.0


def _clip(x) -> np.ndarray:
    return np.clip(np.asarray(x, dtype="float64"), LO, HI)


class GlobalMean:
    name = "global mean"

    def fit(self, train: pd.DataFrame):
        self.mu = float(train["rating"].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.mu)


class _ShrunkMean:
    """Mean rating per key, shrunk toward the global mean with ``k`` pseudo-ratings."""

    key = ""

    def __init__(self, k: float):
        self.k = k

    def fit(self, train: pd.DataFrame):
        self.mu = float(train["rating"].mean())
        g = train.groupby(self.key)["rating"].agg(["sum", "count"])
        self.table = (g["sum"] + self.k * self.mu) / (g["count"] + self.k)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return _clip(df[self.key].map(self.table).fillna(self.mu))


class ItemMean(_ShrunkMean):
    name = "item mean"
    key = "product_id"


class UserMean(_ShrunkMean):
    name = "user mean"
    key = "author_id"


class BiasBaseline:
    """Koren's baseline: mu + b_u + b_i, fitted by alternating regularised least squares."""

    name = "user+item bias"

    def __init__(self, lam_user: float = 15.0, lam_item: float = 10.0, iters: int = 10):
        self.lam_user, self.lam_item, self.iters = lam_user, lam_item, iters

    def fit(self, train: pd.DataFrame):
        self.mu = float(train["rating"].mean())
        u, u_keys = pd.factorize(train["author_id"])
        i, i_keys = pd.factorize(train["product_id"])
        r = train["rating"].to_numpy(dtype="float64") - self.mu
        n_u, n_i = np.bincount(u), np.bincount(i)
        b_u, b_i = np.zeros(len(u_keys)), np.zeros(len(i_keys))
        for _ in range(self.iters):
            b_u = np.bincount(u, weights=r - b_i[i]) / (self.lam_user + n_u)
            b_i = np.bincount(i, weights=r - b_u[u]) / (self.lam_item + n_i)
        self.b_u = pd.Series(b_u, index=u_keys)
        self.b_i = pd.Series(b_i, index=i_keys)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        bu = df["author_id"].map(self.b_u).fillna(0.0).to_numpy()
        bi = df["product_id"].map(self.b_i).fillna(0.0).to_numpy()
        return _clip(self.mu + bu + bi)


def asof_additive(df: pd.DataFrame, mu: float, k_user: float, k_item: float) -> np.ndarray:
    """mu + shrunk user deviation + shrunk item deviation, from strictly earlier rows only.

    ``df`` must carry author_id, product_id, rating, submission_time for *all*
    rows (every split), exactly like the neural model's as-of features.
    """
    vals = pd.DataFrame({"n": 1.0, "s": df["rating"].to_numpy(dtype="float64") - mu}, index=df.index)
    pu = asof_prior_sums(df["author_id"], df["submission_time"], vals)
    pi = asof_prior_sums(df["product_id"], df["submission_time"], vals)
    pred = mu + pu["s"] / (pu["n"] + k_user) + pi["s"] / (pi["n"] + k_item)
    return _clip(pred.to_numpy())


class ItemKNN:
    """Item-based kNN with means (the idea behind surprise's KNNWithMeans).

    Similarity is the cosine of item-mean-centred rating columns; a prediction is the item's mean plus the
    similarity-weighted mean deviation of the user's ``k`` most similar rated items (positive similarities only).
    Similarities are shrunk by co-rater support. Falls back to the item mean (then the global mean) when the user or item is unseen in train.
    """

    name = "item kNN"

    def __init__(self, k: int = 40, sim_shrinkage: float = 50.0, den_shrinkage: float = 0.0, mean_shrinkage: float = 10.0):
        self.k = k
        self.mean_shrinkage = mean_shrinkage  # shrinks item means toward the global mean (noisy for rarely rated items)
        self.den_shrinkage = den_shrinkage  # adds to sum(sim) in the denominator: weak neighbours move the estimate less
        self.sim_shrinkage = sim_shrinkage  # sim *= n_co / (n_co + shrinkage): distrusts pairs with few co-raters

    def fit(self, train: pd.DataFrame):
        self.mu = float(train["rating"].mean())
        u, u_keys = pd.factorize(train["author_id"])
        self.user_codes = pd.Series(np.arange(len(u_keys)), index=u_keys)
        i, i_keys = pd.factorize(train["product_id"])
        self.item_codes = pd.Series(np.arange(len(i_keys)), index=i_keys)
        r = train["rating"].to_numpy(dtype="float64")
        n_i = np.bincount(i)
        raw_mean = np.bincount(i, weights=r) / n_i
        self.item_mean = (raw_mean * n_i + self.mean_shrinkage * self.mu) / (n_i + self.mean_shrinkage)
        dev = r - raw_mean[i]  # neighbours' deviations use their own raw means
        centred = sparse.csr_matrix((dev, (u, i)), shape=(len(u_keys), len(i_keys)))
        gram = (centred.T @ centred).toarray()
        norm = np.sqrt(np.diag(gram))
        norm[norm == 0] = 1.0
        support = centred.copy()
        support.data[:] = 1.0
        n_co = (support.T @ support).toarray()
        sim = gram / np.outer(norm, norm) * (n_co / (n_co + self.sim_shrinkage))
        self.sim = sim.astype("float32")
        np.fill_diagonal(self.sim, 0.0)
        self.user_rows = centred  # rows: user -> deviations of the items they rated
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        uc = df["author_id"].map(self.user_codes).to_numpy()
        ic = df["product_id"].map(self.item_codes).to_numpy()
        out = np.full(len(df), self.mu, dtype="float64")
        indptr, indices, data = self.user_rows.indptr, self.user_rows.indices, self.user_rows.data
        for n in range(len(df)):
            i = ic[n]
            if np.isnan(i):
                continue
            i = int(i)
            base = self.item_mean[i]
            u = uc[n]
            if not np.isnan(u):
                u = int(u)
                items, devs = indices[indptr[u] : indptr[u + 1]], data[indptr[u] : indptr[u + 1]]
                s = self.sim[i, items]
                pos = s > 0
                if pos.any():
                    s, devs = s[pos], devs[pos]
                    if len(s) > self.k:
                        top = np.argpartition(-s, self.k)[: self.k]
                        s, devs = s[top], devs[top]
                    base = base + float((s * devs).sum() / (s.sum() + self.den_shrinkage))
            out[n] = base
        return _clip(out)
