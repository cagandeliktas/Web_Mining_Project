"""Full-catalog ranking evaluation.

In plain words: Tests how well each method finds the product a user really reviewed among the whole
catalogue (the "which product next?" question), rather than predicting a rating.

Question: at the moment a user posts a review they liked, how high does a
method rank *that product* among every product they could have picked?

Protocol (identical for every method):
- targets are held-out interactions rated >= ``threshold`` ("liked");
- candidates are all catalog products first reviewed on or before that day,
  minus products the user had already reviewed (strictly earlier);
- item-side features are rebuilt *as of that day* (popularity / mean rating
  from strictly earlier reviews), so nothing from the future is used;
- ties count half (average rank), so degenerate all-equal scores get no credit.

A rating predictor ranks by predicted rating; popularity ranks by prior
review count. They answer different questions, which is the point of showing
both.
"""

import numpy as np
import pandas as pd
import torch

from src.features.item import join_item_static, shrunk_mean


def rank_of_target(scores: np.ndarray, target: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """1-based average rank of ``target`` among valid candidates, per row.

    scores, valid: (n_rows, n_items); target: (n_rows,) column indices (must be valid).
    """
    rows = np.arange(len(target))
    t = scores[rows, target][:, None]
    greater = ((scores > t) & valid).sum(1)
    ties = ((scores == t) & valid).sum(1) - 1  # exclude the target itself
    return 1.0 + greater + 0.5 * ties


def ranking_metrics(ranks: np.ndarray, ks) -> dict:
    """Turn the rank of the true product into: HR@k (was it in the top k?), NDCG@k (higher credit for
    higher ranks) and MRR (average 1/rank). Higher is better.
    """
    out = {"mrr": float(np.mean(1.0 / ranks))}
    for k in ks:
        hit = ranks <= k
        out[f"hr@{k}"] = float(hit.mean())
        out[f"ndcg@{k}"] = float(np.mean(np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0)))
    return out


class CatalogSnapshots:
    """Per-day item state (prior review count / rating sum / launched?) from strictly earlier reviews."""

    def __init__(self, inter: pd.DataFrame):
        """Sort all reviews by time and remember when each product was first reviewed."""
        order = np.argsort(inter["submission_time"].to_numpy(), kind="stable")
        self.t = inter["submission_time"].to_numpy()[order]
        self.items = np.array(sorted(inter["product_id"].unique()))
        self.pos = pd.Series(np.arange(len(self.items)), index=self.items)
        self.j = self.pos.reindex(inter["product_id"]).to_numpy()[order]
        self.r = inter["rating"].to_numpy(dtype="float64")[order]
        self.first_seen = (
            pd.Series(inter["submission_time"].to_numpy()).groupby(inter["product_id"].to_numpy()).min().reindex(self.items)
        ).to_numpy()

    def iter_days(self, days):
        """Yield (day, counts, sums, launched) for ascending ``days``; state covers rows with time < day."""
        counts = np.zeros(len(self.items))
        sums = np.zeros(len(self.items))
        ptr = 0
        for day in sorted(days):
            end = np.searchsorted(self.t, day, side="left")
            if end > ptr:
                np.add.at(counts, self.j[ptr:end], 1.0)
                np.add.at(sums, self.j[ptr:end], self.r[ptr:end])
                ptr = end
            yield day, counts.copy(), sums.copy(), self.first_seen <= day


def item_frame_at(snap_items, counts, sums, static: pd.DataFrame, mu: float, shrinkage: float) -> pd.DataFrame:
    """Raw item feature frame for every catalog item as of one day (same columns build_features produces)."""
    frame = pd.DataFrame({"product_id": snap_items})
    frame["i_hist_count"] = counts
    frame["i_rating_mean"] = shrunk_mean(sums, counts, mu, shrinkage)
    return pd.concat([frame, join_item_static(frame, static)], axis=1)


def evaluate_ranking(
    fs, inter: pd.DataFrame, static: pd.DataFrame, model, split: str, *, threshold: float, ks, n_sample: int,
    shrinkage: float, seed: int = 0,
) -> dict:
    """Rank liked held-out interactions of ``split`` against the catalog; returns per-row ranks per method.

    ``inter`` = author_id, product_id, rating, submission_time for ALL rows, aligned to ``fs`` row order.
    """
    rng = np.random.default_rng(seed)
    cand = np.where((fs.split == split) & (fs.y >= threshold))[0]
    rows = np.sort(rng.choice(cand, size=min(n_sample, len(cand)), replace=False))

    snaps = CatalogSnapshots(inter)
    n_items = len(snaps.items)
    target = snaps.pos.reindex(inter["product_id"].iloc[rows]).to_numpy()
    days = inter["submission_time"].iloc[rows].to_numpy()

    # products each sampled user had already reviewed, strictly before the target day
    sub = inter[inter["author_id"].isin(set(inter["author_id"].iloc[rows]))]
    by_author = {a: (g["submission_time"].to_numpy(), snaps.pos.reindex(g["product_id"]).to_numpy())
                 for a, g in sub.groupby("author_id")}

    item_idx = torch.from_numpy(fs.artifacts["item_vocab"].transform(pd.Series(snaps.items)).copy())
    item_tf = fs.artifacts["item_tf"]
    model = model.cpu().eval()
    with torch.no_grad():
        user_emb = model.user_tower(torch.from_numpy(fs.X_user[rows])).numpy()
        item_bias = model.item_bias(item_idx).squeeze(-1).numpy() if model.use_biases else np.zeros(n_items)

    ranks = {"random (expected)": None, "popularity": np.zeros(len(rows)), "item mean rating": np.zeros(len(rows)),
             "two-tower": np.zeros(len(rows))}
    n_valid = np.zeros(len(rows))
    for day, counts, sums, launched in snaps.iter_days(np.unique(days)):
        sel = np.where(days == day)[0]
        valid = np.tile(launched, (len(sel), 1))
        for r, k in enumerate(sel):
            times, pos = by_author.get(inter["author_id"].iloc[rows[k]], (np.array([]), np.array([], dtype=int)))
            valid[r, pos[times < day].astype(int)] = False
        valid[np.arange(len(sel)), target[sel].astype(int)] = True
        n_valid[sel] = valid.sum(1)

        frame = item_frame_at(snaps.items, counts, sums, static, fs.mu, shrinkage)
        with torch.no_grad():
            v = model.item_tower(torch.from_numpy(item_tf.transform(frame))).numpy()
        model_scores = user_emb[sel] @ v.T + item_bias[None, :]
        tgt = target[sel].astype(int)
        ranks["popularity"][sel] = rank_of_target(np.tile(counts, (len(sel), 1)), tgt, valid)
        ranks["item mean rating"][sel] = rank_of_target(np.tile(frame["i_rating_mean"].to_numpy(), (len(sel), 1)), tgt, valid)
        ranks["two-tower"][sel] = rank_of_target(model_scores, tgt, valid)

    # random ranking: the target's rank is uniform on 1..m, so expectations are closed-form
    ranks.pop("random (expected)")
    out = {name: ranking_metrics(r, ks) for name, r in ranks.items()}
    out["random (expected)"] = _random_expectation(n_valid.astype(int), ks)
    out["_ranks"] = ranks
    out["_rows"] = rows
    out["_mean_candidates"] = float(n_valid.mean())
    return out


def _random_expectation(m: np.ndarray, ks) -> dict:
    """Expected MRR / HR@k / NDCG@k when the target lands uniformly among ``m`` candidates (per row)."""
    harmonic = np.cumsum(1.0 / np.arange(1, m.max() + 1))
    gain = np.cumsum(1.0 / np.log2(np.arange(1, m.max() + 1) + 1.0))
    out = {"mrr": float(np.mean(harmonic[m - 1] / m))}
    for k in ks:
        kk = np.minimum(k, m)
        out[f"hr@{k}"] = float(np.mean(kk / m))
        out[f"ndcg@{k}"] = float(np.mean(gain[kk - 1] / m))
    return out


def catalog_embeddings(fs, inter: pd.DataFrame, static: pd.DataFrame, model, shrinkage: float, day=None):
    """(product_ids, item embeddings) for the whole catalog as of ``day`` (default: after the last review)."""
    snaps = CatalogSnapshots(inter)
    day = inter["submission_time"].max() + pd.Timedelta(days=1) if day is None else pd.Timestamp(day)
    _, counts, sums, _ = next(snaps.iter_days([day]))
    frame = item_frame_at(snaps.items, counts, sums, static, fs.mu, shrinkage)
    with torch.no_grad():
        emb = model.cpu().eval().item_tower(torch.from_numpy(fs.artifacts["item_tf"].transform(frame))).numpy()
    return snaps.items, emb
