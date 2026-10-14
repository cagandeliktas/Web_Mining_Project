"""Paired significance testing between two rating predictors on the same held-out rows.

In plain words: Statistics for "is method B really better than method A on the same reviews?".
Reviews by one user resemble each other, so instead of treating every row as independent it
resamples whole users (a "bootstrap") to get honest uncertainty ranges.

Reviews by one user are correlated, so row-level tests (paired t, Wilcoxon)
overstate certainty. The primary evidence here is therefore a **user-clustered
bootstrap**: whole users are resampled, which keeps their rows together. The
row-level tests are reported alongside for reference.

Convention: ``a`` is the reference (baseline), ``b`` the challenger. Every
"delta" is metric(a) - metric(b), so a positive value means b is better.
"""

import numpy as np
from scipy import stats


def _arrays(y, pred_a, pred_b, groups):
    """Check the inputs are valid (same length, 1-D, at least 2 rows) and turn user ids into integer
    group codes.
    """
    y, pa, pb = (np.asarray(v, dtype="float64") for v in (y, pred_a, pred_b))
    if not (y.shape == pa.shape == pb.shape) or y.ndim != 1:
        raise ValueError("y, pred_a and pred_b must be 1-D arrays of the same length")
    if len(y) < 2:
        raise ValueError("need at least 2 paired instances")
    codes = np.unique(np.asarray(groups), return_inverse=True)[1] if groups is not None else np.arange(len(y))
    if len(codes) != len(y):
        raise ValueError("groups must have one entry per row")
    return y, pa, pb, codes


def cluster_bootstrap(y, pred_a, pred_b, groups=None, n_boot: int = 1000, seed: int = 0, ci: float = 0.95) -> dict:
    """Bootstrap delta-RMSE and delta-MAE (a minus b) by resampling whole groups (users)."""
    y, pa, pb, g = _arrays(y, pred_a, pred_b, groups)
    n_groups = g.max() + 1
    stats_by_group = {
        "n": np.bincount(g, minlength=n_groups).astype("float64"),
        "se_a": np.bincount(g, weights=(y - pa) ** 2, minlength=n_groups),
        "se_b": np.bincount(g, weights=(y - pb) ** 2, minlength=n_groups),
        "ae_a": np.bincount(g, weights=np.abs(y - pa), minlength=n_groups),
        "ae_b": np.bincount(g, weights=np.abs(y - pb), minlength=n_groups),
    }

    def deltas(w):
        """For one resample (w = how many times each user was drawn) compute baseline-minus-challenger
        RMSE and MAE.
        """
        n = w @ stats_by_group["n"]
        d_rmse = np.sqrt(w @ stats_by_group["se_a"] / n) - np.sqrt(w @ stats_by_group["se_b"] / n)
        d_mae = (w @ stats_by_group["ae_a"] - w @ stats_by_group["ae_b"]) / n
        return d_rmse, d_mae

    point_rmse, point_mae = deltas(np.ones(n_groups))
    rng = np.random.default_rng(seed)
    draws = np.array([deltas(np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups).astype("float64"))
                      for _ in range(n_boot)])
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    out = {"n_rows": int(len(y)), "n_groups": int(n_groups)}
    for name, point, col in (("rmse", point_rmse, 0), ("mae", point_mae, 1)):
        d = draws[:, col]
        out[f"delta_{name}"] = float(point)
        out[f"delta_{name}_ci"] = [float(np.percentile(d, lo)), float(np.percentile(d, hi))]
        # two-sided bootstrap p: how often the resampled delta falls on the other side of zero
        out[f"delta_{name}_p"] = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1.0 / (n_boot + 1)))
    return out


def row_level_tests(y, pred_a, pred_b) -> dict:
    """Paired t-test on squared errors (RMSE's basis) and Wilcoxon signed-rank on absolute errors."""
    y, pa, pb, _ = _arrays(y, pred_a, pred_b, None)
    sq_a, sq_b = (y - pa) ** 2, (y - pb) ** 2
    ab_a, ab_b = np.abs(y - pa), np.abs(y - pb)
    t_p = 1.0 if np.all(sq_a == sq_b) else float(stats.ttest_rel(sq_a, sq_b).pvalue)
    w_p = 1.0 if np.all(ab_a == ab_b) else float(stats.wilcoxon(ab_a, ab_b).pvalue)
    return {"ttest_sq_error_p": t_p, "wilcoxon_abs_error_p": w_p}


def compare_methods(y, pred_a, pred_b, groups=None, n_boot: int = 1000, seed: int = 0) -> dict:
    """Everything for one (baseline a, challenger b) pair."""
    y_ = np.asarray(y, dtype="float64")
    rmse = lambda p: float(np.sqrt(np.mean((y_ - np.asarray(p, dtype="float64")) ** 2)))
    mae = lambda p: float(np.mean(np.abs(y_ - np.asarray(p, dtype="float64"))))
    result = {"rmse_a": rmse(pred_a), "rmse_b": rmse(pred_b), "mae_a": mae(pred_a), "mae_b": mae(pred_b)}
    result.update(cluster_bootstrap(y, pred_a, pred_b, groups, n_boot, seed))
    result.update(row_level_tests(y, pred_a, pred_b))
    return result


def holm_adjust(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls family-wise error across several comparisons)."""
    p = np.asarray(pvalues, dtype="float64")
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted
