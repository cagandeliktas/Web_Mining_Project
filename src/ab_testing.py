"""Statistical comparison between two recommenders' per-instance prediction errors.

Used to compare the surprise-based CF baseline ("variant A") against the
two-tower neural network ("variant B") on the same held-out (user, item,
true_rating) instances - see scripts/run_ab_test.py for how those paired
errors are actually produced.
"""

import numpy as np
from scipy import stats


def prediction_errors(true_ratings, predicted_ratings) -> np.ndarray:
    """Absolute error per instance: |true - predicted|."""
    true_ratings = np.asarray(true_ratings, dtype=float)
    predicted_ratings = np.asarray(predicted_ratings, dtype=float)
    if true_ratings.shape != predicted_ratings.shape:
        raise ValueError("true_ratings and predicted_ratings must be the same shape")
    return np.abs(true_ratings - predicted_ratings)


def compare_variants(errors_a, errors_b, alpha: float = 0.05) -> dict:
    """Paired significance test between two variants' per-instance errors.

    Runs both a paired t-test (assumes the error differences are roughly
    normal) and a Wilcoxon signed-rank test (no normality assumption, more
    robust to outliers) on the same paired instances - standard practice
    for this kind of comparison, and useful to check they agree.

    Parameters
    ----------
    errors_a, errors_b : array-like
        Per-instance errors (e.g. from `prediction_errors`) for variant A
        and variant B respectively, from the SAME held-out instances in
        the SAME order - this only makes sense as a paired comparison.
    alpha : float
        Significance threshold for `significant_at_alpha`.

    Returns
    -------
    dict
        Summary stats plus both tests' statistic/p-value, and which
        variant had the lower mean error.
    """
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    if errors_a.shape != errors_b.shape:
        raise ValueError("errors_a and errors_b must be paired (same shape)")
    if errors_a.ndim != 1:
        raise ValueError("errors must be 1-D arrays")
    if len(errors_a) < 2:
        raise ValueError("need at least 2 paired instances to run a significance test")

    diff = errors_a - errors_b  # positive => B has lower error (B better)

    if np.all(diff == 0):
        # Zero variance in the differences makes the t-statistic 0/0 (scipy
        # returns NaN); with no difference at all, the honest answer for
        # both tests is "certainly no difference".
        ttest_statistic, ttest_pvalue = 0.0, 1.0
        wilcoxon_statistic, wilcoxon_pvalue = 0.0, 1.0
    else:
        ttest = stats.ttest_rel(errors_a, errors_b)
        ttest_statistic, ttest_pvalue = ttest.statistic, ttest.pvalue
        wilcoxon = stats.wilcoxon(errors_a, errors_b)
        wilcoxon_statistic, wilcoxon_pvalue = wilcoxon.statistic, wilcoxon.pvalue

    if diff.mean() > 0:
        better_variant = "b"
    elif diff.mean() < 0:
        better_variant = "a"
    else:
        better_variant = "tie"

    return {
        "n": int(len(errors_a)),
        "mean_error_a": float(errors_a.mean()),
        "mean_error_b": float(errors_b.mean()),
        "mean_diff": float(diff.mean()),
        "ttest_statistic": float(ttest_statistic),
        "ttest_pvalue": float(ttest_pvalue),
        "wilcoxon_statistic": float(wilcoxon_statistic),
        "wilcoxon_pvalue": float(wilcoxon_pvalue),
        "significant_at_alpha": bool(ttest_pvalue < alpha),
        "better_variant": better_variant,
    }
