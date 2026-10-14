"""Rating metrics, overall and by how much history the user had at prediction time.

In plain words: Basic error measures (RMSE, MAE) and a breakdown by how many reviews the user had
written before.
"""

import numpy as np
import pandas as pd

HISTORY_BUCKETS = [("no history", 0, 0), ("1 prior", 1, 1), ("2-4 prior", 2, 4), ("5+ prior", 5, 10**9)]


def rmse(y, pred) -> float:
    """Root mean squared error: the typical size of the prediction error in stars, punishing big misses
    more. Lower is better.
    """
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def mae(y, pred) -> float:
    """Mean absolute error: the average distance between prediction and truth, in stars. Lower is
    better.
    """
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def history_bucket_masks(u_hist_count) -> dict:
    """Group rows by how many reviews the user had written before (none / 1 / 2-4 / 5+) so we can see
    where a method works.
    """
    c = np.asarray(u_hist_count)
    return {name: (c >= lo) & (c <= hi) for name, lo, hi in HISTORY_BUCKETS}


def rating_report(y, preds: dict, u_hist_count) -> pd.DataFrame:
    """One row per method: overall RMSE/MAE plus RMSE inside each history bucket."""
    masks = history_bucket_masks(u_hist_count)
    rows = {}
    for name, p in preds.items():
        row = {"rmse": rmse(y, p), "mae": mae(y, p)}
        for b, m in masks.items():
            row[f"rmse [{b}]"] = rmse(np.asarray(y)[m], np.asarray(p)[m]) if m.any() else np.nan
        rows[name] = row
    return pd.DataFrame(rows).T
