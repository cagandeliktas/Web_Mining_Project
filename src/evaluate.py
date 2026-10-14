"""Rating metrics, overall and by how much history the user had at prediction time."""

import numpy as np
import pandas as pd

HISTORY_BUCKETS = [("no history", 0, 0), ("1 prior", 1, 1), ("2-4 prior", 2, 4), ("5+ prior", 5, 10**9)]


def rmse(y, pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def mae(y, pred) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def history_bucket_masks(u_hist_count) -> dict:
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
