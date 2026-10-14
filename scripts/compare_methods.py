"""Is the two-tower really better than each baseline? Paired tests on the test split.

Reference methods are the baselines (a); the challenger (b) is the PyTorch model.
Deltas are metric(baseline) - metric(two-tower): positive = the two-tower is better.
Baseline settings are tuned on val (same as scripts.evaluate); the test split is only scored.

Usage:
    python -m scripts.compare_methods [--model runs/latest/model.pt] [--out results] [--n-boot 1000]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ab_testing import compare_methods, holm_adjust
from src.benchmark import rating_predictions
from src.config import DEFAULT_CONFIG_PATH, load_config
from src.models.two_tower import load_model
from src.pipeline import load_or_build_features

CHALLENGER = "two-tower (PyTorch)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", default="runs/latest/model.pt")
    parser.add_argument("--out", default="results")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    fs = load_or_build_features(cfg)
    preds = rating_predictions(fs, load_model(args.model), verbose=False)["test"]
    test = fs.split == "test"
    y = fs.y[test]
    users = fs.meta.loc[test, "author_id"].to_numpy()
    has_history = fs.meta.loc[test, "u_hist_count"].to_numpy() > 0

    baselines = [m for m in preds if m != CHALLENGER]
    rows = {m: compare_methods(y, preds[m], preds[CHALLENGER], users, args.n_boot, cfg.seed) for m in baselines}
    for key in ("delta_rmse_p", "delta_mae_p"):
        for m, adj in zip(baselines, holm_adjust([rows[m][key] for m in baselines])):
            rows[m][key.replace("_p", "_p_holm")] = float(adj)

    # the strongest baseline, split by whether the user had any earlier review
    best = min(baselines, key=lambda m: rows[m]["rmse_a"])
    by_history = {
        label: compare_methods(y[mask], preds[best][mask], preds[CHALLENGER][mask], users[mask], args.n_boot, cfg.seed)
        for label, mask in (("no history", ~has_history), ("has history", has_history)) if mask.sum() > 1
    }

    def fmt(r):
        ci = lambda k: f"[{r[k][0]:+.4f}, {r[k][1]:+.4f}]"
        return {
            "rmse (baseline)": r["rmse_a"], "rmse (two-tower)": r["rmse_b"],
            "delta rmse": r["delta_rmse"], "95% CI": ci("delta_rmse_ci"),
            "p (cluster boot)": r.get("delta_rmse_p_holm", r["delta_rmse_p"]),
            "delta mae": r["delta_mae"], "95% CI (mae)": ci("delta_mae_ci"),
            "p (mae)": r.get("delta_mae_p_holm", r["delta_mae_p"]),
        }

    table = pd.DataFrame({m: fmt(r) for m, r in rows.items()}).T.sort_values("rmse (baseline)")
    by_hist = pd.DataFrame({f"{best} - {k}": fmt(v) for k, v in by_history.items()}).T
    n_users = len(np.unique(users))
    header = (f"Two-tower vs baselines, test split ({test.sum():,} reviews, {n_users:,} users). delta = baseline - two-tower, "
              f"so positive means the two-tower is better. 95% CIs and p-values come from a user-clustered bootstrap "
              f"({args.n_boot} resamples); p-values are Holm-adjusted across the {len(baselines)} baselines. The smallest "
              f"p a bootstrap this size can give is about {1.0 / (args.n_boot + 1):.3f} per comparison, so a value at "
              f"that floor means no resample crossed zero.")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(header, "\n\n", table.to_string(float_format=lambda x: f"{x:.4g}"), "\n\n", by_hist.to_string(float_format=lambda x: f"{x:.4g}"))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ab_test.json").write_text(json.dumps({"versus_baselines": rows, "best_baseline_by_history": by_history,
                                                   "n_boot": args.n_boot}, indent=2))
    (out / "ab_test.md").write_text(
        f"{header}\n\n{table.to_markdown(floatfmt='.4g')}\n\n"
        f"Strongest baseline (`{best}`) by user history:\n\n{by_hist.to_markdown(floatfmt='.4g')}\n")
    print(f"\nwrote {out}/ab_test.json and ab_test.md")


if __name__ == "__main__":
    main()
