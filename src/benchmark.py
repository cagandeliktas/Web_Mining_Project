"""Baselines vs the two-tower on val/test, as one reusable call.

In plain words: Runs every method (all baselines plus the neural network) on validation and test and
collects the numbers into tables.

Baseline hyper-parameters (shrinkage strengths) are chosen on **val only**;
the test split is scored once per call, for every method.
"""

import json
from pathlib import Path

import pandas as pd

from src.baselines import BiasBaseline, GlobalMean, ItemKNN, ItemMean, UserMean, asof_additive
from src.evaluate import rating_report, rmse
from src.features.item import build_item_static
from src.ranking import evaluate_ranking
from src.torch_data import InteractionData
from src.train import predict


def _best(candidates: dict, val_df, verbose: bool):
    """Try several settings and return the one with the lowest validation RMSE (the test set is never
    used to choose).
    """
    scored = {n: rmse(val_df["rating"], m.predict(val_df)) for n, m in candidates.items()}
    name = min(scored, key=scored.get)
    if verbose:
        print(f"  tuned on val: {name} (rmse {scored[name]:.4f})")
    return candidates[name]


def rating_predictions(fs, model, skip_knn: bool = False, verbose: bool = True) -> dict:
    """{'val': {method: preds}, 'test': {...}} for every baseline and the neural model."""
    inter = fs.meta[["author_id", "product_id", "submission_time"]].assign(rating=fs.y.astype("float64"))
    train = inter[fs.split == "train"]
    parts = {s: inter[fs.split == s] for s in ("val", "test")}
    preds = {s: {} for s in parts}

    def add(name, m):
        """Predict the val and test rows with one fitted baseline and store them under `name`."""
        for s, df in parts.items():
            preds[s][name] = m.predict(df)

    add("global mean", GlobalMean().fit(train))
    add("item mean", _best({f"k={k}": ItemMean(k).fit(train) for k in (1, 5, 10, 25, 100)}, parts["val"], verbose))
    add("user mean", _best({f"k={k}": UserMean(k).fit(train) for k in (1, 5, 10, 25, 100)}, parts["val"], verbose))
    add("user+item bias", _best(
        {f"lu={lu},li={li}": BiasBaseline(lu, li).fit(train) for lu in (5, 15, 40) for li in (5, 10, 25)},
        parts["val"], verbose))

    # as-of additive sees every split's strictly-earlier rows, like the neural features do
    grid = {(ku, ki): asof_additive(inter, fs.mu, ku, ki) for ku in (1, 5, 15) for ki in (5, 10, 25)}
    val_mask = fs.split == "val"
    best_k = min(grid, key=lambda k: rmse(fs.y[val_mask], grid[k][val_mask]))
    if verbose:
        print(f"  tuned on val: as-of additive k_user,k_item={best_k}")
    for s in parts:
        preds[s]["user+item as-of"] = grid[best_k][fs.split == s]

    if not skip_knn:
        add("item kNN (k=40)", _best(
            {f"sim_shrink={sh},den={d}": ItemKNN(40, sh, d).fit(train) for sh in (10, 50) for d in (0.3, 1.0)},
            parts["val"], verbose))

    for s in parts:
        preds[s]["two-tower (PyTorch)"] = predict(model, InteractionData(fs.part(s), "cpu"))
    return preds


def run_benchmark(cfg, fs, model, n_rank_sample: int = 20000, skip_knn: bool = False, verbose: bool = True) -> dict:
    """Returns {'rating': {split: DataFrame}, 'ranking': {split: {method: metrics}}}."""
    preds = rating_predictions(fs, model, skip_knn, verbose)
    rating = {
        s: rating_report(fs.y[fs.split == s], preds[s], fs.meta.loc[fs.split == s, "u_hist_count"].to_numpy())
        for s in preds
    }
    inter = fs.meta[["author_id", "product_id", "submission_time"]].assign(rating=fs.y.astype("float64"))
    static = build_item_static(pd.read_csv(cfg.data.product_info))
    ranking = {}
    for s in ("val", "test"):
        r = evaluate_ranking(fs, inter, static, model, s, threshold=cfg.eval.rating_threshold, ks=cfg.eval.top_k,
                             n_sample=n_rank_sample, shrinkage=cfg.features.item_shrinkage, seed=cfg.seed)
        ranking[s] = {k: v for k, v in r.items() if not k.startswith("_")}
        ranking[s]["_meta"] = {"n_targets": len(r["_rows"]), "mean_candidates": r["_mean_candidates"]}
    return {"rating": rating, "ranking": ranking}


def write_results(results: dict, out_dir) -> None:
    """Save the tables as results.json (for programs) and results.md (for people) in out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(
        {"rating": {s: df.to_dict(orient="index") for s, df in results["rating"].items()},
         "ranking": results["ranking"]}, indent=2))
    with open(out / "results.md", "w") as f:
        for s, df in results["rating"].items():
            f.write(f"### Rating prediction - {s}\n\n{df.sort_values('rmse').round(4).to_markdown()}\n\n")
        for s, r in results["ranking"].items():
            table = pd.DataFrame({k: v for k, v in r.items() if k != "_meta"}).T
            f.write(f"### Full-catalog ranking - {s} ({r['_meta']['n_targets']:,} liked interactions, "
                    f"~{r['_meta']['mean_candidates']:.0f} candidates each)\n\n{table.round(4).to_markdown()}\n\n")
