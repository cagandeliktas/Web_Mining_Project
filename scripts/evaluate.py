"""One results table: baselines vs the PyTorch two-tower on val and test.

In plain words: Scores the trained model and all the simple baselines on validation and test data
and writes the results tables.

Usage:
    python -m scripts.evaluate [--model runs/latest/model.pt] [--out runs/latest]
"""

import argparse

import pandas as pd

from src.benchmark import run_benchmark, write_results
from src.config import DEFAULT_CONFIG_PATH, load_config
from src.models.two_tower import load_model
from src.pipeline import load_or_build_features


def main() -> None:
    """Command-line entry point: score all methods on validation and test and write the results tables.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", default="runs/latest/model.pt")
    parser.add_argument("--out", default="runs/latest")
    parser.add_argument("--n-rank-sample", type=int, default=20000)
    parser.add_argument("--skip-knn", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = run_benchmark(cfg, load_or_build_features(cfg), load_model(args.model), args.n_rank_sample, args.skip_knn)
    write_results(results, args.out)
    with pd.option_context("display.width", 220, "display.max_columns", None, "display.float_format", "{:.4f}".format):
        for s, df in results["rating"].items():
            print(f"\n=== rating prediction, {s} ===\n{df.sort_values('rmse')}")
        for s, r in results["ranking"].items():
            print(f"\n=== full-catalog ranking, {s} ===\n{pd.DataFrame({k: v for k, v in r.items() if k != '_meta'}).T}")
    print(f"\nwrote {args.out}/results.json and results.md")


if __name__ == "__main__":
    main()
