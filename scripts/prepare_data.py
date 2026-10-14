"""Clean the raw reviews, cache them as parquet, and print the temporal split report.

In plain words: Step 1 of the pipeline: read the raw review CSVs, clean them, cache the result and
print what the time-based train/val/test split looks like.

Usage:
    python -m scripts.prepare_data [--config configs/default.yaml] [--refresh]
"""

import argparse

import pandas as pd

from src.config import DEFAULT_CONFIG_PATH, load_config
from src.data import load_interactions
from src.split import assert_no_leakage, split_report, temporal_split


def main() -> None:
    """Command-line entry point: clean the raw data, cache it and print the train/val/test split
    report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--refresh", action="store_true", help="ignore the parquet cache")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_interactions(cfg.data.dir, cfg.data.cache_dir, refresh=args.refresh)
    print(f"{len(df):,} interactions | {df['author_id'].nunique():,} users | "
          f"{df['product_id'].nunique():,} products")

    labels, cut = temporal_split(df, cfg.split.val_frac, cfg.split.test_frac)
    assert_no_leakage(df, labels)
    print(f"val starts {cut.val_start.date()}, test starts {cut.test_start.date()}")
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print(split_report(df, labels))


if __name__ == "__main__":
    main()
