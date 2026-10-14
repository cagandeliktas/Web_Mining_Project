"""Package a trained model for the API: model + fitted transforms + a catalog snapshot.

Usage:
    python -m scripts.train                                   # produces runs/latest/model.pt
    python -m scripts.export_serving_bundle [--model runs/latest/model.pt] [--out artifacts/serving]
"""

import argparse

import pandas as pd

from src.config import DEFAULT_CONFIG_PATH, load_config
from src.models.two_tower import load_model
from src.pipeline import load_or_build_features
from src.serving.bundle import build_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", default="runs/latest/model.pt")
    parser.add_argument("--out", default="artifacts/serving")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fs = load_or_build_features(cfg)
    bundle = build_bundle(fs, load_model(args.model), pd.read_csv(cfg.data.product_info), cfg.features.item_shrinkage)
    bundle.save(args.out)
    print(f"wrote {args.out}: {len(bundle.item_ids)} products, snapshot {bundle.snapshot_day}, "
          f"embed_dim {bundle.item_emb.shape[1]}")


if __name__ == "__main__":
    main()
