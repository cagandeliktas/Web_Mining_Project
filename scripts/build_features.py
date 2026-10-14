"""Build and cache the leak-free feature set (temporal split + as-of features).

Usage:
    python -m scripts.build_features [--config configs/default.yaml] [--refresh] [--refresh-data]
"""

import argparse
import time

import numpy as np

from src.config import DEFAULT_CONFIG_PATH, load_config
from src.pipeline import load_or_build_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--refresh", action="store_true", help="rebuild features even if cached")
    parser.add_argument("--refresh-data", action="store_true", help="also re-clean the raw CSVs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    t0 = time.time()
    fs = load_or_build_features(cfg, refresh=args.refresh or args.refresh_data, refresh_data=args.refresh_data)
    print(f"ready in {time.time() - t0:.0f}s | X_user {fs.X_user.shape}, X_item {fs.X_item.shape}, mu={fs.mu:.4f}")
    for name in ("train", "val", "test"):
        m = fs.split == name
        print(f"{name:5s} rows={m.sum():>8,}  cold-user-slot={np.mean(fs.user_idx[m] == 0):.3f}  "
              f"cold-item-slot={np.mean(fs.item_idx[m] == 0):.3f}")


if __name__ == "__main__":
    main()
