"""Train the PyTorch two-tower model on the cached leak-free features.

Reports validation metrics only; the test split is scored once, in the
evaluation step, against the baselines.

Usage:
    python -m scripts.train [--config configs/default.yaml] [--out runs/latest]
"""

import argparse
import json
import time
from pathlib import Path

from src.config import DEFAULT_CONFIG_PATH, load_config
from src.models.two_tower import save_model
from src.pipeline import load_or_build_features
from src.seed import resolve_device
from src.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out", default="runs/latest")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fs = load_or_build_features(cfg)
    device = resolve_device(cfg.train.device)
    print(f"device={device}  train rows={(fs.split == 'train').sum():,}  mu={fs.mu:.4f}")

    t0 = time.time()
    res = train_model(fs, cfg, device=device)
    print(f"best epoch {res.best_epoch}: val_rmse={res.best_val_rmse:.4f}  ({time.time() - t0:.0f}s)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_model(res.model, out / "model.pt")
    (out / "history.json").write_text(json.dumps(res.history, indent=2))
    print(f"saved {out}/model.pt")


if __name__ == "__main__":
    main()
