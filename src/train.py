"""Training loop: Adam, MSE/Huber + bias shrinkage, id dropout, val-based early stopping.

In plain words: The training loop: show the network batches of reviews, adjust it to reduce error,
and stop when validation error stops improving.
"""

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from src.models.two_tower import TwoTower
from src.pipeline import FeatureSet
from src.seed import resolve_device, set_seed
from src.torch_data import InteractionData

RATING_MIN, RATING_MAX = 1.0, 5.0


@dataclass
class TrainResult:
    """What training returns: the best model, the per-epoch history, and which epoch was best."""
    model: TwoTower
    history: list = field(default_factory=list)
    best_epoch: int = -1
    best_val_rmse: float = float("inf")


def build_model(cfg, fs: FeatureSet) -> TwoTower:
    """Create an untrained TwoTower sized to the features and id lists in `fs`, using the settings in
    `cfg`.
    """
    art = fs.artifacts
    return TwoTower(
        n_user_features=fs.X_user.shape[1],
        n_item_features=fs.X_item.shape[1],
        n_users=len(art["user_vocab"]),
        n_items=len(art["item_vocab"]),
        mu=fs.mu,
        embed_dim=cfg.model.embed_dim,
        hidden=cfg.model.hidden,
        dropout=cfg.model.dropout,
        use_biases=cfg.model.use_biases,
    )


@torch.no_grad()
def predict(model: TwoTower, data: InteractionData, batch_size: int = 16384, clip: bool = True) -> np.ndarray:
    """Predict ratings for every row in `data` (clipped to 1-5 by default)."""
    model.eval()
    dev = next(model.parameters()).device  # the data may live on a different device than the model
    out = [model(xu.to(dev), xi.to(dev), ui.to(dev), ii.to(dev)) for xu, xi, ui, ii, _ in data.batches(batch_size)]
    pred = torch.cat(out)
    if clip:
        pred = pred.clamp(RATING_MIN, RATING_MAX)
    return pred.cpu().numpy()


@torch.no_grad()
def evaluate_split(model: TwoTower, data: InteractionData) -> dict:
    """RMSE and MAE of the model on one split."""
    pred = torch.from_numpy(predict(model, data)).to(data.device)
    err = pred - data.y
    return {"rmse": float(err.pow(2).mean().sqrt()), "mae": float(err.abs().mean())}


def apply_id_dropout(idx: torch.Tensor, p: float) -> torch.Tensor:
    """Randomly send ids to the cold slot 0, so the model also learns to predict without id biases."""
    if p <= 0:
        return idx
    return torch.where(torch.rand(idx.shape, device=idx.device) < p, torch.zeros_like(idx), idx)


def train_model(fs: FeatureSet, cfg, device=None, verbose: bool = True) -> TrainResult:
    """Train with early stopping: after each epoch measure validation error, keep the best weights, and
    stop after `patience` epochs without improvement.
    """
    set_seed(cfg.seed)
    device = resolve_device(cfg.train.device) if device is None else torch.device(device)
    tc = cfg.train

    train = InteractionData(fs.part("train"), device)
    val = InteractionData(fs.part("val"), device)
    model = build_model(cfg, fs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    loss_fn = nn.MSELoss() if tc.loss == "mse" else nn.HuberLoss(delta=1.0)

    # Koren-style shrinkage: lambda * b^2 per id, spread over that id's train rows,
    # so ids with few ratings are pulled hard toward 0 and popular ones barely move.
    n_user_rows = torch.bincount(train.user_idx, minlength=model.init_args["n_users"]).clamp(min=1).float()
    n_item_rows = torch.bincount(train.item_idx, minlength=model.init_args["n_items"]).clamp(min=1).float()

    gen = torch.Generator().manual_seed(cfg.seed)
    result = TrainResult(model=model)
    best_state, bad_epochs = None, 0

    for epoch in range(1, tc.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for xu, xi, ui, ii, y in train.batches(tc.batch_size, shuffle=True, generator=gen):
            ui_in = apply_id_dropout(ui, cfg.model.id_dropout)
            ii_in = apply_id_dropout(ii, cfg.model.id_dropout)
            pred = model(xu, xi, ui_in, ii_in)
            loss = loss_fn(pred, y)
            if model.use_biases:
                b_u, b_i = model.biases(ui_in, ii_in)
                reg = (b_u.pow(2) / n_user_rows[ui_in] + b_i.pow(2) / n_item_rows[ii_in]).mean()
                loss = loss + tc.bias_reg * reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(y)
            seen += len(y)

        val_metrics = evaluate_split(model, val)
        row = {"epoch": epoch, "train_loss": total / seen, "val_rmse": val_metrics["rmse"], "val_mae": val_metrics["mae"]}
        result.history.append(row)
        if verbose:
            print(f"epoch {epoch:2d}  train_loss {row['train_loss']:.4f}  val_rmse {row['val_rmse']:.4f}  val_mae {row['val_mae']:.4f}")

        if val_metrics["rmse"] < result.best_val_rmse - 1e-5:
            result.best_val_rmse, result.best_epoch = val_metrics["rmse"], epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= tc.patience:
                if verbose:
                    print(f"early stop at epoch {epoch} (best {result.best_epoch}, val_rmse {result.best_val_rmse:.4f})")
                break

    model.load_state_dict(best_state)
    result.model = model.eval()
    return result
