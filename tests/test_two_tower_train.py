"""Tests the model formula, training, early stopping, repeatability and saving/loading."""
import numpy as np
import pytest
import torch

from src.models.two_tower import TwoTower, load_model, save_model
from src.pipeline import build_features
from src.split import temporal_split
from src.torch_data import InteractionData
from src.train import apply_id_dropout, build_model, evaluate_split, predict, train_model
from tests.synthetic import make_interactions, make_product_info, small_cfg


@pytest.fixture(scope="module")
def fs():
    df = make_interactions(n=600, n_users=50, n_items=15)
    labels, _ = temporal_split(df, 0.15, 0.15)
    return build_features(df, make_product_info(), labels, small_cfg())


def _cfg(**train):
    from src.config import load_config

    return load_config(overrides={
        "features": {"svd_components": 4, "tfidf_min_df": 1, "tfidf_max_df": 1.0, "tfidf_max_features": 50},
        "model": {"hidden": [16], "embed_dim": 8},
        "train": {"batch_size": 64, "epochs": 12, "device": "cpu", "lr": 0.01, **train},
    })


def test_forward_matches_manual_formula_and_cold_slot_has_zero_bias(fs):
    model = build_model(_cfg(), fs).eval()
    with torch.no_grad():
        model.user_bias.weight[1] = 0.7
        model.item_bias.weight[2] = -0.3
    data = InteractionData(fs.part("train")); xu, xi, ui, ii, _ = data[:6]
    ui = torch.tensor([1, 1, 0, 0, 1, 0]); ii = torch.tensor([2, 0, 2, 0, 2, 2])
    with torch.no_grad():
        out = model(xu, xi, ui, ii)
        dot = (model.user_tower(xu) * model.item_tower(xi)).sum(-1)
    expected = model.mu + dot + torch.tensor([0.7, 0.7, 0.0, 0.0, 0.7, 0.0]) + torch.tensor([-0.3, 0, -0.3, 0, -0.3, -0.3])
    assert out.shape == (6,)
    torch.testing.assert_close(out, expected)
    assert model.user_bias.weight[0].abs().sum() == 0 and model.item_bias.weight[0].abs().sum() == 0


def test_cold_slot_stays_zero_after_training(fs):
    res = train_model(fs, _cfg(), verbose=False)
    assert res.model.user_bias.weight[0].abs().sum() == 0
    assert res.model.item_bias.weight[0].abs().sum() == 0


def test_id_dropout_sends_roughly_p_of_ids_to_slot_zero_and_p0_is_noop():
    idx = torch.arange(1, 10001)
    out = apply_id_dropout(idx, 0.2)
    assert (out == 0).float().mean().item() == pytest.approx(0.2, abs=0.02)
    assert torch.equal(apply_id_dropout(idx, 0.0), idx)


def test_training_beats_the_constant_mu_baseline_on_train_rows(fs):
    res = train_model(fs, _cfg(patience=50, epochs=25), verbose=False)
    tr = InteractionData(fs.part("train"))
    fit = evaluate_split(res.model, tr)["rmse"]
    const = float(np.sqrt(np.mean((fs.part("train")["y"] - fs.mu) ** 2)))
    assert fit < const
    assert all(np.isfinite(h["val_rmse"]) for h in res.history)


def test_early_stopping_restores_best_epoch_weights(fs):
    res = train_model(fs, _cfg(patience=2, epochs=40, lr=0.05), verbose=False)
    best = min(h["val_rmse"] for h in res.history)
    assert res.best_val_rmse == pytest.approx(best)
    assert res.history[res.best_epoch - 1]["val_rmse"] == pytest.approx(best)
    assert len(res.history) <= 40
    val = InteractionData(fs.part("val"))
    assert evaluate_split(res.model, val)["rmse"] == pytest.approx(best, abs=1e-4)  # weights really were restored


def test_same_seed_gives_identical_predictions(fs):
    a = train_model(fs, _cfg(epochs=3), verbose=False)
    b = train_model(fs, _cfg(epochs=3), verbose=False)
    val = InteractionData(fs.part("val"))
    np.testing.assert_allclose(predict(a.model, val), predict(b.model, val), rtol=0, atol=1e-6)


def test_save_load_roundtrip_and_prediction_range(fs, tmp_path):
    res = train_model(fs, _cfg(epochs=3), verbose=False)
    path = tmp_path / "m.pt"
    save_model(res.model, path)
    back = load_model(path)
    test = InteractionData(fs.part("test"))
    p1, p2 = predict(res.model, test), predict(back, test)
    np.testing.assert_allclose(p1, p2, atol=1e-6)
    assert p1.min() >= 1.0 and p1.max() <= 5.0
