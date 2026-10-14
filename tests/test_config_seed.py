"""Tests that settings load and merge, and that seeding makes results repeatable."""
import torch

from src.config import load_config
from src.seed import resolve_device, set_seed


def test_default_config_loads_and_overrides_merge():
    cfg = load_config()
    assert cfg.split.method == "temporal"
    assert cfg.model.hidden == [256, 128]
    assert cfg.split.val_frac + cfg.split.test_frac < 1

    cfg2 = load_config(overrides={"train": {"lr": 0.5}})
    assert cfg2.train.lr == 0.5
    assert cfg2.train.epochs == cfg.train.epochs  # untouched keys survive the merge


def test_set_seed_makes_torch_deterministic():
    set_seed(7)
    a = torch.randn(5)
    set_seed(7)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_resolve_device_explicit_cpu():
    assert resolve_device("cpu") == torch.device("cpu")
