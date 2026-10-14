"""Load the YAML config into a nested, attribute-accessible structure.

In plain words: Reads the settings file configs/default.yaml so no setting is hard-coded anywhere
else.
"""

from pathlib import Path
from types import SimpleNamespace

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def _to_namespace(obj):
    """Turn nested dicts into objects so a setting is read as cfg.train.lr instead of
    cfg['train']['lr'].
    """
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    return obj


def load_config(path=DEFAULT_CONFIG_PATH, overrides: dict | None = None) -> SimpleNamespace:
    """Read ``path``; ``overrides`` is a nested dict merged over the file's values."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if overrides:
        _merge(raw, overrides)
    return _to_namespace(raw)


def _merge(base: dict, extra: dict) -> None:
    """Copy values from `extra` into `base`, going into nested dicts instead of replacing them whole.
    """
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
