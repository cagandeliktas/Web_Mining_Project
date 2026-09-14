import math

import numpy as np
import pytest

from src.preprocessing import normalize_size


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("30 ml", 30.0),
        ("30mL", 30.0),
        ("1.7 oz", 1.7 * 29.5735),
        ("50 g", 50.0),
        ("100 grams", 100.0),
        # first numeric+unit match in the string wins when both units are present
        ("1 oz / 30 ml", 1 * 29.5735),
        (".5 oz / 15 mL", 0.5 * 29.5735),
        ("2 x 0.5oz/15ml", 0.5 * 29.5735),
    ],
)
def test_normalize_size_parses_known_formats(raw, expected):
    assert normalize_size(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [None, np.nan, "assorted", "", "N/A"])
def test_normalize_size_returns_nan_when_unparseable(raw):
    assert math.isnan(normalize_size(raw))
