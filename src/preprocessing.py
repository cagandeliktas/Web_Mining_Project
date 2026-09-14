"""Feature-cleaning utilities used to build the item feature matrix.

Extracted from NeuralNetworkApproach/NN_w_textFeatures.ipynb so the logic can be
unit tested and reused outside the notebook.
"""
import re

import numpy as np
import pandas as pd


def normalize_size(text):
    """Convert the 'size' field to millilitres (mL), with oz converted to mL.

    Solids in g are left as-is. If parsing fails, returns np.nan.
    """
    if pd.isna(text):
        return np.nan

    text = str(text).strip().lower()

    # Match the first numeric value and a valid unit (ml, mL, g, oz), with optional
    # spaces or separators. Handles: "1 oz / 30 ml", "1oz/30ml", ".5 oz / 15 mL",
    # "2 x 0.5oz/15ml"
    m = re.search(r'(?<!\w)([\d.]+)\s*(oz|fl oz|ml|g|mL|grams?)(?!\w)', text)

    if not m:
        return np.nan

    val = m.group(1)
    unit = m.group(2)

    try:
        val = float(val)
    except ValueError:
        return np.nan

    unit = unit.lower()
    if unit in ('ml', 'mL'):
        return val
    elif unit in ('oz', 'fl oz'):
        return val * 29.5735  # Assume oz means fluid oz
    elif unit in ('g', 'gram', 'grams'):
        return val  # Leave solids in g
    else:
        return np.nan
