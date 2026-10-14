"""Fit-on-train preprocessing: scaling, imputation, one-hot, and id vocabularies.

In plain words: Preprocessing that is learned on the training data only: scale numbers, fill gaps,
one-hot encode categories and map ids to integers.

Everything learned (medians, means, stds, category lists, id lists) comes from
the rows passed to ``fit`` - the pipeline only ever passes train rows. The
same fitted object is what serving must load, so training and inference
cannot disagree about column order or scaling.
"""

import numpy as np
import pandas as pd

OTHER = "__other__"


class TabularTransformer:
    """Numeric (optional log1p) -> impute -> standardise; binary passthrough; fixed-vocab one-hot.

    Output column order is fixed at fit time (``feature_names_``):
    numeric, missing indicators, binary, then one-hot blocks.
    """

    def __init__(self, numeric=(), log1p=(), binary=(), categorical=(), missing_indicators=(), min_count=1):
        """numeric: columns to scale (log1p: take log(1+x) first); binary: 0/1 columns passed through;
        categorical: columns to one-hot; missing_indicators: columns that also get a 0/1 "was
        missing" flag; min_count: rarest category to keep.
        """
        self.numeric = list(numeric)
        self.log1p = set(log1p)
        self.binary = list(binary)
        self.categorical = list(categorical)
        self.missing_indicators = list(missing_indicators)
        self.min_count = min_count

    def _numeric_values(self, df: pd.DataFrame, c: str) -> np.ndarray:
        """Read a numeric column as floats (optionally log-transformed); bad values become NaN."""
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
        if c in self.log1p:
            x = np.log1p(np.clip(x, 0.0, None))  # NaN stays NaN
        return x

    def fit(self, df: pd.DataFrame) -> "TabularTransformer":
        """Learn from the training rows only: fill values, means and spreads, and the list of
        categories.
        """
        self.fill_, self.mean_, self.std_ = {}, {}, {}
        for c in self.numeric:
            x = self._numeric_values(df, c)
            fill = np.nanmedian(x) if np.isfinite(x).any() else 0.0
            filled = np.where(np.isnan(x), fill, x)
            std = filled.std()
            self.fill_[c] = float(fill)
            self.mean_[c] = float(filled.mean())
            self.std_[c] = float(std) if std > 0 else 1.0
        self.categories_ = {}
        for c in self.categorical:
            counts = df[c].dropna().astype(str).value_counts()
            self.categories_[c] = sorted(counts[counts >= self.min_count].index)
        self.feature_names_ = (
            self.numeric
            + [f"{c}__missing" for c in self.missing_indicators]
            + self.binary
            + [f"{c}={v}" for c in self.categorical for v in self.categories_[c] + [OTHER]]
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply what was learned: fill gaps, scale, flag missing, one-hot; returns a float32 matrix.
        """
        cols = []
        for c in self.numeric:
            x = self._numeric_values(df, c)
            x = np.where(np.isnan(x), self.fill_[c], x)
            cols.append((x - self.mean_[c]) / self.std_[c])
        for c in self.missing_indicators:
            cols.append(pd.to_numeric(df[c], errors="coerce").isna().to_numpy(dtype="float64"))
        for c in self.binary:
            cols.append(pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype="float64"))
        for c in self.categorical:
            values = df[c].astype(object).where(df[c].notna(), None)
            values = values.map(lambda v: None if v is None else str(v)).to_numpy()
            known = set(self.categories_[c])
            for cat in self.categories_[c]:
                cols.append((values == cat).astype("float64"))
            cols.append(np.array([v is None or v not in known for v in values], dtype="float64"))
        return np.column_stack(cols).astype(np.float32)


class IdVocab:
    """Maps ids seen in train to 1..N; anything else (cold) maps to 0."""

    def fit(self, ids: pd.Series) -> "IdVocab":
        """Learn the ids that appear in training and number them 1..N."""
        self.index_ = {v: i + 1 for i, v in enumerate(sorted(pd.unique(ids.dropna())))}
        return self

    def __len__(self) -> int:  # number of embedding rows incl. the cold slot 0
        """Number of slots: the known ids plus slot 0 for "unknown"."""
        return len(self.index_) + 1

    def transform(self, ids: pd.Series) -> np.ndarray:
        """Convert ids to integers; ids never seen in training become 0."""
        return ids.map(self.index_).fillna(0).to_numpy(dtype="int64")
