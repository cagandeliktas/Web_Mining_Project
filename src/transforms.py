"""Fit-on-train preprocessing: scaling, imputation, one-hot, and id vocabularies.

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
        self.numeric = list(numeric)
        self.log1p = set(log1p)
        self.binary = list(binary)
        self.categorical = list(categorical)
        self.missing_indicators = list(missing_indicators)
        self.min_count = min_count

    def _numeric_values(self, df: pd.DataFrame, c: str) -> np.ndarray:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
        if c in self.log1p:
            x = np.log1p(np.clip(x, 0.0, None))  # NaN stays NaN
        return x

    def fit(self, df: pd.DataFrame) -> "TabularTransformer":
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
        self.index_ = {v: i + 1 for i, v in enumerate(sorted(pd.unique(ids.dropna())))}
        return self

    def __len__(self) -> int:  # number of embedding rows incl. the cold slot 0
        return len(self.index_) + 1

    def transform(self, ids: pd.Series) -> np.ndarray:
        return ids.map(self.index_).fillna(0).to_numpy(dtype="int64")
