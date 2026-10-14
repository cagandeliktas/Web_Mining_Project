"""Everything the API needs at request time, exported once from a trained run.

In plain words: The package of files the API needs: the model, the fitted preprocessing and a ready-
made snapshot of the product catalogue.

The bundle holds the fitted text/user transforms and the user tower (to embed a
request), plus a catalog snapshot (item embeddings, item biases, review counts
and display metadata) computed *as of the last review in the data* with the
same code the evaluation uses. Because the item tower does not depend on the
user, its output is precomputed, so a request costs one small MLP pass plus a
matrix-vector product.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.features.item import build_item_static
from src.models.two_tower import TwoTower, load_model, save_model
from src.ranking import catalog_embeddings

META_COLUMNS = ["product_name", "brand_name", "price_usd", "primary_category"]


@dataclass
class ServingBundle:
    """All files the API needs, in memory: model, fitted preprocessing, and the catalogue snapshot
    (product vectors, offsets, review counts, names).
    """
    model: TwoTower
    tfidf: object
    svd: object
    user_tf: object
    mu: float
    item_ids: np.ndarray        # (n_items,) str
    item_emb: np.ndarray        # (n_items, embed_dim) float32
    item_bias: np.ndarray       # (n_items,) float32 (0 for products unseen in train)
    item_count: np.ndarray      # (n_items,) reviews in the snapshot
    item_meta: pd.DataFrame     # indexed by product_id; NaN where the catalog has no row
    snapshot_day: str

    def save(self, directory) -> None:
        """Write the bundle to a folder (model.pt, preprocess.joblib, catalog.joblib)."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        save_model(self.model, d / "model.pt")
        joblib.dump({"tfidf": self.tfidf, "svd": self.svd, "user_tf": self.user_tf, "mu": self.mu},
                    d / "preprocess.joblib")
        joblib.dump({"item_ids": self.item_ids, "item_emb": self.item_emb, "item_bias": self.item_bias,
                     "item_count": self.item_count, "item_meta": self.item_meta,
                     "snapshot_day": self.snapshot_day}, d / "catalog.joblib")

    @classmethod
    def load(cls, directory) -> "ServingBundle":
        """Read a bundle from a folder; gives a clear error if it was never exported."""
        d = Path(directory)
        if not (d / "model.pt").exists():
            raise FileNotFoundError(
                f"No serving bundle at {d}. Create one with: python -m scripts.export_serving_bundle"
            )
        pre = joblib.load(d / "preprocess.joblib")
        cat = joblib.load(d / "catalog.joblib")
        return cls(model=load_model(d / "model.pt"), **pre, **cat)


def build_bundle(fs, model: TwoTower, product_info: pd.DataFrame, shrinkage: float) -> ServingBundle:
    """Snapshot the catalog as of the day after the last review and package the request-time pieces."""
    inter = fs.meta[["author_id", "product_id", "submission_time"]].assign(rating=fs.y.astype("float64"))
    static = build_item_static(product_info)
    model = model.cpu().eval()
    ids, emb = catalog_embeddings(fs, inter, static, model, shrinkage)

    with torch.no_grad():
        idx = torch.from_numpy(fs.artifacts["item_vocab"].transform(pd.Series(ids)).copy())
        bias = model.item_bias(idx).squeeze(-1).numpy() if model.use_biases else np.zeros(len(ids), "float32")

    info = product_info.assign(product_id=product_info["product_id"].astype(str)).drop_duplicates("product_id")
    meta = info.set_index("product_id").reindex(index=ids, columns=META_COLUMNS)  # absent columns -> NaN
    meta.index.name = "product_id"
    counts = inter.groupby("product_id").size().reindex(ids).to_numpy(dtype="float64")
    return ServingBundle(
        model=model,
        tfidf=fs.artifacts["tfidf"],
        svd=fs.artifacts["svd"],
        user_tf=fs.artifacts["user_tf"],
        mu=float(fs.mu),
        item_ids=np.asarray(ids),
        item_emb=emb.astype("float32"),
        item_bias=np.asarray(bias, dtype="float32"),
        item_count=counts,
        item_meta=meta,
        snapshot_day=str((inter["submission_time"].max() + pd.Timedelta(days=1)).date()),
    )
