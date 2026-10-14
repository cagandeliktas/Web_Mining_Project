"""Tests the API: request features equal training features, fast scoring equals the full model,
endpoints behave.
"""
import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from src.pipeline import build_features
from src.ranking import CatalogSnapshots, item_frame_at
from src.features.item import build_item_static
from src.serving.api import create_app
from src.serving.bundle import ServingBundle, build_bundle
from src.serving.recommender import Recommender, request_user_features
from src.split import temporal_split
from src.train import train_model
from tests.synthetic import make_interactions, make_product_info, small_cfg

DEMO = ["skin_tone", "skin_type", "eye_color", "hair_color"]
HISTORY_COLS = ["rating", "is_recommended", "review_text", "submission_time", "product_id", *DEMO]


@pytest.fixture(scope="module")
def served():
    df = make_interactions(n=900, n_users=40, n_items=15)
    info = make_product_info()
    labels, _ = temporal_split(df, 0.15, 0.25)
    cfg = small_cfg()
    cfg.train.device, cfg.train.epochs, cfg.train.batch_size = "cpu", 3, 64
    cfg.model.hidden, cfg.model.embed_dim = [16], 8
    fs = build_features(df, info, labels, cfg)
    model = train_model(fs, cfg, verbose=False).model
    bundle = build_bundle(fs, model, info, cfg.features.item_shrinkage)
    return df, info, fs, cfg, model, bundle


def _history_before(df, row):
    r = df.iloc[row]
    h = df[(df["author_id"] == r["author_id"]) & (df["submission_time"] < r["submission_time"])]
    return h[HISTORY_COLS].reset_index(drop=True), {c: r[c] for c in DEMO}


# ------------------------------------------------------------------ feature parity with training
def test_request_features_equal_the_training_features_for_the_same_user_state(served):
    df, _, fs, _, _, bundle = served
    with_hist = np.where(fs.meta["u_hist_count"].to_numpy() >= 2)[0]
    without = np.where(fs.meta["u_hist_count"].to_numpy() == 0)[0]
    assert len(with_hist) > 10 and len(without) > 10
    for row in list(with_hist[::37][:8]) + list(without[::29][:5]):
        hist, profile = _history_before(df, row)
        got = request_user_features(bundle, hist if len(hist) else None, profile)
        np.testing.assert_allclose(got[0], fs.X_user[row], atol=1e-5, err_msg=f"row {row}")


def test_history_order_and_missing_timestamps_do_not_change_the_features(served):
    df, _, fs, _, _, bundle = served
    row = int(np.where(fs.meta["u_hist_count"].to_numpy() >= 3)[0][5])
    hist, profile = _history_before(df, row)
    base = request_user_features(bundle, hist, profile)
    shuffled = request_user_features(bundle, hist.sample(frac=1.0, random_state=1), profile)
    no_times = request_user_features(bundle, hist.drop(columns="submission_time"), profile)
    np.testing.assert_allclose(base, shuffled, atol=1e-6)
    np.testing.assert_allclose(base, no_times, atol=1e-6)


def test_profile_falls_back_to_latest_answer_in_history(served):
    df, _, _, _, _, bundle = served
    hist = pd.DataFrame({"rating": [4.0, 5.0], "submission_time": ["2020-01-01", "2020-02-01"],
                         "skin_tone": ["fair", "dark"]})
    inferred = request_user_features(bundle, hist, None)
    explicit = request_user_features(bundle, hist, {"skin_tone": "dark"})
    other = request_user_features(bundle, hist, {"skin_tone": "fair"})
    np.testing.assert_allclose(inferred, explicit)
    assert not np.allclose(inferred, other)


# ------------------------------------------------------------------ scoring parity with the real model
def test_precomputed_catalog_scores_equal_a_full_model_forward(served):
    df, info, fs, cfg, model, bundle = served
    inter = fs.meta[["author_id", "product_id", "submission_time"]].assign(rating=fs.y.astype("float64"))
    snaps = CatalogSnapshots(inter)
    day = inter["submission_time"].max() + pd.Timedelta(days=1)
    _, counts, sums, _ = next(snaps.iter_days([day]))
    frame = item_frame_at(snaps.items, counts, sums, build_item_static(info), fs.mu, cfg.features.item_shrinkage)
    x_item = torch.from_numpy(fs.artifacts["item_tf"].transform(frame))
    item_idx = torch.from_numpy(fs.artifacts["item_vocab"].transform(pd.Series(snaps.items)).copy())

    rec = Recommender(bundle)
    hist, profile = _history_before(df, int(np.where(fs.meta["u_hist_count"].to_numpy() >= 2)[0][3]))
    xu = torch.from_numpy(request_user_features(bundle, hist, profile))
    with torch.no_grad():
        full = model(xu.repeat(len(x_item), 1), x_item, torch.zeros(len(x_item), dtype=torch.long), item_idx).numpy()
    fast = rec.raw_scores(rec.user_embedding(hist, profile))
    assert list(bundle.item_ids) == list(snaps.items)
    np.testing.assert_allclose(fast, full, atol=1e-4)


# ------------------------------------------------------------------ recommender behaviour
def test_recommend_excludes_reviewed_and_listed_products_and_is_sorted(served):
    _, _, _, _, _, bundle = served
    rec = Recommender(bundle)
    reviews = pd.DataFrame({"rating": [5.0, 4.0], "product_id": ["P1", "P2"], "review_text": ["love it", None]})
    out = rec.recommend(reviews, None, top_k=6, exclude=["P3"])
    assert len(out) == 6 and not {"P1", "P2", "P3"} & set(out["product_id"])
    assert out["predicted_rating"].between(1, 5).all()
    assert out["predicted_rating"].is_monotonic_decreasing
    assert set(out.columns) >= {"product_id", "product_name", "brand_name", "price_usd", "primary_category"}


def test_min_item_reviews_filters_thin_products_and_predict_agrees_with_recommend(served):
    _, _, _, _, _, bundle = served
    rec = Recommender(bundle)
    cutoff = int(np.median(bundle.item_count))
    out = rec.recommend(None, None, top_k=50, min_item_reviews=cutoff)
    counts = pd.Series(bundle.item_count, index=bundle.item_ids)
    assert (counts[out["product_id"]] >= cutoff).all() and len(out) < len(bundle.item_ids)
    top = rec.recommend(None, None, top_k=1).iloc[0]
    assert rec.predict(top["product_id"]) == pytest.approx(top["predicted_rating"])
    with pytest.raises(KeyError):
        rec.predict("NOPE")


def test_bundle_save_load_roundtrip_gives_identical_scores(served, tmp_path):
    _, _, _, _, _, bundle = served
    bundle.save(tmp_path)
    back = ServingBundle.load(tmp_path)
    reviews = pd.DataFrame({"rating": [5.0, 3.0], "review_text": ["soft glow", "meh"], "skin_tone": ["fair", "fair"]})
    a, b = Recommender(bundle), Recommender(back)
    np.testing.assert_allclose(a.raw_scores(a.user_embedding(reviews)), b.raw_scores(b.user_embedding(reviews)), atol=1e-6)
    assert back.snapshot_day == bundle.snapshot_day


def test_missing_bundle_fails_fast_with_a_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="export_serving_bundle"):
        ServingBundle.load(tmp_path / "nothing-here")


# ------------------------------------------------------------------ HTTP API
@pytest.fixture()
def client(served):
    bundle = served[-1]
    with TestClient(create_app(lambda: bundle)) as c:
        yield c


def test_health(client, served):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["n_products"] == len(served[-1].item_ids)


def test_recommendations_for_a_brand_new_user_and_for_a_user_with_history(client):
    cold = client.post("/recommendations", json={"top_k": 5}).json()
    assert cold["n_reviews_used"] == 0 and len(cold["recommendations"]) == 5

    body = {"top_k": 5, "profile": {"skin_type": "dry"},
            "reviews": [{"rating": 5, "is_recommended": True, "review_text": "Loved it!", "product_id": "P1",
                         "skin_tone": "fair"}, {"rating": 2}]}
    warm = client.post("/recommendations", json=body).json()
    ids = [r["product_id"] for r in warm["recommendations"]]
    assert warm["n_reviews_used"] == 2 and "P1" not in ids
    assert all(1 <= r["predicted_rating"] <= 5 for r in warm["recommendations"])
    assert client.post("/recommendations", json=body).json() == warm  # deterministic


def test_predict_matches_recommendations_and_unknown_product_is_404(client):
    top = client.post("/recommendations", json={"top_k": 1}).json()["recommendations"][0]
    got = client.post("/predict", json={"product_id": top["product_id"]})
    assert got.status_code == 200 and got.json()["predicted_rating"] == pytest.approx(top["predicted_rating"])
    missing = client.post("/predict", json={"product_id": "NOPE"})
    assert missing.status_code == 404 and "NOPE" in missing.json()["detail"]


@pytest.mark.parametrize("payload", [
    {"top_k": 0}, {"top_k": 101}, {"reviews": [{"rating": 7}]}, {"reviews": [{"rating": 0.5}]}, {"min_item_reviews": -1},
])
def test_invalid_requests_are_rejected_with_422(client, payload):
    assert client.post("/recommendations", json=payload).status_code == 422


def test_missing_catalog_price_serialises_as_null(served):
    bundle = served[-1]
    meta = bundle.item_meta.copy()
    meta["price_usd"] = np.nan
    patched = ServingBundle(**{**bundle.__dict__, "item_meta": meta})
    with TestClient(create_app(lambda: patched)) as c:
        recs = c.post("/recommendations", json={"top_k": 3}).json()["recommendations"]
    assert all(r["price_usd"] is None for r in recs)


def test_app_refuses_to_start_without_a_bundle():
    def broken():
        raise FileNotFoundError("no bundle")

    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(broken)):
            pass
