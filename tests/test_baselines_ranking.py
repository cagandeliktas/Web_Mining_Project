import numpy as np
import pandas as pd
import pytest

from src.baselines import BiasBaseline, GlobalMean, ItemKNN, ItemMean, UserMean, asof_additive
from src.evaluate import history_bucket_masks, rating_report
from src.features.item import build_item_static
from src.pipeline import build_features
from src.ranking import CatalogSnapshots, evaluate_ranking, item_frame_at, rank_of_target, ranking_metrics
from src.split import temporal_split
from src.train import train_model
from tests.synthetic import make_interactions, make_product_info, small_cfg


def _df(rows):
    return pd.DataFrame(rows, columns=["author_id", "product_id", "rating"])


# ---------------------------------------------------------------- baselines
def test_shrunk_item_and_user_mean_and_unseen_fallback():
    train = _df([("u1", "A", 5), ("u1", "B", 1), ("u2", "A", 3)])
    mu = 3.0
    im = ItemMean(k=2).fit(train)
    out = im.predict(_df([("x", "A", 0), ("x", "Z", 0)]))
    assert out[0] == pytest.approx((8 + 2 * mu) / (2 + 2))   # item A: sum 8, n 2
    assert out[1] == pytest.approx(mu)                       # unseen -> global mean
    assert UserMean(k=1).fit(train).predict(_df([("u2", "A", 0)]))[0] == pytest.approx((3 + mu) / 2)
    assert GlobalMean().fit(train).predict(train).tolist() == [mu] * 3


def test_bias_baseline_recovers_an_additive_structure():
    rng = np.random.default_rng(0)
    bu, bi = rng.uniform(-0.6, 0.6, 30), rng.uniform(-0.8, 0.8, 10)
    rows = [(f"u{u}", f"i{i}", 3.0 + bu[u] + bi[i]) for u in range(30) for i in range(10)]
    train = _df(rows)
    model = BiasBaseline(lam_user=0.01, lam_item=0.01, iters=30).fit(train)
    assert np.sqrt(np.mean((model.predict(train) - train["rating"]) ** 2)) < 0.02
    assert model.predict(_df([("new", "new", 0)]))[0] == pytest.approx(model.mu)  # cold ids add nothing


def test_item_knn_uses_correlated_neighbour_and_falls_back():
    pairs = [(1, 1), (2, 2), (4, 4), (5, 5)]  # items A and B always rated alike
    rows = [(f"u{k}", it, r) for k, (a, b) in enumerate(pairs) for it, r in (("A", a), ("B", b))]
    rows += [("fan", "A", 5)]                 # A mean is now 3.4; fan is +1.6 above it
    train = _df(rows)
    knn = ItemKNN(k=5, sim_shrinkage=0.0, mean_shrinkage=0.0).fit(train)
    pred = knn.predict(_df([("fan", "B", 0), ("nobody", "B", 0), ("fan", "unseen", 0)]))
    b_mean = train[train.product_id == "B"].rating.mean()
    assert pred[0] > b_mean + 1.0                       # neighbour's positive deviation carried over
    assert pred[1] == pytest.approx(b_mean)            # unknown user -> item mean
    assert pred[2] == pytest.approx(train.rating.mean())  # unknown item -> global mean


def test_asof_additive_ignores_the_present_and_future():
    df = make_interactions()
    T = df["submission_time"].quantile(0.5, interpolation="nearest")
    base = asof_additive(df, 3.8, 5, 10)
    d2 = df.copy()
    d2.loc[d2["submission_time"] >= T, "rating"] = 1
    other = asof_additive(d2, 3.8, 5, 10)
    upto = (df["submission_time"] <= T).to_numpy()
    np.testing.assert_allclose(base[upto], other[upto])
    assert not np.allclose(base[~upto], other[~upto])


def test_rating_report_buckets():
    y = np.array([5.0, 4.0, 3.0, 5.0])
    hist = np.array([0, 1, 3, 9])
    masks = history_bucket_masks(hist)
    assert [m.sum() for m in masks.values()] == [1, 1, 1, 1]
    rep = rating_report(y, {"perfect": y, "const4": np.full(4, 4.0)}, hist)
    assert rep.loc["perfect", "rmse"] == 0
    assert rep.loc["const4", "rmse [no history]"] == pytest.approx(1.0)


# ---------------------------------------------------------------- ranking
def test_rank_of_target_handles_ties_and_invalid_candidates():
    scores = np.array([[3.0, 1.0, 2.0, 2.0]])
    valid = np.ones((1, 4), dtype=bool)
    assert rank_of_target(scores, np.array([3]), valid)[0] == 2.5       # one above, one tie
    valid[0, 0] = False                                                  # remove the higher-scored item
    assert rank_of_target(scores, np.array([3]), valid)[0] == 1.5
    assert rank_of_target(np.zeros((1, 4)), np.array([2]), np.ones((1, 4), bool))[0] == 2.5  # all-equal: no free credit


def test_ranking_metrics_by_hand():
    m = ranking_metrics(np.array([1.0, 2.0, 12.0]), [10])
    assert m["hr@10"] == pytest.approx(2 / 3)
    assert m["ndcg@10"] == pytest.approx((1.0 + 1 / np.log2(3)) / 3)
    assert m["mrr"] == pytest.approx((1 + 0.5 + 1 / 12) / 3)


@pytest.fixture(scope="module")
def trained():
    df = make_interactions(n=900, n_users=60, n_items=15)
    labels, _ = temporal_split(df, 0.15, 0.25)
    cfg = small_cfg()
    cfg.train.device, cfg.train.epochs, cfg.train.batch_size = "cpu", 4, 64
    cfg.model.hidden, cfg.model.embed_dim = [16], 8
    fs = build_features(df, make_product_info(), labels, cfg)
    res = train_model(fs, cfg, verbose=False)
    inter = fs.meta[["author_id", "product_id", "submission_time"]].assign(rating=fs.y.astype("float64"))
    return cfg, fs, inter, res.model


def test_day_snapshot_item_features_equal_the_training_features(trained):
    cfg, fs, inter, _ = trained
    static = build_item_static(make_product_info())
    snaps = CatalogSnapshots(inter)
    days = inter["submission_time"].iloc[::37]
    item_tf = fs.artifacts["item_tf"]
    checked = 0
    for day, counts, sums, _launched in snaps.iter_days(days.unique()):
        X = item_tf.transform(item_frame_at(snaps.items, counts, sums, static, fs.mu, cfg.features.item_shrinkage))
        for row in np.where(inter["submission_time"].to_numpy() == day)[0][:3]:
            j = snaps.pos[inter["product_id"].iloc[row]]
            np.testing.assert_allclose(X[j], fs.X_item[row], atol=1e-5)
            checked += 1
    assert checked > 5


def test_evaluate_ranking_returns_sane_metrics_for_every_method(trained):
    cfg, fs, inter, model = trained
    static = build_item_static(make_product_info())
    out = evaluate_ranking(fs, inter, static, model, "test", threshold=4.0, ks=[5, 10], n_sample=200,
                           shrinkage=cfg.features.item_shrinkage, seed=0)
    assert {"popularity", "item mean rating", "two-tower", "random (expected)"} <= set(out)
    for name in ["popularity", "item mean rating", "two-tower", "random (expected)"]:
        for k, v in out[name].items():
            assert 0.0 <= v <= 1.0, (name, k, v)
    assert out["_mean_candidates"] > 1
    ranks = out["_ranks"]["two-tower"]
    assert ranks.min() >= 1 and ranks.max() <= out["_mean_candidates"] * 3 + 15
    assert out["random (expected)"]["hr@10"] >= out["random (expected)"]["hr@5"]


def test_catalog_embeddings_cover_every_item_and_match_the_tower(trained):
    from src.ranking import catalog_embeddings

    cfg, fs, inter, model = trained
    ids, emb = catalog_embeddings(fs, inter, build_item_static(make_product_info()), model, cfg.features.item_shrinkage)
    assert len(ids) == inter["product_id"].nunique() and emb.shape == (len(ids), model.init_args["embed_dim"])
    assert np.isfinite(emb).all()


def test_run_benchmark_and_write_results_end_to_end(trained, tmp_path):
    import json

    from src.benchmark import run_benchmark, write_results

    cfg, fs, _inter, model = trained
    csv = tmp_path / "product_info.csv"
    make_product_info().to_csv(csv, index=False)
    cfg.data.product_info = str(csv)
    res = run_benchmark(cfg, fs, model, n_rank_sample=100, verbose=False)

    for split in ("val", "test"):
        table = res["rating"][split]
        assert {"two-tower (PyTorch)", "global mean", "user+item bias", "item kNN (k=40)"} <= set(table.index)
        assert table["rmse"].between(0, 5).all()
        assert res["ranking"][split]["_meta"]["n_targets"] > 0
    write_results(res, tmp_path / "out")
    saved = json.loads((tmp_path / "out" / "results.json").read_text())
    assert set(saved["rating"]) == {"val", "test"}
    assert "two-tower" in (tmp_path / "out" / "results.md").read_text()
