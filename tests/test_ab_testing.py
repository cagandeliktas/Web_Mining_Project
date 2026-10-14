"""Tests the statistics: a better method is detected, ties give p=1, user-clustering widens the
uncertainty.
"""
import numpy as np
import pytest

from src.ab_testing import cluster_bootstrap, compare_methods, holm_adjust, row_level_tests


def _data(n_users=300, per_user=4, seed=0, noise_b=0.8, noise_a=1.0):
    rng = np.random.default_rng(seed)
    users = np.repeat(np.arange(n_users), per_user)
    y = rng.integers(1, 6, len(users)).astype(float)
    return y, y + rng.normal(0, noise_a, len(y)), y + rng.normal(0, noise_b, len(y)), users


def test_a_clearly_better_challenger_is_detected_by_every_test():
    y, pa, pb, users = _data()
    r = compare_methods(y, pa, pb, users, n_boot=400)
    assert r["delta_rmse"] > 0 and r["delta_rmse_ci"][0] > 0 and r["delta_rmse_p"] < 0.05
    assert r["ttest_sq_error_p"] < 0.05 and r["wilcoxon_abs_error_p"] < 0.05


def test_identical_predictions_give_zero_delta_and_p_one():
    y, pa, _, users = _data()
    r = compare_methods(y, pa, pa.copy(), users, n_boot=200)
    assert r["delta_rmse"] == 0 and r["delta_mae"] == 0
    assert r["ttest_sq_error_p"] == 1.0 and r["wilcoxon_abs_error_p"] == 1.0
    assert r["delta_rmse_ci"] == [0.0, 0.0]


def test_ci_brackets_the_point_estimate_and_flips_sign_when_roles_swap():
    y, pa, pb, users = _data(seed=3, noise_b=0.95)
    r, swapped = cluster_bootstrap(y, pa, pb, users, 400), cluster_bootstrap(y, pb, pa, users, 400)
    assert r["delta_rmse_ci"][0] <= r["delta_rmse"] <= r["delta_rmse_ci"][1]
    assert swapped["delta_rmse"] == pytest.approx(-r["delta_rmse"])


def test_clustering_widens_the_interval_when_a_users_rows_share_an_error():
    rng = np.random.default_rng(1)
    users = np.repeat(np.arange(200), 10)
    y = rng.integers(1, 6, len(users)).astype(float)
    shared = np.repeat(rng.normal(0, 0.6, 200), 10)  # one error shared by all of a user's rows
    pa = y + shared + rng.normal(0, 0.2, len(y))
    pb = y + 0.5 * shared + rng.normal(0, 0.2, len(y))
    clustered = cluster_bootstrap(y, pa, pb, users, 600)
    naive = cluster_bootstrap(y, pa, pb, None, 600)
    width = lambda r: r["delta_rmse_ci"][1] - r["delta_rmse_ci"][0]
    assert width(clustered) > 1.5 * width(naive)


def test_validation_errors():
    with pytest.raises(ValueError):
        row_level_tests([1, 2, 3], [1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        cluster_bootstrap([1.0], [1.0], [1.0])
    with pytest.raises(ValueError):
        cluster_bootstrap([1, 2, 3], [1, 2, 3], [1, 2, 3], groups=[0, 1])


def test_holm_adjust_matches_the_textbook_example():
    adj = holm_adjust([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])
    assert np.all(adj >= np.array([0.01, 0.04, 0.03, 0.005])) and adj.max() <= 1.0
