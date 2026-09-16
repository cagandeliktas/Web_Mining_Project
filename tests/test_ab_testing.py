import numpy as np
import pytest

from src.ab_testing import compare_variants, prediction_errors


def test_prediction_errors_is_absolute_difference():
    errors = prediction_errors([5, 3, 1], [4, 3, 5])
    np.testing.assert_allclose(errors, [1, 0, 4])


def test_prediction_errors_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        prediction_errors([1, 2, 3], [1, 2])


def test_compare_variants_detects_a_clearly_better_variant():
    rng = np.random.default_rng(0)
    # variant B's errors are consistently ~0.5 lower than A's, same noise
    errors_a = np.abs(rng.normal(loc=1.0, scale=0.2, size=200))
    errors_b = np.abs(errors_a - 0.5 + rng.normal(loc=0.0, scale=0.05, size=200))

    result = compare_variants(errors_a, errors_b)

    assert result["better_variant"] == "b"
    assert result["significant_at_alpha"] is True
    assert result["ttest_pvalue"] < 0.05
    assert result["wilcoxon_pvalue"] < 0.05
    assert result["mean_error_b"] < result["mean_error_a"]


def test_compare_variants_finds_no_difference_for_identical_errors():
    rng = np.random.default_rng(1)
    errors = np.abs(rng.normal(loc=1.0, scale=0.3, size=100))

    result = compare_variants(errors, errors.copy())

    assert result["better_variant"] == "tie"
    assert result["significant_at_alpha"] is False
    assert result["ttest_pvalue"] == pytest.approx(1.0)
    assert result["wilcoxon_pvalue"] == pytest.approx(1.0)


def test_compare_variants_finds_no_significant_difference_for_matched_noise():
    rng = np.random.default_rng(2)
    # Same distribution, independent noise - no real difference expected
    errors_a = np.abs(rng.normal(loc=1.0, scale=0.3, size=60))
    errors_b = np.abs(rng.normal(loc=1.0, scale=0.3, size=60))

    result = compare_variants(errors_a, errors_b)

    assert result["significant_at_alpha"] is False


def test_compare_variants_rejects_unpaired_shapes():
    with pytest.raises(ValueError):
        compare_variants([1, 2, 3], [1, 2])


def test_compare_variants_requires_at_least_two_instances():
    with pytest.raises(ValueError):
        compare_variants([1.0], [1.0])
