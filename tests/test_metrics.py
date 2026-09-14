import pytest

from src.metrics import measures_at_k_nn, precision_recall_f1_nn_vectorized

# Two users, five ratings, threshold 3.5:
#   user 1: true [5, 4, 2] (relevant, relevant, not-relevant)
#            pred [4, 4, 3] (relevant, relevant, not-relevant) -> all 3 correct
#   user 2: true [5, 1]    (relevant, not-relevant)
#            pred [2, 1]   (not-relevant, not-relevant) -> 1 of 2 correct
USER_IDS = [1, 1, 1, 2, 2]
Y_TRUE = [5, 4, 2, 5, 1]
Y_PRED = [4, 4, 3, 2, 1]


def test_precision_recall_f1_nn_vectorized_matches_hand_computed_values():
    precision, recall, f1, accuracy = precision_recall_f1_nn_vectorized(
        USER_IDS, Y_TRUE, Y_PRED, threshold=3.5
    )
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)
    assert accuracy == pytest.approx(0.75)


def test_precision_recall_f1_nn_vectorized_handles_no_positive_predictions():
    # Neither user ever predicts or has a "relevant" rating -> precision/recall/f1
    # fall back to 0 (via the fillna(0) in the implementation) rather than raising
    # on a division by zero, and accuracy is 1.0 since every negative is correct.
    precision, recall, f1, accuracy = precision_recall_f1_nn_vectorized(
        [1, 1], [1, 2], [1, 2], threshold=3.5
    )
    assert (precision, recall, f1, accuracy) == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_measures_at_k_nn_matches_hand_computed_values():
    avg_ap, avg_prec_at_k, avg_ndcg_at_k = measures_at_k_nn(
        USER_IDS, Y_TRUE, Y_PRED, k=2, threshold=3.5
    )
    assert avg_ap == pytest.approx(0.75)
    assert avg_prec_at_k == pytest.approx(0.5)
    assert avg_ndcg_at_k == pytest.approx(0.9877364423854824)


def test_measures_at_k_nn_handles_users_with_a_single_rating():
    # Each user has exactly one (user, item) pair, exercising the len(sorted_group) == 1
    # branch that skips the sklearn ndcg_score call.
    avg_ap, avg_prec_at_k, avg_ndcg_at_k = measures_at_k_nn(
        [1, 2], [5, 1], [4, 1], k=1, threshold=3.5
    )
    assert (avg_ap, avg_prec_at_k, avg_ndcg_at_k) == pytest.approx((0.5, 0.5, 0.5))
