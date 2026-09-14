"""Ranking and classification metrics for the two-tower recommender.

Extracted from NeuralNetworkApproach/NN_w_textFeatures.ipynb so the logic can be
unit tested and reused outside the notebook.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, ndcg_score, precision_score


def precision_recall_f1_nn_vectorized(user_ids_test, y_test_orig, y_pred_orig, threshold=3.5):
    """
    Computes average precision, recall, F1, and accuracy across users in a vectorized manner.

    Parameters:
      - user_ids_test: array-like, user IDs for each test record.
      - y_test_orig: array-like, true ratings in the original scale.
      - y_pred_orig: array-like, predicted ratings in the original scale.
      - threshold: float, threshold above which a rating is considered "positive".

    Returns:
      avg_precision, avg_recall, avg_f1, avg_accuracy
    """

    y_true_bin = (np.array(y_test_orig).flatten() >= threshold).astype(int)
    y_pred_bin = (np.array(y_pred_orig).flatten() >= threshold).astype(int)

    df = pd.DataFrame({
        'user': user_ids_test,
        'true': y_true_bin,
        'pred': y_pred_bin
    })

    # Group by user and compute aggregated statistics:
    # n_tp: number of true positives
    # n_pred: total predicted positives
    # n_true: total actual positives
    # n_total: total number of ratings for the user
    # n_correct: total correct predictions
    grouped = df.groupby('user').agg(
        n_tp=('true', lambda x: np.sum(x * df.loc[x.index, 'pred'])),
        n_pred=('pred', 'sum'),
        n_true=('true', 'sum'),
        n_total=('true', 'size'),
        n_correct=('true', lambda x: np.sum(x == df.loc[x.index, 'pred']))
    )

    grouped['precision'] = grouped['n_tp'] / grouped['n_pred'].replace(0, np.nan)
    grouped['recall'] = grouped['n_tp'] / grouped['n_true'].replace(0, np.nan)
    grouped['accuracy'] = grouped['n_correct'] / grouped['n_total']
    grouped['f1'] = 2 * (grouped['precision'] * grouped['recall']) / (grouped['precision'] + grouped['recall'])

    grouped = grouped.fillna(0)

    avg_precision = grouped['precision'].mean()
    avg_recall = grouped['recall'].mean()
    avg_f1 = grouped['f1'].mean()
    avg_accuracy = grouped['accuracy'].mean()

    return avg_precision, avg_recall, avg_f1, avg_accuracy


def measures_at_k_nn(user_ids_test, y_test_orig, y_pred_orig, k=10, threshold=3.5):
    """
    Computes average average_precision, precision@k, and nDCG@k across users
    for predictions from a two-tower model.

    Parameters:
      - user_ids_test: array-like of user IDs (one per test record)
      - y_test_orig: array-like of true ratings (original scale)
      - y_pred_orig: array-like of predicted ratings (original scale)
      - k: int, the number of top items to consider for "at k" metrics
      - threshold: float, the rating value at or above which an item is considered relevant

    Returns:
      avg_average_precision, avg_precision_at_k, avg_ndcg_at_k
    """
    user_ids_test = np.array(user_ids_test).flatten()
    y_test_orig = np.array(y_test_orig).flatten()
    y_pred_orig = np.array(y_pred_orig).flatten()

    df = pd.DataFrame({
        'user': user_ids_test,
        'true': y_test_orig,
        'pred': y_pred_orig
    })

    df['true_bin'] = (df['true'] >= threshold).astype(int)
    df['pred_bin'] = (df['pred'] >= threshold).astype(int)

    user_est_true = df.groupby('user')

    average_precisions = {}
    precisions_at_k = {}
    ndcgs_at_k = {}

    for uid, group in user_est_true:
        # Convert each user's group to a list of tuples: (predicted rating, true rating, binary true)
        # Sort the user's items by predicted rating in descending order.
        sorted_group = group.sort_values(by='pred', ascending=False)
        y_true = sorted_group['true_bin'].tolist()
        y_pred = sorted_group['pred_bin'].tolist()

        y_true_at_k = y_true[:k]
        y_pred_at_k = y_pred[:k]

        # Compute average precision over the full ranked list if the user has any relevant items.
        if sum(y_true) > 0:
            average_precisions[uid] = average_precision_score(sorted_group['true_bin'], sorted_group['pred_bin'])
        else:
            average_precisions[uid] = 0.0

        # Compute precision at k.
        precisions_at_k[uid] = precision_score(y_true_at_k, y_pred_at_k, zero_division=0)

        # For nDCG, raw ratings are needed and consider the ranking.
        # nDCG expects input arrays of shape (1, n); convert the sorted true and predicted ratings.
        if len(sorted_group) > 1:
            true_rel = np.asarray(sorted_group['true'].tolist()).reshape(1, -1)
            pred_rel = np.asarray(sorted_group['pred'].tolist()).reshape(1, -1)
            ndcgs_at_k[uid] = ndcg_score(true_rel, pred_rel, k=k)
        else:
            # If there's only one item, then return a simple binary match.
            ndcgs_at_k[uid] = 1.0 if (y_true[0] == y_pred[0] and y_true[0] == 1) else 0.0

    # Average metrics across all users.
    avg_average_precision = np.mean(list(average_precisions.values()))
    avg_precision_at_k = np.mean(list(precisions_at_k.values()))
    avg_ndcg_at_k = np.mean(list(ndcgs_at_k.values()))

    return avg_average_precision, avg_precision_at_k, avg_ndcg_at_k
