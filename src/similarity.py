"""Nearest-neighbour lookup over item embeddings learned by the two-tower model.

Extracted from NeuralNetworkApproach/NN_w_textFeatures.ipynb. The notebook version
of this function reads `prod_ids` and `vm_all` from notebook globals; here they are
explicit parameters so the function can be unit tested in isolation. The distance
computation and tie-breaking logic are unchanged.
"""
import numpy as np
import pandas as pd


def most_similar_items(query_prod_id, prod_ids, vm_all, top_k=5):
    """Find the top_k items most similar to query_prod_id by embedding distance.

    Parameters
    ----------
    query_prod_id : the product id to look up.
    prod_ids : ndarray of shape (n_items,), product ids aligned with the rows of vm_all.
    vm_all : ndarray of shape (n_items, embed_dim), item embeddings from the trained model.
    top_k : int, number of neighbours to return.

    Returns
    -------
    DataFrame with columns product_id and sq_distance, ordered by increasing distance,
    excluding the query item itself and any duplicate product ids.
    """
    try:
        q_idx = np.where(prod_ids == query_prod_id)[0][0]
    except IndexError:
        raise ValueError(f"Product ID '{query_prod_id}' not found!")

    q_vec = vm_all[q_idx]  # shape (embed_dim,)
    dists = (np.sum(vm_all**2, axis=1) +
             np.sum(q_vec**2) -
             2 * vm_all @ q_vec)

    # Get top_k closest items (excluding the query item itself)
    nearest = np.argsort(dists)
    nearest = nearest[nearest != q_idx]  # exclude self

    # ensure unique product_ids in result (just in case)
    seen = set()
    unique_nearest = []
    for idx in nearest:
        pid = prod_ids[idx]
        if pid not in seen:
            seen.add(pid)
            unique_nearest.append(idx)
        if len(unique_nearest) == top_k:
            break

    return pd.DataFrame({
        "product_id": prod_ids[unique_nearest],
        "sq_distance": dists[unique_nearest]
    })
