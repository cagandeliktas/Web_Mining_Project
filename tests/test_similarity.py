import numpy as np
import pytest

from src.similarity import most_similar_items

# Four items on a simple 2D embedding grid: B and C are both distance 1 from A,
# D is far away.
PROD_IDS = np.array(["A", "B", "C", "D"])
EMBEDDINGS = np.array(
    [
        [0.0, 0.0],  # A
        [1.0, 0.0],  # B
        [0.0, 1.0],  # C
        [5.0, 5.0],  # D
    ]
)


def test_most_similar_items_excludes_the_query_item():
    result = most_similar_items("A", PROD_IDS, EMBEDDINGS, top_k=3)
    assert "A" not in result["product_id"].values


def test_most_similar_items_orders_by_increasing_distance():
    result = most_similar_items("A", PROD_IDS, EMBEDDINGS, top_k=3)
    assert list(result["product_id"]) == ["B", "C", "D"]
    assert result["sq_distance"].is_monotonic_increasing


def test_most_similar_items_respects_top_k():
    result = most_similar_items("A", PROD_IDS, EMBEDDINGS, top_k=2)
    assert len(result) == 2
    assert set(result["product_id"]) == {"B", "C"}


def test_most_similar_items_raises_on_unknown_product_id():
    with pytest.raises(ValueError, match="not found"):
        most_similar_items("does-not-exist", PROD_IDS, EMBEDDINGS, top_k=2)
