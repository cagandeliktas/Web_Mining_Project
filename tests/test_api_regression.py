"""End-to-end tests for the FastAPI service.

These need the real model (TensorFlow), the full artifact set, and
vaderSentiment - see requirements-api.txt. Marked ``api`` so the fast,
dependency-light default test run (tests.yml) skips them; a separate
workflow (api-tests.yml) installs requirements-api.txt and runs
``pytest -m api``.

Exact top-K rankings aren't pinned as golden values here: the model's
top candidates are separated by scores as tight as ~0.001-0.01 (on a
1-5 scale), which is well within the range that can flip order across
different TensorFlow builds/CPU architectures. Instead these tests check
the invariants that would actually catch a real regression (wrong
input shape, broken scaling, swapped user/item order, non-determinism).
"""

import pytest

pytestmark = pytest.mark.api

# importorskip (not a plain import) so the fast, dependency-light default
# test run can still *collect* this file without fastapi/tensorflow
# installed - it skips the whole module instead of erroring out.
TestClient = pytest.importorskip("fastapi.testclient").TestClient
app = pytest.importorskip("app.main").app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_recommend_respects_top_k(client):
    r = client.post("/recommendations", json={"reviews": [], "top_k": 7})
    assert r.status_code == 200
    assert len(r.json()["recommendations"]) == 7


def test_recommend_scores_are_sane_and_sorted(client):
    r = client.post("/recommendations", json={"reviews": [], "top_k": 20})
    recs = r.json()["recommendations"]

    scores = [rec["pred_score"] for rec in recs]
    assert scores == sorted(scores, reverse=True)
    assert all(3.0 <= s <= 5.0 for s in scores)

    ids = [rec["product_id"] for rec in recs]
    assert len(set(ids)) == len(ids)

    for rec in recs:
        assert rec["product_name"]
        assert rec["brand_name"]


def test_recommend_is_deterministic(client):
    payload = {"reviews": [], "top_k": 10}
    r1 = client.post("/recommendations", json=payload).json()
    r2 = client.post("/recommendations", json=payload).json()
    assert r1 == r2


def test_reviewed_user_differs_from_anonymous_user(client):
    anon = client.post("/recommendations", json={"reviews": [], "top_k": 10}).json()
    reviewed = client.post(
        "/recommendations",
        json={
            "reviews": [
                {
                    "rating": 5,
                    "is_recommended": True,
                    "helpfulness": 0.9,
                    "review_text": "Loved it, great texture!",
                    "skin_tone": "medium",
                    "skin_type": "oily",
                    "eye_color": "brown",
                    "hair_color": "black",
                }
            ],
            "top_k": 10,
        },
    ).json()

    assert anon != reviewed


def test_recommend_rejects_out_of_range_top_k(client):
    assert client.post("/recommendations", json={"reviews": [], "top_k": 0}).status_code == 422
    assert client.post("/recommendations", json={"reviews": [], "top_k": 1000}).status_code == 422


def test_recommend_rejects_invalid_rating(client):
    r = client.post("/recommendations", json={"reviews": [{"rating": 9}], "top_k": 5})
    assert r.status_code == 422
