"""FastAPI service for the PyTorch two-tower model.

In plain words: The web API (FastAPI): three endpoints that return recommendations and rating
predictions for a user.

    python -m scripts.export_serving_bundle          # once, after training
    uvicorn src.serving.api:app --reload             # RECSYS_BUNDLE overrides artifacts/serving

Ranking is by predicted *rating*. That is what the model is trained for; as the
evaluation in README.md shows, it is a weak retrieval signal, so treat
/recommendations as "products this user is predicted to rate highest", and
/predict (a rating for one product) as the model's stronger use.
"""

import math
import os
from contextlib import asynccontextmanager
from typing import Callable, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from src.serving.bundle import ServingBundle
from src.serving.recommender import Recommender
from src.serving.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ProductRecommendation,
    RecommendationRequest,
    RecommendationResponse,
    UserContext,
)

DEFAULT_BUNDLE_DIR = "artifacts/serving"


def _history(ctx: UserContext) -> Optional[pd.DataFrame]:
    """Turn the request's reviews into a table, or None for a new user with no reviews."""
    if not ctx.reviews:
        return None
    return pd.DataFrame([r.model_dump() for r in ctx.reviews])


def _profile(ctx: UserContext) -> Optional[dict]:
    """Turn the optional skin/eye/hair profile into a plain dict, or None."""
    return ctx.profile.model_dump() if ctx.profile else None


def _clean(value):
    """NaN -> None so the JSON is valid."""
    return None if isinstance(value, float) and math.isnan(value) else value


def create_app(bundle_loader: Optional[Callable[[], ServingBundle]] = None) -> FastAPI:
    """Build the web app: load the bundle at startup and register the three endpoints. A different
    bundle loader can be passed in (used by tests).
    """
    loader = bundle_loader or (lambda: ServingBundle.load(os.environ.get("RECSYS_BUNDLE", DEFAULT_BUNDLE_DIR)))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Startup hook: load the model bundle once when the server starts (fails fast if it is
        missing).
        """
        app.state.recommender = Recommender(loader())  # fail fast if the bundle is missing
        yield

    app = FastAPI(title="Sephora two-tower recommender (PyTorch)", version="1.0", lifespan=lifespan)

    def rec(request: Request) -> Recommender:
        """Fetch the loaded Recommender from the app's state."""
        return request.app.state.recommender

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        """GET /health: is the server up, how many products it knows and which date its catalogue
        snapshot is from.
        """
        b = rec(request).bundle
        return HealthResponse(status="ok", n_products=len(b.item_ids), catalog_snapshot=b.snapshot_day)

    @app.post("/recommendations", response_model=RecommendationResponse)
    def recommendations(body: RecommendationRequest, request: Request):
        """POST /recommendations: the top-K products by predicted rating for this user."""
        r = rec(request)
        table = r.recommend(_history(body), _profile(body), body.top_k, body.exclude_product_ids, body.min_item_reviews)
        items = [
            ProductRecommendation(**{k: _clean(row[k]) for k in ProductRecommendation.model_fields})
            for row in table.to_dict("records")
        ]
        return RecommendationResponse(n_reviews_used=len(body.reviews), catalog_snapshot=r.bundle.snapshot_day,
                                      recommendations=items)

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, request: Request):
        """POST /predict: the predicted rating of one product for this user (404 if the product is
        unknown).
        """
        try:
            score = rec(request).predict(body.product_id, _history(body), _profile(body))
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown product_id '{body.product_id}'")
        return PredictResponse(product_id=body.product_id, predicted_rating=score)

    return app


app = create_app()
