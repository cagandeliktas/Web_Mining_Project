"""FastAPI serving layer for the two-tower recommender.

Run locally with:
    uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.inference import RecommenderService
from app.schemas import ProductRecommendation, RecommendRequest, RecommendResponse

_service: RecommenderService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    _service = RecommenderService()
    yield
    _service = None


app = FastAPI(
    title="Sephora Two-Tower Recommender",
    description="Serves recommendations from the trained two-tower neural network.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _service is not None}


@app.post("/recommendations", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    recommendations: list[ProductRecommendation] = _service.recommend(
        request.reviews, request.top_k
    )
    return RecommendResponse(recommendations=recommendations)
