"""Request/response models for the recommendation API."""

from typing import Optional

from pydantic import BaseModel, Field


class ReviewIn(BaseModel):
    """One review written by the user asking for recommendations."""

    rating: Optional[float] = Field(None, ge=1, le=5, description="Star rating, 1-5")
    is_recommended: Optional[bool] = Field(
        None, description="Whether the reviewer said they'd recommend the product"
    )
    helpfulness: Optional[float] = Field(
        None, ge=0, le=1, description="Fraction of readers who found the review helpful"
    )
    review_text: Optional[str] = Field(None, description="Free-text review body")
    skin_tone: Optional[str] = None
    skin_type: Optional[str] = None
    eye_color: Optional[str] = None
    hair_color: Optional[str] = None


class RecommendRequest(BaseModel):
    """A recommendation request for a single user.

    Leave ``reviews`` empty for a fully cold-start/anonymous user - the
    model falls back to training-time population averages for anything it
    can't infer from review history.
    """

    reviews: list[ReviewIn] = Field(default_factory=list)
    top_k: int = Field(10, ge=1, le=100, description="How many products to return")


class ProductRecommendation(BaseModel):
    product_id: str
    product_name: Optional[str] = None
    brand_name: Optional[str] = None
    price_usd: Optional[float] = None
    primary_category: Optional[str] = None
    pred_score: float = Field(..., description="Predicted rating, on the original 1-5 scale")


class RecommendResponse(BaseModel):
    recommendations: list[ProductRecommendation]
