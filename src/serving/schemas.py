"""Request / response models for the API."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Profile(BaseModel):
    """Self-reported attributes; any field may be omitted."""

    skin_tone: Optional[str] = None
    skin_type: Optional[str] = None
    eye_color: Optional[str] = None
    hair_color: Optional[str] = None


class ReviewIn(Profile):
    """One earlier review by the user. Only ``rating`` is required."""

    rating: float = Field(ge=1, le=5)
    is_recommended: Optional[bool] = None
    review_text: Optional[str] = Field(default=None, max_length=20000)
    product_id: Optional[str] = Field(default=None, description="If given, this product is excluded from recommendations.")
    submission_time: Optional[datetime] = None


class UserContext(BaseModel):
    reviews: list[ReviewIn] = Field(default_factory=list, max_length=500,
                                    description="The user's earlier reviews; empty for a new/anonymous user.")
    profile: Optional[Profile] = Field(default=None, description="Overrides demographics taken from the reviews.")


class RecommendationRequest(UserContext):
    top_k: int = Field(default=10, ge=1, le=100)
    exclude_product_ids: list[str] = Field(default_factory=list, max_length=5000)
    min_item_reviews: int = Field(default=0, ge=0, description="Skip products with fewer reviews in the snapshot.")


class ProductRecommendation(BaseModel):
    product_id: str
    product_name: Optional[str] = None
    brand_name: Optional[str] = None
    price_usd: Optional[float] = None
    primary_category: Optional[str] = None
    predicted_rating: float


class RecommendationResponse(BaseModel):
    n_reviews_used: int
    catalog_snapshot: str
    recommendations: list[ProductRecommendation]


class PredictRequest(UserContext):
    product_id: str


class PredictResponse(BaseModel):
    product_id: str
    predicted_rating: float


class HealthResponse(BaseModel):
    status: str
    n_products: int
    catalog_snapshot: str
