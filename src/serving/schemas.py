"""Request / response models for the API.

In plain words: The exact shape of the requests and responses of the web API (input is validated
automatically).
"""

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
    """What we know about the user: their earlier reviews and an optional profile."""
    reviews: list[ReviewIn] = Field(default_factory=list, max_length=500,
                                    description="The user's earlier reviews; empty for a new/anonymous user.")
    profile: Optional[Profile] = Field(default=None, description="Overrides demographics taken from the reviews.")


class RecommendationRequest(UserContext):
    """Input of POST /recommendations: the user context plus how many results and which products to
    skip.
    """
    top_k: int = Field(default=10, ge=1, le=100)
    exclude_product_ids: list[str] = Field(default_factory=list, max_length=5000)
    min_item_reviews: int = Field(default=0, ge=0, description="Skip products with fewer reviews in the snapshot.")


class ProductRecommendation(BaseModel):
    """One recommended product with its name, brand, price, category and predicted rating."""
    product_id: str
    product_name: Optional[str] = None
    brand_name: Optional[str] = None
    price_usd: Optional[float] = None
    primary_category: Optional[str] = None
    predicted_rating: float


class RecommendationResponse(BaseModel):
    """Output of POST /recommendations: the list of products and some bookkeeping."""
    n_reviews_used: int
    catalog_snapshot: str
    recommendations: list[ProductRecommendation]


class PredictRequest(UserContext):
    """Input of POST /predict: the user context plus the product to score."""
    product_id: str


class PredictResponse(BaseModel):
    """Output of POST /predict: the predicted rating."""
    product_id: str
    predicted_rating: float


class HealthResponse(BaseModel):
    """Output of GET /health."""
    status: str
    n_products: int
    catalog_snapshot: str
