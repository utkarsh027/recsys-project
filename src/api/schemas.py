from pydantic import BaseModel, Field
from typing import List, Literal


class RecommendRequest(BaseModel):
    user_id: int = Field(..., ge=1, le=610, description="User ID between 1 and 610")
    model:   Literal["neural", "gb"] = Field("neural", description="Which model to use")
    top_k:   int = Field(10, ge=1, le=50, description="Number of recommendations")


class MovieRecommendation(BaseModel):
    rank:     int
    movie_id: int
    title:    str
    score:    float
    genres:   str


class RecommendResponse(BaseModel):
    user_id:         int
    model:           str
    top_k:           int
    recommendations: List[MovieRecommendation]


class HealthResponse(BaseModel):
    status:        str
    models_loaded: List[str]
    neural_auc:    float
    gb_auc:        float
    version:       str