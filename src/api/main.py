import sys
import os
sys.path.append(os.path.abspath("."))

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from src.api.schemas import (
    RecommendRequest, RecommendResponse,
    HealthResponse, MovieRecommendation
)
from src.api.recommender import Recommender

recommender = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    print("Loading models into memory...")
    recommender = Recommender()
    print("API ready to serve requests.")
    yield
    print("Shutting down API.")


app = FastAPI(
    title="Movie Recommendation API",
    description="Two-tower neural network + gradient boosting recommendation system. Built with PyTorch, FastAPI, and Docker.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "message":   "Movie Recommendation API is live",
        "docs":      "/docs",
        "health":    "/health",
        "recommend": "POST /recommend",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        models_loaded=["neural", "gb"],
        neural_auc=0.8211,
        gb_auc=0.8403,
        version="1.0.0",
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    results = recommender.recommend(
        user_id=request.user_id,
        model=request.model,
        top_k=request.top_k,
    )

    if results is None:
        raise HTTPException(
            status_code=404,
            detail=f"User {request.user_id} not found in dataset"
        )

    return RecommendResponse(
        user_id=request.user_id,
        model=request.model,
        top_k=request.top_k,
        recommendations=[MovieRecommendation(**r) for r in results],
    )