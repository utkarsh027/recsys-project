# Production-Scale Recommendation System

A production-ready recommendation engine built with a two-tower neural network, full MLOps pipeline, A/B testing, and real-time serving — designed to mirror how recommendation systems work at companies like Amazon, Flipkart, and YouTube.

## Architecture

| Layer | Technology |
|---|---|
| Model | Two-tower neural network (TensorFlow Recommenders) |
| Experiment tracking | MLflow |
| API serving | FastAPI + Uvicorn |
| Containerisation | Docker |
| Drift monitoring | Evidently AI |
| Dataset | MovieLens Small (100k ratings) |

## Project Structure

```
recsys-project/
├── data/
│   ├── raw/              # Downloaded datasets
│   └── processed/        # Cleaned and transformed data
├── notebooks/
│   └── 01_eda.ipynb      # Exploratory data analysis
├── src/
│   ├── data_loader.py    # Dataset download and path management
│   ├── features/         # Feature engineering pipeline
│   ├── models/           # Two-tower model training
│   └── api/              # FastAPI serving layer
└── monitoring/           # EDA plots and Evidently AI drift reports
```

## Progress

| Phase | Status | Description |
|---|---|---|
| Phase 1 | ✅ Complete | Project setup, virtual environment, dependencies, Git |
| Phase 2 | ✅ Complete | Data ingestion, EDA, 5 visualisations, modelling decisions |
| Phase 3 | 🔄 Up next | Feature engineering and two-tower neural model |
| Phase 4 | ⏳ Pending | A/B test: matrix factorisation vs two-tower model |
| Phase 5 | ⏳ Pending | FastAPI serving and Docker containerisation |
| Phase 6 | ⏳ Pending | Drift monitoring with Evidently AI |

## EDA Key Findings

Analysis of 100,836 ratings from 610 users across 9,724 movies revealed five critical insights that directly shaped modelling decisions:

| Finding | Data Evidence | Modelling Decision |
|---|---|---|
| Extreme sparsity | 98.3% of interaction matrix is empty | Embedding-based two-tower model over nearest-neighbour |
| Popularity bias | Top 15 rated movies are all pre-2000 classics | Two-tower over pure collaborative filtering |
| Power law activity | Users range from 20 to 2,698 ratings | Stratified train/test split by user activity |
| Temporal drift | Ratings span 1996–2018 (22 years) | Timestamp encoded as recency feature |
| Multi-label genres | 20 genres, pipe-separated per movie | Explode and embed genres, not one-hot encode |

## Today's Progress — Phase 2

**Date:** 2 July 2026

**What was built:**
- `src/data_loader.py` — idempotent download pipeline with automatic extraction and cleanup
- `notebooks/01_eda.ipynb` — full exploratory analysis across ratings, users, movies, genres, and time
- 5 production-quality visualisations saved to `monitoring/`

**Dataset statistics:**

```
Total ratings:        100,836
Unique users:         610
Unique movies:        9,724
Sparsity:             98.30%
Average rating:       3.50 / 5.0
Median ratings/user:  70
Date range:           1996–2018
Most common rating:   4.0
Total genres:         20
```

**Visualisations generated:**
- Rating distribution — confirmed selection bias and whole-number preference
- Ratings per user — confirmed power law distribution
- Genre distribution — confirmed popularity imbalance across 20 genres
- Ratings by year — confirmed temporal drift across 22 years
- Top 15 most rated movies — confirmed popularity bias toward older classics
