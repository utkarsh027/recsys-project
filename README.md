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
| Dataset | MovieLens 25M |

## Project Structure
recsys-project/

├── data/          # Raw and processed datasets

├── notebooks/     # EDA and experimentation

├── src/

│   ├── features/  # Feature engineering pipeline

│   ├── models/    # Two-tower model training

│   └── api/       # FastAPI serving layer

└── monitoring/    # Evidently AI drift reports
## Phases

- [x] Phase 1 — Project setup and environment
- [ ] Phase 2 — Data ingestion and EDA
- [ ] Phase 3 — Feature engineering and two-tower model
- [ ] Phase 4 — A/B test: baseline vs neural model
- [ ] Phase 5 — FastAPI serving and Docker
- [ ] Phase 6 — Drift monitoring with Evidently AI

## Key Results

*To be updated as each phase is completed.*
