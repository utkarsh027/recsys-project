# Production-Scale Recommendation System

A production-ready recommendation engine built with a two-tower neural network, full MLOps pipeline, A/B testing, and real-time serving — designed to mirror how recommendation systems work at companies like Amazon, Flipkart, and YouTube.

## Architecture

| Layer | Technology |(venv) (base) utkarsh@UTKARSHs-MacBook-Air-2 recsys-project % git commit -m "phase 2 complete: data ingestio
n and EDA"
[main e7b5c54] phase 2 complete: data ingestion and EDA
 7 files changed, 48 insertions(+)
 create mode 100644 monitoring/genre_distribution.png
 create mode 100644 monitoring/rating_distribution.png
 create mode 100644 monitoring/ratings_by_year.png
 create mode 100644 monitoring/ratings_per_user.png
 create mode 100644 monitoring/top_movies.png
 create mode 100644 notebooks/01_eda.ipynb
 create mode 100644 src/data_loader.py
(venv) (base) utkarsh@UTKARSHs-MacBook-Air-2 recsys-project % git push
To https://github.com/utkarsh027/recsys-project.git
 ! [rejected]        main -> main (fetch first)
error: failed to push some refs to 'https://github.com/utkarsh027/recsys-project.git'
hint: Updates were rejected because the remote contains work that you do not
hint: have locally. This is usually caused by another repository pushing to
hint: the same ref. If you want to integrate the remote changes, use
hint: 'git pull' before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.
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
