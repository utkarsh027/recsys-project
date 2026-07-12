# Production-Scale Movie Recommendation System

<div align="center">

An end-to-end production ML system that trains, compares, serves, and monitors two recommendation models — a **Gradient Boosting baseline** and a **Two-Tower Neural Network** — mirroring how recommendation systems work at Amazon, Flipkart, and YouTube.

**AUC 0.84 · NDCG@10 improvement +3,195% · p=0.0000 · Served via Docker · Drift monitored**

</div>

---

## Quick Start

```bash
git clone https://github.com/utkarsh027/recsys-project.git
cd recsys-project
docker-compose up --build
```

```bash
# Health check
curl http://localhost:8000/health

# Get top-5 movie recommendations for User 1
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "model": "neural", "top_k": 5}'
```

**Response:**
```json
{
  "user_id": 1,
  "model": "neural",
  "top_k": 5,
  "recommendations": [
    {"rank": 1, "title": "Magnolia (1999)",             "score": 0.9108, "genres": "Drama"},
    {"rank": 2, "title": "Gran Torino (2008)",          "score": 0.9103, "genres": "Crime|Drama"},
    {"rank": 3, "title": "Terms of Endearment (1983)",  "score": 0.9101, "genres": "Comedy|Drama"},
    {"rank": 4, "title": "Producers, The (1968)",       "score": 0.9099, "genres": "Comedy"},
    {"rank": 5, "title": "Doctor Zhivago (1965)",       "score": 0.9099, "genres": "Drama|Romance|War"}
  ]
}
```

Interactive docs at **http://localhost:8000/docs**

---

## Architecture

| Layer | Technology | Purpose |
|---|---|---|
| Neural model | PyTorch + Apple MPS GPU | Two-tower embedding architecture |
| Baseline model | scikit-learn GradientBoosting | Strong tabular baseline for comparison |
| Feature engineering | pandas · numpy · multi-hot encoding | User stats, genre vectors, recency |
| Experiment tracking | MLflow | Every run logged with metrics and artifacts |
| API serving | FastAPI + Uvicorn + Pydantic | Input validation, health checks, JSON serving |
| Containerisation | Docker + Docker Compose | One-command deployment anywhere |
| Drift monitoring | Evidently AI | Feature distribution shift detection |
| Dataset | MovieLens Small | 100,836 ratings · 610 users · 9,724 movies |

---

## Project Structure

```
recsys-project/
├── data/
│   ├── raw/
│   │   └── ml-latest-small/          # Downloaded MovieLens dataset
│   ├── processed/
│   │   ├── interactions.parquet       # 100,836 × 15 interaction features
│   │   ├── movie_features.parquet     # Movie stats
│   │   ├── user_features.parquet      # User behavioural stats
│   │   ├── genre_matrix.npy           # 9,742 × 20 multi-hot genre vectors
│   │   └── genre_vocab.json           # Genre index mapping
│   └── models/
│       ├── gb_model.pkl               # Trained gradient boosting model
│       ├── scaler.pkl                 # StandardScaler for GB features
│       └── two_tower_final.pt         # Best PyTorch model weights (epoch 6)
├── notebooks/
│   ├── 01_eda.ipynb                   # Full exploratory data analysis
│   └── 02_model.ipynb                 # Model experimentation
├── src/
│   ├── data_loader.py                 # Idempotent dataset download + extraction
│   ├── features/
│   │   └── feature_engineering.py    # Complete feature pipeline
│   ├── models/
│   │   ├── two_tower.py               # PyTorch two-tower architecture (704,929 params)
│   │   ├── train.py                   # Gradient boosting training + MLflow
│   │   ├── train_neural.py            # Neural training on MPS GPU + MLflow
│   │   └── ab_test.py                 # A/B test: Precision@K, NDCG@10, t-test
│   └── api/
│       ├── main.py                    # FastAPI application with lifespan
│       ├── schemas.py                 # Pydantic request/response validation
│       └── recommender.py             # Model loading and inference engine
├── monitoring/
│   ├── drift_monitor.py               # Evidently AI drift detection script
│   ├── drift_report.html              # Generated HTML drift report
│   ├── ab_test_results.csv            # Per-user A/B test scores (465 users)
│   ├── ab_test_comparison.png         # Metric comparison bar chart
│   ├── ab_test_ndcg_dist.png          # NDCG@10 distribution per user
│   ├── rating_distribution.png        # EDA: rating distribution
│   ├── ratings_per_user.png           # EDA: power law user activity
│   ├── genre_distribution.png         # EDA: genre imbalance
│   ├── ratings_by_year.png            # EDA: temporal drift
│   └── top_movies.png                 # EDA: popularity bias
├── Dockerfile                         # Container definition
├── docker-compose.yml                 # One-command deployment
└── requirements_api.txt               # Minimal API dependencies
```

---

## Phase Progress Overview

| Phase | Status | What was built |
|---|---|---|
| 1 — Setup | ✅ Complete | Virtual environment, dependencies, Git, VS Code |
| 2 — EDA | ✅ Complete | Data ingestion, 5 visualisations, 5 modelling decisions |
| 3 — Modelling | ✅ Complete | Feature pipeline, two-tower neural, GB baseline |
| 4 — A/B Test | ✅ Complete | Ranking metrics, statistical significance, winner declared |
| 5 — Serving | ✅ Complete | FastAPI endpoints, Pydantic validation, Docker |
| 6 — Monitoring | ✅ Complete | Evidently AI drift detection, HTML report |

---

## Phase 1 — Project Setup

**What was built:**
- Python virtual environment (`venv`) for isolated dependency management
- Modular folder structure: `data/`, `src/features/`, `src/models/`, `src/api/`, `monitoring/`, `notebooks/`
- Git repository initialised with `.gitignore` excluding `venv/`, `__pycache__/`, raw data
- `requirements.txt` frozen with `pip freeze` for full reproducibility
- VS Code configured with Python, Pylance, Jupyter, GitLens extensions

**Key decision:** Project structured before writing any ML code — separating data, features, models, and API concerns from day one, mirroring how production ML teams organise repositories.

---

## Phase 2 — Data Ingestion & EDA

**What was built:**
- `src/data_loader.py` — idempotent download pipeline for MovieLens Small dataset with automatic extraction and cleanup
- `notebooks/01_eda.ipynb` — full exploratory analysis across ratings, users, movies, genres, and time
- 5 production-quality visualisations saved to `monitoring/`

**Dataset statistics:**

```
Total ratings:        100,836
Unique users:         610  (all have ≥ 20 ratings)
Unique movies:        9,724
Matrix sparsity:      98.30%
Average rating:       3.50 / 5.0
Median ratings/user:  70
Date range:           March 1996 – September 2018
Most common rating:   4.0  (selection bias — users rate what they chose to watch)
Total genres:         20
Positive labels:      48,580  (48.2% — well balanced, no class weighting needed)
```

**Five EDA findings that drove five modelling decisions:**

| Finding | Evidence | Decision |
|---|---|---|
| 98.3% sparsity | 5.9M possible, 100k observed | Two-tower embeddings over nearest-neighbour |
| Popularity bias | Top 15 rated = all pre-2000 classics | Neural model over pure collaborative filtering |
| Power law users | 20 to 2,698 ratings per user | log1p scaling on count features |
| Temporal drift | Ratings span 1996–2018 (22 years) | Recency feature (0–1 normalised timestamp) |
| Multi-label genres | 20 genres, pipe-separated | Multi-hot encoding + dedicated genre Dense(32) layer |

**Visualisations generated:** rating distribution, ratings per user (power law), genre distribution, ratings by year (temporal drift), top-15 most rated movies (popularity bias).

---

## Phase 3 — Feature Engineering & Model Training

### Feature Engineering

Every interaction becomes a row of 28 numbers fed to the model:

| Feature group | Features | What they capture |
|---|---|---|
| User identity | `user_idx` | Embedding table lookup key |
| User behaviour | `user_mean_rating` | Generous vs harsh rater |
| User behaviour | `user_rating_count` | Activity level — log1p scaled |
| User behaviour | `user_rating_std` | Opinionated vs flat rater |
| Movie identity | `movie_idx` | Embedding table lookup key |
| Movie content | `genre_vector[20]` | Multi-hot genre encoding |
| Movie quality | `movie_mean_rating` | Global quality signal |
| Movie quality | `movie_rating_count` | Popularity — log1p scaled |
| Temporal | `recency` | How recent the interaction is (0=1996, 1=2018) |
| Target | `label` | 1 if rating ≥ 4.0, else 0 |

**Why log1p on count features?** Raw counts range from 20 to 2,698 (135× difference). Without transformation, high-activity users dominate the model by data volume rather than preference quality. `log1p` compresses this to a 2.6× range.

**Why binary label not rating regression?** Predicting 3.5 vs 4.0 is noise — the difference reflects mood, not true preference. Binary classification directly optimises for the business objective: will the user engage positively? This mirrors how YouTube and Spotify frame the problem.

**Output artifacts:** `interactions.parquet` (100,836 × 15), `movie_features.parquet`, `user_features.parquet`, `genre_matrix.npy` (9,742 × 20), `genre_vocab.json`.

### Two-Tower Neural Architecture

```
User inputs                              Movie inputs
(user_idx, mean, count, std)             (movie_idx, genre[20], mean, count)
         │                                       │
   Embedding(610, 64)                  Embedding(9724, 64)
         │                             Genre Dense(20 → 32, ReLU)
         │                                       │
   Concatenate → 67 dims               Concatenate → 98 dims
         │                                       │
   Dense(128, ReLU)                    Dense(128, ReLU)
   BatchNorm1d → Dropout(0.3)          BatchNorm1d → Dropout(0.3)
   Dense(64, ReLU) → BatchNorm         Dense(64, ReLU) → BatchNorm
   Dense(32, linear)                   Dense(32, linear)
         │                                       │
         └────── F.normalize ──── dot ───────────┘
                                  │
                      Concatenate(dot_score, recency)
                                  │
                            Dense(16, ReLU)
                            Dense(1, sigmoid)
                                  │
                      P(user likes movie) → 0.0 to 1.0
```

**Total parameters: 704,929** — trained on Apple M2 GPU via MPS in ~90 seconds.

**Key architectural decisions:**

- **Separate towers**: user and movie towers are independent — movie vectors can be pre-computed once and cached. At inference, only the user tower runs per request.
- **Cosine similarity via F.normalize**: normalises vectors to unit length before dot product. Magnitude-invariant — compares direction of taste, not data volume.
- **Recency injected after dot product**: prevents temporal signal from distorting the embedding space.
- **No activation on final Dense(32)**: allows unbounded positive and negative values in embedding space. ReLU would destroy half the information capacity.
- **Genre sub-network**: 20-dimensional genre vector passes through Dense(32) before joining the main flow.

### Training curve

| Epoch | Train Loss | Val AUC | Notes |
|---|---|---|---|
| 1 | 0.6924 | 0.6665 | Random embeddings — learning starts |
| 2 | 0.6336 | 0.7891 | +18.4% AUC jump — embedding space organising |
| 3 | 0.5561 | 0.8119 | Genre combinations captured |
| 4 | 0.5291 | 0.8190 | Fine-tuning phase |
| 5 | 0.5181 | 0.8201 | Converging |
| **6** | **0.5108** | **0.8211** | **Best model — saved to disk** |
| 7 | 0.5043 | 0.8202 | Plateau |
| 8 | 0.4994 | 0.8198 | No improvement |
| 9 | 0.4940 | 0.8188 | Early stopping triggered |

### Gradient Boosting Baseline

200 trees, learning_rate=0.1, max_depth=5, subsample=0.8, trained on StandardScaler-normalised features in ~20 seconds on CPU.

### Phase 3 — Classification metrics (80/20 stratified split)

| Metric | Gradient Boosting | Two-Tower Neural | Winner |
|---|---|---|---|
| AUC | **0.8403** | 0.8211 | GB |
| Precision | **0.7386** | 0.7203 | GB |
| Recall | **0.7627** | 0.7574 | GB |
| Train time | ~20 seconds | ~90 seconds | GB |
| Cold start | ❌ Crashes on new users | ✅ Embedding defaults | Neural |
| New movies | ❌ Requires retraining | ✅ Movie tower runs instantly | Neural |
| Scale to 1B items | ❌ Linear scoring | ✅ ANN vector search <10ms | Neural |
| Explainability | ✅ Feature importance | ❌ Black box embeddings | GB |

**Key engineering decisions:**
- PyTorch chosen over TensorFlow — MPS GPU gave 10× speedup on Apple Silicon vs CPU-only TensorFlow
- Genre vectors saved as `.npy` — numpy arrays do not serialise cleanly in pandas Parquet columns
- Both models tracked in MLflow with full hyperparameter and metric logging

---

## Phase 4 — A/B Testing

**Evaluation strategy:** Leave-one-out with 20% holdout — for each of 465 users, 20% of their liked movies were held out as ground truth. Both models scored all unseen movies and we measured whether the holdout movies appeared in the top-10 recommendations.

### Ranking metrics (K=10)

| Metric | Gradient Boosting | Two-Tower Neural | Improvement | Winner |
|---|---|---|---|---|
| Precision@10 | 0.0004 | 0.0092 | **+2,050%** | Neural ✅ |
| Recall@10 | 0.0000 | 0.0069 | **+17,626%** | Neural ✅ |
| NDCG@10 | 0.0003 | 0.0112 | **+3,195%** | Neural ✅ |
| Users with hit in top-10 | 1 / 465 | 38 / 465 | **38×** | Neural ✅ |

**Statistical significance:** Paired t-test → t = −5.19, **p = 0.0000** ✅

### The metric inversion — most important insight

GB won on AUC (0.84 vs 0.82) but lost on every ranking metric by thousands of percent. **AUC measures global pairwise ranking quality — NDCG@10 measures whether the best movies appear in the top-10 slots the user actually sees.**

Neural embeddings learn the direction of taste in vector space — users and movies they like point in the same direction. This naturally concentrates the most relevant movies at the very top of the ranked list. GB has no shared embedding space — it cannot do this.

> *"Our A/B test revealed a metric inversion — GB had higher AUC but neural won every ranking metric with p=0.0000. This is why AUC alone is insufficient for recommendation evaluation. We deploy based on NDCG@10, not AUC."*

**What was built:** `src/models/ab_test.py` — leave-one-out evaluation with Precision@10, Recall@10, NDCG@10, paired t-test, MLflow logging, and two visualisation plots (`ab_test_comparison.png`, `ab_test_ndcg_dist.png`).

---

## Phase 5 — FastAPI Serving & Docker

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API info and available routes |
| `/health` | GET | Both models loaded, AUC scores, version |
| `/recommend` | POST | Personalised top-K movie recommendations |
| `/docs` | GET | Auto-generated interactive Swagger UI |

### Request schema

```json
{
  "user_id": 1,        // int · required · range 1–610
  "model":   "neural", // "neural" or "gb" · default "neural"
  "top_k":   10        // int · range 1–50 · default 10
}
```

### API design decisions

- **Lifespan context manager** — both models loaded once at startup and kept in RAM. Not reloaded per request. Eliminates 5-second loading time per call.
- **Pydantic field validation** — user_id range, model enum, top_k bounds all validated before model code runs. Invalid requests return 422 with clear error message.
- **Separate recommender class** — inference logic decoupled from routing logic. Easier to test, swap models, and add new models without touching API routes.

### Docker

- **Dockerfile** — `python:3.11-slim` base, gcc/curl installed, dependencies cached in separate layer from code, PyTorch CPU wheel (MPS unavailable in Linux containers)
- **docker-compose.yml** — port mapping (8000:8000), volume mount for live data updates, restart policy, health check every 30s

**What was built:** `src/api/schemas.py` (Pydantic validation), `src/api/recommender.py` (model loading + inference), `src/api/main.py` (FastAPI routes + lifespan), `Dockerfile`, `docker-compose.yml`, `requirements_api.txt`.

**Verified working:** Live API tested with curl — returns real personalised recommendations, rejects invalid inputs with 422 errors, runs identically inside Docker container.

---

## Phase 6 — Drift Monitoring

Evidently AI compares 5,000 samples from training data against 5,000 simulated production samples:

| Feature | Training mean | Production mean | Status |
|---|---|---|---|
| user_mean_rating | 3.5016 | 3.8014 | ⚠️ DRIFT — users rating more generously |
| user_rating_count | 5.8194 | 6.3196 | ⚠️ DRIFT — users more active |
| user_rating_std | 0.9140 | 0.9140 | ✅ Stable |
| movie_mean_rating | 3.5016 | 3.3017 | ⚠️ DRIFT — movies rated lower overall |
| movie_rating_count | 3.4855 | 3.4855 | ✅ Stable |
| recency | 0.7842 | 0.8982 | ⚠️ DRIFT — interactions skewing more recent |

**What 4 drifted features means in production:** The model was calibrated on a world where users averaged 3.50 ratings. If users now average 3.80, the model's threshold for "liked" (≥ 4.0) is now underestimating positive sentiment. Combined with recency drift, this signals the model should be retrained on data from the past 6 months.

**What was built:** `monitoring/drift_monitor.py` — builds a feature dataframe, simulates production drift, runs Evidently's `DataDriftPreset`, saves an HTML report, and prints a plain-text summary.

**To view the full HTML report:**
```bash
open monitoring/drift_report.html
```

---

## Key Numbers — Quick Reference

```
Dataset         100,836 ratings · 610 users · 9,724 movies · 98.3% sparse
Neural model    704,929 parameters · AUC 0.8211 · trained in 90s on M2 GPU
GB baseline     200 trees · AUC 0.8403 · trained in 20s on CPU
A/B test        465 users · NDCG@10 +3,195% · p=0.0000
API             FastAPI + Docker · /health + /recommend + /docs
Drift           4/6 features drifted · user_mean_rating +8.5% · recency +14.5%
```

---

## What This Project Demonstrates

| Skill | Evidence |
|---|---|
| ML architecture design | Two-tower with cosine similarity, recency injection, genre sub-network |
| Production evaluation | A/B test with NDCG@10, not just AUC — metric inversion discovered |
| MLOps | MLflow tracking, Docker serving, drift monitoring — full lifecycle |
| Engineering rigour | Pydantic validation, lifespan model loading, log1p scaling, stratified splits |
| Business understanding | AUC vs NDCG distinction, cold start handling, scalability to ANN retrieval |
