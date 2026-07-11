# Production-Scale Recommendation System

A production-ready recommendation engine that trains and compares two models — a **Gradient Boosting baseline** and a **Two-Tower Neural Network** — with a full MLOps pipeline, statistically validated A/B testing, and real-time serving via FastAPI and Docker. Designed to mirror how recommendation systems work at companies like Amazon, Flipkart, and YouTube.

## Architecture

| Layer | Technology |
|---|---|
| Neural model | Two-tower network (PyTorch + Apple MPS GPU) |
| Baseline model | Gradient Boosting (scikit-learn) |
| Feature engineering | pandas · numpy · multi-hot genre encoding |
| Experiment tracking | MLflow |
| API serving | FastAPI + Uvicorn |
| Containerisation | Docker + Docker Compose |
| Drift monitoring | Evidently AI |
| Dataset | MovieLens Small (100k ratings, 610 users, 9,724 movies) |

## Quick Start — Run the API

```bash
git clone https://github.com/utkarsh027/recsys-project.git
cd recsys-project
docker-compose up --build
```

Then call the API:

```bash
# Health check
curl http://localhost:8000/health

# Get top-5 recommendations for User 1 using neural model
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "model": "neural", "top_k": 5}'
```

Example response:
```json
{
  "user_id": 1,
  "model": "neural",
  "top_k": 5,
  "recommendations": [
    {"rank": 1, "movie_id": 3160, "title": "Magnolia (1999)",    "score": 0.9108, "genres": "Drama"},
    {"rank": 2, "movie_id": 64614,"title": "Gran Torino (2008)", "score": 0.9103, "genres": "Crime|Drama"},
    {"rank": 3, "movie_id": 1958, "title": "Terms of Endearment (1983)", "score": 0.9101, "genres": "Comedy|Drama"},
    {"rank": 4, "movie_id": 2300, "title": "Producers, The (1968)", "score": 0.9099, "genres": "Comedy"},
    {"rank": 5, "movie_id": 2067, "title": "Doctor Zhivago (1965)", "score": 0.9099, "genres": "Drama|Romance|War"}
  ]
}
```

Interactive API docs available at **http://localhost:8000/docs**

---

## Project Structure

```
recsys-project/
├── data/
│   ├── raw/                          # Downloaded MovieLens dataset
│   └── processed/                    # Parquet features + genre matrix
│       ├── interactions.parquet
│       ├── movie_features.parquet
│       ├── user_features.parquet
│       ├── genre_matrix.npy          # 9742 × 20 multi-hot genre vectors
│       └── genre_vocab.json
├── data/models/
│   ├── gb_model.pkl                  # Trained gradient boosting model
│   ├── scaler.pkl                    # Feature scaler
│   └── two_tower_final.pt            # Trained PyTorch two-tower model
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   └── 02_model.ipynb                # Model experiments
├── src/
│   ├── data_loader.py                # Idempotent dataset download
│   ├── features/
│   │   └── feature_engineering.py   # Full feature pipeline
│   ├── models/
│   │   ├── two_tower.py              # PyTorch two-tower architecture
│   │   ├── train.py                  # Gradient boosting training
│   │   ├── train_neural.py           # Neural model training on MPS GPU
│   │   └── ab_test.py                # A/B test with ranking metrics
│   └── api/
│       ├── main.py                   # FastAPI application
│       ├── schemas.py                # Pydantic request/response models
│       └── recommender.py            # Model loading and inference
├── monitoring/                       # EDA plots · A/B results · drift reports
│   ├── ab_test_results.csv
│   ├── ab_test_comparison.png
│   └── ab_test_ndcg_dist.png
├── Dockerfile                        # Container definition
├── docker-compose.yml                # One-command deployment
└── requirements_api.txt              # API dependencies
```

---

## Progress

| Phase | Status | Description |
|---|---|---|
| Phase 1 | ✅ Complete | Project setup, virtual environment, dependencies, Git |
| Phase 2 | ✅ Complete | Data ingestion, EDA, 5 visualisations, modelling decisions |
| Phase 3 | ✅ Complete | Feature engineering, two-tower neural model, GB baseline |
| Phase 4 | ✅ Complete | A/B test — Precision@K, NDCG@10, statistical significance |
| Phase 5 | ✅ Complete | FastAPI serving + Docker containerisation + live demo |
| Phase 6 | 🔄 Up next | Drift monitoring with Evidently AI |

---

## Phase 5 — API Endpoints — 12 July 2026

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API info and available endpoints |
| `/health` | GET | Model status, AUC scores, version |
| `/recommend` | POST | Top-K personalised movie recommendations |
| `/docs` | GET | Auto-generated interactive API documentation |

### Request schema

```json
{
  "user_id": 1,        // int, required, 1–610
  "model":   "neural", // "neural" or "gb", default "neural"
  "top_k":   10        // int, 1–50, default 10
}
```

### What happens inside the API

1. Pydantic validates the request — invalid `user_id` or `model` values are rejected with a clear 422 error before the model is ever called
2. The recommender loads both models **once at startup** and keeps them in memory — no reloading per request
3. For neural model: scores all unseen movies via the two-tower forward pass, sorts by cosine similarity score, returns top-K with titles and genres
4. For GB model: constructs tabular feature matrix, scales it, runs predict_proba, returns top-K

### Docker

The entire application — Python, PyTorch, models, data — is packaged into a single Docker container. Zero setup required:

```bash
docker-compose up --build   # first run ~5 minutes (downloading dependencies)
docker-compose up           # subsequent runs ~30 seconds
```

---

## Phase 4 — A/B Test Results

**Evaluation strategy:** Leave-one-out with 20% holdout — for each of 465 users, 20% of their liked movies were held out as ground truth. Both models scored all unseen movies and we measured whether the holdout movies appeared in top-10 recommendations.

| Metric | Gradient Boosting | Two-Tower Neural | Improvement | Winner |
|---|---|---|---|---|
| Precision@10 | 0.0004 | 0.0092 | +2,050% | Neural ✅ |
| Recall@10 | 0.0000 | 0.0069 | +17,626% | Neural ✅ |
| NDCG@10 | 0.0003 | 0.0112 | +3,195% | Neural ✅ |
| Users with hit in top-10 | 1 / 465 | 38 / 465 | 38× more | Neural ✅ |

**Statistical significance:** Paired t-test on NDCG@10 → t = −5.19, **p = 0.0000** ✅

### The metric inversion

GB had higher AUC (0.84 vs 0.82) in Phase 3 — yet the neural model won every ranking metric by thousands of percent. AUC measures global pairwise ranking. NDCG@10 measures whether the best movies appear in the top-10 slots the user actually sees. Neural embeddings naturally concentrate the most relevant movies at the very top. GB cannot do this — it has no shared embedding space between users and movies.

**Interview line:** *"Our A/B test revealed a metric inversion — GB had higher AUC but the neural model won every ranking metric with p=0.0000. This is why AUC alone is insufficient for recommendation evaluation."*

---

## Phase 3 — Model Comparison

| Metric | Gradient Boosting | Two-Tower Neural | Winner |
|---|---|---|---|
| AUC | **0.8403** | 0.8211 | GB (small dataset) |
| Precision | **0.7386** | 0.7203 | GB |
| Recall | **0.7627** | 0.7574 | GB |
| Cold start | ❌ Fails | ✅ Handles via embedding defaults | Neural |
| New movies | ❌ Needs retraining | ✅ Movie tower runs instantly | Neural |
| Scalability | ❌ Scores all movies linearly | ✅ ANN vector search in <10ms | Neural |
| NDCG@10 (A/B test) | 0.0003 | **0.0112** | **Neural ✅** |

---

## Two-Tower Neural Architecture

```
User inputs                          Movie inputs
(user_idx, mean, count, std)         (movie_idx, genre[20], mean, count)
        │                                    │
  Embedding(610, 64)               Embedding(9724, 64)
        │                          Genre Dense(20→32)
        │                                    │
  Concatenate (67-dim)              Concatenate (98-dim)
        │                                    │
  Dense(128) → BatchNorm → Dropout   Dense(128) → BatchNorm → Dropout
        │                                    │
  Dense(64) → BatchNorm              Dense(64) → BatchNorm
        │                                    │
  Dense(32) ← user vector            movie vector → Dense(32)
        │                                    │
        └──── Cosine similarity (dot product) ────┘
                         │
               Concatenate recency
                         │
               Dense(16) → Dense(1, sigmoid)
                         │
               P(user likes movie) → 0.0 to 1.0
```

Total parameters: **704,929** — trained on Apple M2 GPU via MPS in ~90 seconds.

### Training curve

| Epoch | Train Loss | Val AUC | Notes |
|---|---|---|---|
| 1 | 0.6924 | 0.6665 | Embeddings initialising |
| 2 | 0.6336 | 0.7891 | Major jump — patterns emerging |
| 3 | 0.5561 | 0.8119 | Genre combinations learned |
| 4 | 0.5291 | 0.8190 | Fine-tuning |
| 5 | 0.5181 | 0.8201 | Converging |
| **6** | **0.5108** | **0.8211** | **Best model — saved** |
| 7–9 | — | plateau | Early stopping triggered at epoch 9 |

---

## EDA Key Findings

| Finding | Data Evidence | Modelling Decision |
|---|---|---|
| Extreme sparsity | 98.3% of interaction matrix is empty | Embedding-based two-tower model |
| Popularity bias | Top 15 rated movies are all pre-2000 classics | Two-tower over pure collaborative filtering |
| Power law activity | Users range from 20 to 2,698 ratings | Stratified train/test split + log1p scaling |
| Temporal drift | Ratings span 1996–2018 (22 years) | Timestamp encoded as recency feature |
| Multi-label genres | 20 genres, pipe-separated per movie | Multi-hot encoding + dedicated genre Dense layer |

---

## Phase 5 Progress — 12 July 2026

**What was built:**
- `src/api/schemas.py` — Pydantic request/response schemas with field validation (user_id 1–610, model neural/gb, top_k 1–50)
- `src/api/recommender.py` — model loading at startup, user feature lookup, unseen movie filtering, neural + GB scoring
- `src/api/main.py` — FastAPI app with lifespan context manager, three endpoints, automatic /docs generation
- `Dockerfile` — Python 3.11 slim, PyTorch CPU, all dependencies, model files copied into container
- `docker-compose.yml` — port mapping, volume mount, health check, restart policy
- `requirements_api.txt` — minimal dependency set for API serving only

**Key engineering decisions:**
- Models loaded once at startup via FastAPI lifespan — not reloaded per request
- PyTorch CPU in Docker — MPS is Mac-specific, CPU works everywhere for serving
- Pydantic field validators reject bad inputs before model code runs
- Health check endpoint monitors liveness every 30 seconds

## Phase 4 Progress — 7 July 2026

**What was built:**
- `src/models/ab_test.py` — leave-one-out evaluation, Precision@10, Recall@10, NDCG@10, paired t-test, MLflow logging, two plots
- `monitoring/ab_test_results.csv` — per-user scores for all 465 users
- `monitoring/ab_test_comparison.png` — bar chart comparing all three metrics
- `monitoring/ab_test_ndcg_dist.png` — NDCG@10 distribution + difference histogram

## Phase 3 Progress — 5 July 2026

**What was built:**
- `src/features/feature_engineering.py` — full pipeline: user behavioural stats, multi-hot genre vectors (9742×20), log1p scaling, recency, parquet + npy output
- `src/models/two_tower.py` — PyTorch two-tower architecture (704,929 parameters)
- `src/models/train_neural.py` — MPS GPU training with early stopping and MLflow tracking
- `src/models/train.py` — Gradient Boosting baseline

## Phase 2 Progress — 2 July 2026

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
Positive labels:      48,580 (48.2%)
```
