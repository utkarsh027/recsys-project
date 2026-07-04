# Production-Scale Recommendation System

A production-ready recommendation engine that trains and compares two models — a **Gradient Boosting baseline** and a **Two-Tower Neural Network** — with a full MLOps pipeline, A/B testing, and real-time serving. Designed to mirror how recommendation systems work at companies like Amazon, Flipkart, and YouTube.

## Architecture

| Layer | Technology |
|---|---|
| Neural model | Two-tower network (PyTorch + Apple MPS GPU) |
| Baseline model | Gradient Boosting (scikit-learn) |
| Feature engineering | pandas · numpy · multi-hot genre encoding |
| Experiment tracking | MLflow |
| API serving | FastAPI + Uvicorn |
| Containerisation | Docker |
| Drift monitoring | Evidently AI |
| Dataset | MovieLens Small (100k ratings, 610 users, 9,724 movies) |

## Project Structure

```
recsys-project/
├── data/
│   ├── raw/                        # Downloaded MovieLens dataset
│   └── processed/                  # Parquet features + genre matrix
│       ├── interactions.parquet
│       ├── movie_features.parquet
│       ├── user_features.parquet
│       ├── genre_matrix.npy        # 9742 × 20 multi-hot genre vectors
│       └── genre_vocab.json
├── data/models/
│   ├── gb_model.pkl                # Trained gradient boosting model
│   ├── scaler.pkl                  # Feature scaler
│   └── two_tower_final.pt          # Trained PyTorch two-tower model
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   └── 02_model.ipynb              # Model experiments
├── src/
│   ├── data_loader.py              # Idempotent dataset download
│   ├── features/
│   │   └── feature_engineering.py # Full feature pipeline
│   ├── models/
│   │   ├── two_tower.py            # PyTorch two-tower architecture
│   │   ├── train.py                # Gradient boosting training
│   │   └── train_neural.py         # Neural model training on MPS GPU
│   └── api/                        # FastAPI serving layer
└── monitoring/                     # EDA plots + Evidently AI drift reports
```

## Progress

| Phase | Status | Description |
|---|---|---|
| Phase 1 | ✅ Complete | Project setup, virtual environment, dependencies, Git |
| Phase 2 | ✅ Complete | Data ingestion, EDA, 5 visualisations, modelling decisions |
| Phase 3 | ✅ Complete | Feature engineering, two-tower neural model, GB baseline, model comparison |
| Phase 4 | 🔄 Up next | A/B test: Precision@K, NDCG@10, statistical significance |
| Phase 5 | ⏳ Pending | FastAPI serving and Docker containerisation |
| Phase 6 | ⏳ Pending | Drift monitoring with Evidently AI |

## Model Comparison — Phase 3 Results

Two models trained and evaluated on the same 80/20 stratified train/test split.

| Metric | Gradient Boosting | Two-Tower Neural | Winner |
|---|---|---|---|
| AUC | **0.8403** | 0.8211 | GB (small dataset advantage) |
| Precision | **0.7386** | 0.7203 | GB |
| Recall | **0.7627** | 0.7574 | GB |
| Cold start | ❌ Fails | ✅ Handles via embedding defaults | Neural |
| New movies | ❌ Needs retraining | ✅ Movie tower runs instantly | Neural |
| Scalability | ❌ Scores all movies linearly | ✅ ANN vector search in <10ms | Neural |
| Explainability | ✅ Feature importance | ❌ Black box embeddings | GB |

**Why GB wins on AUC but neural wins in production:** Tree models excel on tabular data at 100k scale. With 10M+ interactions (real production scale), neural embeddings get progressively richer and significantly outperform tree models — which is exactly why YouTube, Amazon, and Spotify all use two-tower architectures, not gradient boosting.

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

## Neural Model Training Curve

| Epoch | Train Loss | Val AUC | Notes |
|---|---|---|---|
| 1 | 0.6924 | 0.6665 | Embeddings initialising |
| 2 | 0.6336 | 0.7891 | Major jump — patterns emerging |
| 3 | 0.5561 | 0.8119 | Genre combinations learned |
| 4 | 0.5291 | 0.8190 | Fine-tuning |
| 5 | 0.5181 | 0.8201 | Converging |
| **6** | **0.5108** | **0.8211** | **Best model — saved** |
| 7 | 0.5043 | 0.8202 | Plateau |
| 8 | 0.4994 | 0.8198 | No improvement |
| 9 | 0.4940 | 0.8188 | Early stopping triggered |

## EDA Key Findings

Analysis of 100,836 ratings from 610 users across 9,724 movies revealed five critical insights that directly shaped modelling decisions:

| Finding | Data Evidence | Modelling Decision |
|---|---|---|
| Extreme sparsity | 98.3% of interaction matrix is empty | Embedding-based two-tower model |
| Popularity bias | Top 15 rated movies are all pre-2000 classics | Two-tower over pure collaborative filtering |
| Power law activity | Users range from 20 to 2,698 ratings | Stratified train/test split + log1p scaling |
| Temporal drift | Ratings span 1996–2018 (22 years) | Timestamp encoded as recency feature |
| Multi-label genres | 20 genres, pipe-separated per movie | Multi-hot encoding + dedicated genre Dense layer |

## Phase 3 Progress — 5 July 2026

**What was built:**

- `src/features/feature_engineering.py` — full pipeline: user behavioural stats, multi-hot genre vectors (9742×20), log1p scaling, recency normalisation, binary label creation, parquet + npy output
- `src/models/two_tower.py` — PyTorch two-tower architecture with user tower, movie tower, cosine similarity, recency injection, sigmoid output
- `src/models/train_neural.py` — PyTorch training loop with MPS GPU acceleration, early stopping, LR scheduling, MLflow tracking
- `src/models/train.py` — Gradient Boosting baseline with StandardScaler and MLflow tracking
- `data/models/two_tower_final.pt` — best neural model weights (epoch 6, AUC 0.8211)
- `data/models/gb_model.pkl` — trained GB baseline (AUC 0.8403)

**Key engineering decisions:**

- Genre vectors saved as `.npy` because numpy arrays do not serialise cleanly in pandas DataFrames
- PyTorch chosen over TensorFlow — MPS GPU gave 10x speedup on Apple Silicon
- Binary label (rating ≥ 4.0) rather than regression — mirrors production framing at YouTube and Spotify
- Cosine similarity via normalised dot product — magnitude-invariant, fair across users with different activity levels

## Phase 2 Progress — 2 July 2026

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
Positive labels:      48,580 (48.2%)
```
