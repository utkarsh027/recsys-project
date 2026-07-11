import os
import sys
import numpy as np
import pandas as pd
import pickle
import torch

sys.path.append(os.path.abspath("."))
from src.models.two_tower import TwoTowerModel

PROCESSED_DIR = "data/processed"
MODEL_DIR     = "data/models"
DEVICE        = torch.device("cpu")


class Recommender:
    def __init__(self):
        print("Loading data...")
        self.df           = pd.read_parquet(os.path.join(PROCESSED_DIR, "interactions.parquet"))
        self.movie_feat   = pd.read_parquet(os.path.join(PROCESSED_DIR, "movie_features.parquet"))
        self.genre_matrix = np.load(os.path.join(PROCESSED_DIR, "genre_matrix.npy"))
        self.movies_raw   = pd.read_csv("data/raw/ml-latest-small/movies.csv")

        self.mid_to_title = dict(zip(self.movies_raw["movieId"], self.movies_raw["title"]))
        self.mid_to_genre = dict(zip(self.movies_raw["movieId"], self.movies_raw["genres"]))

        mid_to_genre_row  = {mid: i for i, mid in enumerate(self.movie_feat["movieId"].values)}
        self.mid_to_genre_row = mid_to_genre_row

        mid_to_idx = {}
        for _, row in self.df.drop_duplicates("movieId").iterrows():
            mid_to_idx[int(row["movieId"])] = int(row["movie_idx"])
        self.mid_to_idx = mid_to_idx

        print("Loading GB model...")
        with open(os.path.join(MODEL_DIR, "gb_model.pkl"), "rb") as f:
            self.gb_model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)

        print("Loading neural model...")
        num_users  = int(self.df["user_idx"].max()) + 1
        num_movies = int(self.df["movie_idx"].max()) + 1
        self.nn_model = TwoTowerModel(num_users, num_movies, 20, 64).to(DEVICE)
        self.nn_model.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, "two_tower_final.pt"), map_location=DEVICE)
        )
        self.nn_model.eval()
        print(f"All models loaded on {DEVICE}")

    def get_user_stats(self, user_id: int):
        row = self.df[self.df["userId"] == user_id]
        if len(row) == 0:
            return None
        r = row.iloc[0]
        return {
            "user_idx":          int(r["user_idx"]),
            "user_mean_rating":  float(r["user_mean_rating"]),
            "user_rating_count": float(r["user_rating_count"]),
            "user_rating_std":   float(r["user_rating_std"]),
            "recency":           float(r["recency"]),
        }

    def get_unseen_movies(self, user_id: int):
        rated   = set(self.df[self.df["userId"] == user_id]["movieId"].values)
        unseen  = self.movie_feat[~self.movie_feat["movieId"].isin(rated)].copy()
        unseen  = unseen[unseen["movieId"].isin(self.mid_to_idx)].reset_index(drop=True)
        unseen["movie_idx"] = unseen["movieId"].map(self.mid_to_idx)
        genres  = np.array([
            self.genre_matrix[self.mid_to_genre_row[m]]
            for m in unseen["movieId"].values
        ])
        return unseen, genres

    def score_neural(self, u, movies, genres):
        n = len(movies)
        with torch.no_grad():
            scores = self.nn_model(
                torch.full((n,),  u["user_idx"],          dtype=torch.long).to(DEVICE),
                torch.full((n,1), u["user_mean_rating"],  dtype=torch.float32).to(DEVICE),
                torch.full((n,1), u["user_rating_count"], dtype=torch.float32).to(DEVICE),
                torch.full((n,1), u["user_rating_std"],   dtype=torch.float32).to(DEVICE),
                torch.tensor(movies["movie_idx"].values,  dtype=torch.long).to(DEVICE),
                torch.tensor(genres,                      dtype=torch.float32).to(DEVICE),
                torch.tensor(movies["movie_mean_rating"].values,  dtype=torch.float32).unsqueeze(1).to(DEVICE),
                torch.tensor(movies["movie_rating_count"].values, dtype=torch.float32).unsqueeze(1).to(DEVICE),
                torch.full((n,1), u["recency"], dtype=torch.float32).to(DEVICE),
            ).cpu().numpy().flatten()
        return scores

    def score_gb(self, u, movies, genres):
        n = len(movies)
        X = np.column_stack([
            np.full(n, u["user_idx"]),
            movies["movie_idx"].values,
            np.full(n, u["user_mean_rating"]),
            np.full(n, u["user_rating_count"]),
            np.full(n, u["user_rating_std"]),
            movies["movie_mean_rating"].values,
            movies["movie_rating_count"].values,
            np.full(n, u["recency"]),
            genres,
        ]).astype(np.float32)
        return self.gb_model.predict_proba(self.scaler.transform(X))[:, 1]

    def recommend(self, user_id: int, model: str = "neural", top_k: int = 10):
        u = self.get_user_stats(user_id)
        if u is None:
            return None

        unseen, genres = self.get_unseen_movies(user_id)

        scores  = self.score_neural(u, unseen, genres) if model == "neural" else self.score_gb(u, unseen, genres)
        top_idx = np.argsort(-scores)[:top_k]
        top_movies = unseen.iloc[top_idx].reset_index(drop=True)
        top_scores = scores[top_idx]

        return [
            {
                "rank":     rank,
                "movie_id": int(row["movieId"]),
                "title":    self.mid_to_title.get(int(row["movieId"]), "Unknown"),
                "score":    round(float(top_scores[rank-1]), 4),
                "genres":   self.mid_to_genre.get(int(row["movieId"]), ""),
            }
            for rank, (_, row) in enumerate(top_movies.iterrows(), 1)
        ]
