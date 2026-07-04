import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import sys

sys.path.append(os.path.abspath("."))
from src.data_loader import get_file_paths, EXTRACTED_DIR


def load_raw_data():
    paths = get_file_paths(EXTRACTED_DIR)
    ratings = pd.read_csv(paths["ratings"])
    movies  = pd.read_csv(paths["movies"])
    return ratings, movies


def build_genre_vocab(movies):
    all_genres = set()
    for g in movies["genres"].dropna():
        for genre in g.split("|"):
            all_genres.add(genre)
    genre_vocab = {g: i for i, g in enumerate(sorted(all_genres))}
    return genre_vocab


def encode_genres(movies, genre_vocab):
    def multi_hot(genre_str):
        vec = np.zeros(len(genre_vocab), dtype=np.float32)
        for g in genre_str.split("|"):
            if g in genre_vocab:
                vec[genre_vocab[g]] = 1.0
        return vec

    movies = movies.copy()
    movies["genre_vector"] = movies["genres"].apply(multi_hot)
    return movies


def build_user_features(ratings):
    user_stats = ratings.groupby("userId").agg(
        user_mean_rating  = ("rating", "mean"),
        user_rating_count = ("rating", "count"),
        user_rating_std   = ("rating", "std")
    ).reset_index()

    user_stats["user_rating_std"]   = user_stats["user_rating_std"].fillna(0.0)
    user_stats["user_mean_rating"]  = user_stats["user_mean_rating"].astype(np.float32)
    user_stats["user_rating_count"] = np.log1p(user_stats["user_rating_count"]).astype(np.float32)
    user_stats["user_rating_std"]   = user_stats["user_rating_std"].astype(np.float32)
    return user_stats


def build_movie_features(ratings, movies, genre_vocab):
    movie_stats = ratings.groupby("movieId").agg(
        movie_mean_rating  = ("rating", "mean"),
        movie_rating_count = ("rating", "count")
    ).reset_index()

    movie_stats["movie_mean_rating"]  = movie_stats["movie_mean_rating"].astype(np.float32)
    movie_stats["movie_rating_count"] = np.log1p(movie_stats["movie_rating_count"]).astype(np.float32)

    movies_encoded = encode_genres(movies, genre_vocab)
    movie_features = movies_encoded.merge(movie_stats, on="movieId", how="left")
    movie_features["movie_mean_rating"]  = movie_features["movie_mean_rating"].fillna(3.5).astype(np.float32)
    movie_features["movie_rating_count"] = movie_features["movie_rating_count"].fillna(0.0).astype(np.float32)
    return movie_features


def encode_ids(ratings, movies):
    user_encoder  = LabelEncoder()
    movie_encoder = LabelEncoder()

    ratings = ratings.copy()
    movies  = movies.copy()

    ratings["user_idx"]  = user_encoder.fit_transform(ratings["userId"])
    ratings["movie_idx"] = movie_encoder.fit_transform(ratings["movieId"])
    movies["movie_idx"]  = movie_encoder.transform(
        movies["movieId"].where(
            movies["movieId"].isin(movie_encoder.classes_),
            other=movie_encoder.classes_[0]
        )
    )
    return ratings, movies, user_encoder, movie_encoder


def add_temporal_features(ratings):
    ratings = ratings.copy()
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings["days_since_epoch"] = (ratings["datetime"] - pd.Timestamp("1970-01-01")).dt.days
    max_day = ratings["days_since_epoch"].max()
    ratings["recency"] = (ratings["days_since_epoch"] / max_day).astype(np.float32)
    return ratings


def build_interaction_matrix(ratings, user_features, movie_features):
    df = ratings.merge(user_features, on="userId", how="left")
    df = df.merge(
        movie_features[["movieId", "movie_mean_rating", "movie_rating_count"]],
        on="movieId", how="left"
    )
    df["label"] = (df["rating"] >= 4.0).astype(np.float32)
    return df


def save_genre_matrix(movie_features, genre_vocab, path):
    genre_dim = len(genre_vocab)
    n = len(movie_features)
    matrix = np.zeros((n, genre_dim), dtype=np.float32)
    for i, vec in enumerate(movie_features["genre_vector"].values):
        matrix[i] = vec
    np.save(path, matrix)
    print(f"  Genre matrix saved: {matrix.shape} → {path}")
    return matrix


def run_feature_pipeline():
    print("Loading raw data...")
    ratings, movies = load_raw_data()

    print("Building genre vocabulary...")
    genre_vocab = build_genre_vocab(movies)
    print(f"  {len(genre_vocab)} unique genres found")

    print("Building user features...")
    user_features = build_user_features(ratings)

    print("Building movie features...")
    movie_features = build_movie_features(ratings, movies, genre_vocab)

    print("Encoding IDs...")
    ratings, movies, user_encoder, movie_encoder = encode_ids(ratings, movies)

    print("Adding temporal features...")
    ratings = add_temporal_features(ratings)

    print("Building interaction dataframe...")
    interactions = build_interaction_matrix(ratings, user_features, movie_features)

    os.makedirs("data/processed", exist_ok=True)

    interactions.drop(columns=["genre_vector"], errors="ignore").to_parquet(
        "data/processed/interactions.parquet", index=False
    )

    movie_features_save = movie_features.drop(columns=["genre_vector", "genres"], errors="ignore")
    movie_features_save.to_parquet("data/processed/movie_features.parquet", index=False)
    user_features.to_parquet("data/processed/user_features.parquet", index=False)

    save_genre_matrix(
        movie_features, genre_vocab,
        "data/processed/genre_matrix.npy"
    )

    import json
    with open("data/processed/genre_vocab.json", "w") as f:
        json.dump(genre_vocab, f)
    print("  Genre vocab saved → data/processed/genre_vocab.json")

    print("\nFeature pipeline complete.")
    print(f"  Interactions: {interactions.shape}")
    print(f"  Users:        {user_features.shape[0]}")
    print(f"  Movies:       {movie_features.shape[0]}")
    print(f"  Genres:       {len(genre_vocab)}")
    print(f"  Positive labels: {interactions['label'].sum():.0f} ({interactions['label'].mean()*100:.1f}%)")

    return interactions, user_features, movie_features, genre_vocab, user_encoder, movie_encoder


if __name__ == "__main__":
    run_feature_pipeline()