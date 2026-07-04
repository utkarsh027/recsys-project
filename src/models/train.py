import os
import sys
import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, log_loss
from sklearn.preprocessing import StandardScaler
import pickle

sys.path.append(os.path.abspath("."))

PROCESSED_DIR = "data/processed"
MODEL_DIR     = "data/models"
RANDOM_STATE  = 42
TEST_SIZE     = 0.2


def load_data():
    df           = pd.read_parquet(os.path.join(PROCESSED_DIR, "interactions.parquet"))
    movie_feat   = pd.read_parquet(os.path.join(PROCESSED_DIR, "movie_features.parquet"))
    genre_matrix = np.load(os.path.join(PROCESSED_DIR, "genre_matrix.npy"))
    print(f"Interactions:  {df.shape}")
    print(f"Genre matrix:  {genre_matrix.shape}")
    return df, movie_feat, genre_matrix


def build_features(df, movie_feat, genre_matrix):
    movie_id_to_row = dict(zip(movie_feat["movieId"].values, range(len(movie_feat))))
    genre_vectors = np.array([
        genre_matrix[movie_id_to_row.get(mid, 0)]
        for mid in df["movieId"].values
    ], dtype=np.float32)

    tabular = df[[
        "user_idx", "movie_idx",
        "user_mean_rating", "user_rating_count", "user_rating_std",
        "movie_mean_rating", "movie_rating_count", "recency"
    ]].values.astype(np.float32)

    X = np.hstack([tabular, genre_vectors])
    y = df["label"].values.astype(np.int32)
    print(f"Feature matrix: {X.shape}")
    return X, y


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df, movie_feat, genre_matrix = load_data()
    X, y = build_features(df, movie_feat, genre_matrix)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(y_train):,}   Test: {len(y_test):,}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    mlflow.set_experiment("two-tower-recsys")

    with mlflow.start_run(run_name="gradient_boosting_v1"):

        params = {
            "n_estimators":   200,
            "learning_rate":  0.1,
            "max_depth":      5,
            "subsample":      0.8,
            "random_state":   RANDOM_STATE,
        }
        mlflow.log_params(params)

        print("\nTraining Gradient Boosting model...")
        model = GradientBoostingClassifier(**params, verbose=1)
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred       = (y_pred_proba >= 0.5).astype(int)

        auc       = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        loss      = log_loss(y_test, y_pred_proba)

        print(f"\n{'='*40}")
        print(f"TEST RESULTS")
        print(f"{'='*40}")
        print(f"Loss:      {loss:.4f}")
        print(f"AUC:       {auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"{'='*40}")

        mlflow.log_metrics({
            "test_loss":      loss,
            "test_auc":       auc,
            "test_precision": precision,
            "test_recall":    recall,
        })

        model_path  = os.path.join(MODEL_DIR, "gb_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

        with open(model_path,  "wb") as f: pickle.dump(model,  f)
        with open(scaler_path, "wb") as f: pickle.dump(scaler, f)

        mlflow.sklearn.log_model(model, "gb_model")

        print(f"\nModel saved to {MODEL_DIR}")
        print(f"MLflow run logged.")

    return model, scaler


if __name__ == "__main__":
    train()
