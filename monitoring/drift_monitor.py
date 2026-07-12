import os
import sys
import pandas as pd
import numpy as np
from evidently import Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently import Report
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath("."))

PROCESSED_DIR = "data/processed"
MONITORING_DIR = "monitoring"


def load_data():
    df           = pd.read_parquet(os.path.join(PROCESSED_DIR, "interactions.parquet"))
    movie_feat   = pd.read_parquet(os.path.join(PROCESSED_DIR, "movie_features.parquet"))
    genre_matrix = np.load(os.path.join(PROCESSED_DIR, "genre_matrix.npy"))
    return df, movie_feat, genre_matrix


def build_feature_df(df, movie_feat, genre_matrix):
    movie_id_to_row = {mid: i for i, mid in enumerate(movie_feat["movieId"].values)}

    genre_vectors = np.array([
        genre_matrix[movie_id_to_row.get(mid, 0)]
        for mid in df["movieId"].values
    ], dtype=np.float32)

    genre_df = pd.DataFrame(
        genre_vectors,
        columns=[f"genre_{i}" for i in range(genre_matrix.shape[1])]
    )

    feature_df = pd.concat([
        df[["user_mean_rating", "user_rating_count", "user_rating_std",
            "movie_mean_rating", "movie_rating_count", "recency", "label"]].reset_index(drop=True),
        genre_df.reset_index(drop=True)
    ], axis=1)

    return feature_df


def simulate_production_drift(feature_df):
    prod_df = feature_df.copy()
    np.random.seed(42)
    n = len(prod_df)

    prod_df["user_mean_rating"]  = prod_df["user_mean_rating"]  + np.random.normal(0.3,  0.1, n)
    prod_df["user_rating_count"] = prod_df["user_rating_count"] + np.random.normal(0.5,  0.2, n)
    prod_df["recency"]           = prod_df["recency"]           + np.random.normal(0.15, 0.05, n)
    prod_df["movie_mean_rating"] = prod_df["movie_mean_rating"] + np.random.normal(-0.2, 0.1, n)

    prod_df["user_mean_rating"]  = prod_df["user_mean_rating"].clip(0.5, 5.0)
    prod_df["recency"]           = prod_df["recency"].clip(0.0, 1.0)
    prod_df["movie_mean_rating"] = prod_df["movie_mean_rating"].clip(0.5, 5.0)

    return prod_df


def run_drift_report(reference_df, production_df):
    os.makedirs(MONITORING_DIR, exist_ok=True)

    columns = [
        "user_mean_rating", "user_rating_count", "user_rating_std",
        "movie_mean_rating", "movie_rating_count", "recency"
    ]

    ref  = reference_df[columns].sample(5000, random_state=42).reset_index(drop=True)
    prod = production_df[columns].sample(5000, random_state=99).reset_index(drop=True)

    definition = DataDefinition(
        numerical_columns=columns
    )

    ref_dataset  = Dataset.from_pandas(ref,  data_definition=definition)
    prod_dataset = Dataset.from_pandas(prod, data_definition=definition)

    report = Report(metrics=[DataDriftPreset()])
    my_eval = report.run(reference_data=ref_dataset, current_data=prod_dataset)

    html_path = os.path.join(MONITORING_DIR, "drift_report.html")
    my_eval.save_html(html_path)
    print(f"Drift report saved → {html_path}")
    return my_eval


def print_summary(feature_df, prod_df):
    print("\n" + "="*55)
    print("DRIFT MONITORING SUMMARY")
    print("="*55)
    print(f"{'Feature':<25} {'Train mean':>12} {'Prod mean':>12} {'Drift':>10}")
    print("-"*55)

    features = [
        "user_mean_rating", "user_rating_count",
        "user_rating_std",  "movie_mean_rating",
        "movie_rating_count", "recency"
    ]

    for feat in features:
        train_mean = feature_df[feat].mean()
        prod_mean  = prod_df[feat].mean()
        pct_change = abs(prod_mean - train_mean) / (abs(train_mean) + 1e-8) * 100
        flag = "⚠️  DRIFT" if pct_change > 5 else "✅ stable"
        print(f"{feat:<25} {train_mean:>12.4f} {prod_mean:>12.4f} {flag:>10}")

    print("="*55)
    print("\nWhat this means:")
    print("  ⚠️  DRIFT  → feature distribution has shifted significantly")
    print("              → model may need retraining")
    print("  ✅ stable  → feature distribution is consistent")
    print("              → model predictions remain reliable")
    print("\nFull HTML report → monitoring/drift_report.html")
    print("Open in browser:  open monitoring/drift_report.html")


if __name__ == "__main__":
    print("Loading data...")
    df, movie_feat, genre_matrix = load_data()

    print("Building feature dataframe...")
    feature_df = build_feature_df(df, movie_feat, genre_matrix)
    print(f"  Shape: {feature_df.shape}")

    print("Simulating production data with drift...")
    prod_df = simulate_production_drift(feature_df)

    print("Running Evidently drift report...")
    run_drift_report(feature_df, prod_df)

    print_summary(feature_df, prod_df)