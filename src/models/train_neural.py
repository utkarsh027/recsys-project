import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import mlflow

sys.path.append(os.path.abspath("."))
from src.models.two_tower import TwoTowerModel

PROCESSED_DIR = "data/processed"
MODEL_DIR     = "data/models"
EMBEDDING_DIM = 64
BATCH_SIZE    = 1024
EPOCHS        = 10
LEARNING_RATE = 0.001
TEST_SIZE     = 0.2
RANDOM_STATE  = 42

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on: {DEVICE}")


class RecDataset(Dataset):
    def __init__(self, df, movie_feat, genre_matrix, labels):
        movie_id_to_row = dict(zip(movie_feat["movieId"].values, range(len(movie_feat))))
        self.genre_vectors   = torch.tensor(
            np.array([genre_matrix[movie_id_to_row.get(mid, 0)] for mid in df["movieId"].values]),
            dtype=torch.float32
        )
        self.user_idx        = torch.tensor(df["user_idx"].values,          dtype=torch.long)
        self.user_mean       = torch.tensor(df["user_mean_rating"].values,  dtype=torch.float32).unsqueeze(1)
        self.user_count      = torch.tensor(df["user_rating_count"].values, dtype=torch.float32).unsqueeze(1)
        self.user_std        = torch.tensor(df["user_rating_std"].values,   dtype=torch.float32).unsqueeze(1)
        self.movie_idx       = torch.tensor(df["movie_idx"].values,         dtype=torch.long)
        self.movie_mean      = torch.tensor(df["movie_mean_rating"].values, dtype=torch.float32).unsqueeze(1)
        self.movie_count     = torch.tensor(df["movie_rating_count"].values,dtype=torch.float32).unsqueeze(1)
        self.recency         = torch.tensor(df["recency"].values,           dtype=torch.float32).unsqueeze(1)
        self.labels          = torch.tensor(labels,                         dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "user_idx":    self.user_idx[idx],
            "user_mean":   self.user_mean[idx],
            "user_count":  self.user_count[idx],
            "user_std":    self.user_std[idx],
            "movie_idx":   self.movie_idx[idx],
            "genre":       self.genre_vectors[idx],
            "movie_mean":  self.movie_mean[idx],
            "movie_count": self.movie_count[idx],
            "recency":     self.recency[idx],
            "label":       self.labels[idx],
        }


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        preds = model(
            batch["user_idx"].to(DEVICE),
            batch["user_mean"].to(DEVICE),
            batch["user_count"].to(DEVICE),
            batch["user_std"].to(DEVICE),
            batch["movie_idx"].to(DEVICE),
            batch["genre"].to(DEVICE),
            batch["movie_mean"].to(DEVICE),
            batch["movie_count"].to(DEVICE),
            batch["recency"].to(DEVICE),
        )
        loss = criterion(preds, batch["label"].to(DEVICE))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            preds = model(
                batch["user_idx"].to(DEVICE),
                batch["user_mean"].to(DEVICE),
                batch["user_count"].to(DEVICE),
                batch["user_std"].to(DEVICE),
                batch["movie_idx"].to(DEVICE),
                batch["genre"].to(DEVICE),
                batch["movie_mean"].to(DEVICE),
                batch["movie_count"].to(DEVICE),
                batch["recency"].to(DEVICE),
            )
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch["label"].numpy())

    all_preds  = np.vstack(all_preds).flatten()
    all_labels = np.vstack(all_labels).flatten()

    auc       = roc_auc_score(all_labels, all_preds)
    precision = precision_score(all_labels, (all_preds >= 0.5).astype(int))
    recall    = recall_score(all_labels,    (all_preds >= 0.5).astype(int))
    return auc, precision, recall, all_preds, all_labels


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df           = pd.read_parquet(os.path.join(PROCESSED_DIR, "interactions.parquet"))
    movie_feat   = pd.read_parquet(os.path.join(PROCESSED_DIR, "movie_features.parquet"))
    genre_matrix = np.load(os.path.join(PROCESSED_DIR, "genre_matrix.npy"))

    labels = df["label"].values.astype(np.float32)
    idx    = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )

    train_ds = RecDataset(df.iloc[train_idx].reset_index(drop=True), movie_feat, genre_matrix, labels[train_idx])
    test_ds  = RecDataset(df.iloc[test_idx].reset_index(drop=True),  movie_feat, genre_matrix, labels[test_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds):,}   Test: {len(test_ds):,}")

    num_users  = int(df["user_idx"].max()) + 1
    num_movies = int(df["movie_idx"].max()) + 1

    model     = TwoTowerModel(num_users, num_movies, 20, EMBEDDING_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    mlflow.set_experiment("two-tower-recsys")

    with mlflow.start_run(run_name="two_tower_pytorch_v1"):
        mlflow.log_params({
            "model":          "two_tower_pytorch",
            "embedding_dim":  EMBEDDING_DIM,
            "batch_size":     BATCH_SIZE,
            "epochs":         EPOCHS,
            "learning_rate":  LEARNING_RATE,
            "device":         str(DEVICE),
        })

        best_auc    = 0
        best_epoch  = 0
        no_improve  = 0

        print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val AUC':>10} {'Precision':>10} {'Recall':>10}")
        print("-" * 55)

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            val_auc, val_precision, val_recall, _, _ = evaluate(model, test_loader)
            scheduler.step(val_auc)

            print(f"{epoch:>6} {train_loss:>12.4f} {val_auc:>10.4f} {val_precision:>10.4f} {val_recall:>10.4f}")

            mlflow.log_metrics({
                "train_loss":    train_loss,
                "val_auc":       val_auc,
                "val_precision": val_precision,
                "val_recall":    val_recall,
            }, step=epoch)

            if val_auc > best_auc:
                best_auc   = val_auc
                best_epoch = epoch
                no_improve = 0
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, "two_tower_best.pt"))
            else:
                no_improve += 1
                if no_improve >= 3:
                    print(f"\nEarly stopping at epoch {epoch}. Best AUC: {best_auc:.4f} at epoch {best_epoch}")
                    break

        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "two_tower_best.pt")))
        final_auc, final_precision, final_recall, _, _ = evaluate(model, test_loader)

        print(f"\n{'='*45}")
        print(f"FINAL TEST RESULTS (best model epoch {best_epoch})")
        print(f"{'='*45}")
        print(f"AUC:       {final_auc:.4f}")
        print(f"Precision: {final_precision:.4f}")
        print(f"Recall:    {final_recall:.4f}")
        print(f"{'='*45}")

        mlflow.log_metrics({
            "test_auc":       final_auc,
            "test_precision": final_precision,
            "test_recall":    final_recall,
        })

        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "two_tower_final.pt"))
        print(f"\nModel saved → {MODEL_DIR}/two_tower_final.pt")

    return model


if __name__ == "__main__":
    train()
