import os, sys, numpy as np, pandas as pd, pickle, torch
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
import mlflow, warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath("."))
from src.models.two_tower import TwoTowerModel

PROCESSED_DIR = "data/processed"
MODEL_DIR     = "data/models"
K             = 10
DEVICE        = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_everything():
    df           = pd.read_parquet(os.path.join(PROCESSED_DIR, "interactions.parquet"))
    movie_feat   = pd.read_parquet(os.path.join(PROCESSED_DIR, "movie_features.parquet"))
    genre_matrix = np.load(os.path.join(PROCESSED_DIR, "genre_matrix.npy"))
    movie_feat   = movie_feat.reset_index(drop=True)
    movie_feat["movie_idx"] = movie_feat.index
    with open(os.path.join(MODEL_DIR, "gb_model.pkl"), "rb") as f: gb_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"),   "rb") as f: scaler   = pickle.load(f)
    num_users  = int(df["user_idx"].max()) + 1
    num_movies = int(df["movie_idx"].max()) + 1
    nn_model = TwoTowerModel(num_users, num_movies, 20, 64).to(DEVICE)
    nn_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "two_tower_final.pt"), map_location=DEVICE))
    nn_model.eval()
    return df, movie_feat, genre_matrix, gb_model, scaler, nn_model


def precision_at_k(top_k, gt): return len(set(top_k) & gt) / K
def recall_at_k(top_k, gt):    return len(set(top_k) & gt) / len(gt) if gt else 0.0
def ndcg_at_k(top_k, gt):
    dcg   = sum(1.0/np.log2(i+2) for i,m in enumerate(top_k) if m in gt)
    ideal = sum(1.0/np.log2(i+2) for i in range(min(len(gt), K)))
    return dcg/ideal if ideal > 0 else 0.0


def score_gb(u, movies, genres, gb_model, scaler):
    n = len(movies)
    X = np.column_stack([
        np.full(n, u["user_idx"]),         movies["movie_idx"].values,
        np.full(n, u["user_mean_rating"]),  np.full(n, u["user_rating_count"]),
        np.full(n, u["user_rating_std"]),   movies["movie_mean_rating"].values,
        movies["movie_rating_count"].values, np.full(n, u["recency"]), genres,
    ]).astype(np.float32)
    return gb_model.predict_proba(scaler.transform(X))[:, 1]


def score_nn(u, movies, genres, nn_model):
    n = len(movies)
    with torch.no_grad():
        return nn_model(
            torch.full((n,),  u["user_idx"],          dtype=torch.long).to(DEVICE),
            torch.full((n,1), u["user_mean_rating"],  dtype=torch.float32).to(DEVICE),
            torch.full((n,1), u["user_rating_count"], dtype=torch.float32).to(DEVICE),
            torch.full((n,1), u["user_rating_std"],   dtype=torch.float32).to(DEVICE),
            torch.tensor(movies["movie_idx"].values, dtype=torch.long).to(DEVICE),
            torch.tensor(genres, dtype=torch.float32).to(DEVICE),
            torch.tensor(movies["movie_mean_rating"].values,  dtype=torch.float32).unsqueeze(1).to(DEVICE),
            torch.tensor(movies["movie_rating_count"].values, dtype=torch.float32).unsqueeze(1).to(DEVICE),
            torch.full((n,1), u["recency"], dtype=torch.float32).to(DEVICE),
        ).cpu().numpy().flatten()


def run_ab_test():
    print("Loading data and models...")
    df, movie_feat, genre_matrix, gb_model, scaler, nn_model = load_everything()
    mid_to_row = {mid: i for i, mid in enumerate(movie_feat["movieId"].values)}

    liked_df = df[df["label"] == 1].copy()
    user_liked_counts = liked_df.groupby("userId").size()
    eval_users = user_liked_counts[user_liked_counts >= 20].index.tolist()

    print(f"Evaluating {len(eval_users)} users (20% holdout strategy)")
    print(f"\n{'User':>6} {'n_hold':>7} {'GB P@10':>9} {'NN P@10':>9} {'GB NDCG':>9} {'NN NDCG':>9}")
    print("-" * 55)

    results = []

    for i, uid in enumerate(eval_users):
        user_liked = liked_df[liked_df["userId"] == uid]["movieId"].values
        np.random.seed(42)
        np.random.shuffle(user_liked)

        n_holdout  = max(2, int(len(user_liked) * 0.20))
        holdout    = set(user_liked[:n_holdout])
        train_seen = set(df[df["userId"] == uid]["movieId"].values) - holdout

        cand_movies = movie_feat[movie_feat["movieId"].isin(mid_to_row.keys())].copy()
        cand_movies = cand_movies[~cand_movies["movieId"].isin(train_seen)].reset_index(drop=True)
        cand_genres = np.array([genre_matrix[mid_to_row[m]] for m in cand_movies["movieId"].values])

        if len(cand_movies) < K:
            continue

        user_row   = df[df["userId"] == uid].iloc[0]
        u = {
            "user_idx":          int(user_row["user_idx"]),
            "user_mean_rating":  float(user_row["user_mean_rating"]),
            "user_rating_count": float(user_row["user_rating_count"]),
            "user_rating_std":   float(user_row["user_rating_std"]),
            "recency":           float(user_row["recency"]),
        }

        gb_scores = score_gb(u, cand_movies, cand_genres, gb_model, scaler)
        nn_scores = score_nn(u, cand_movies, cand_genres, nn_model)

        gb_top = cand_movies["movieId"].values[np.argsort(-gb_scores)[:K]].tolist()
        nn_top = cand_movies["movieId"].values[np.argsort(-nn_scores)[:K]].tolist()

        row = {
            "user_id":      uid,
            "n_holdout":    n_holdout,
            "gb_precision": precision_at_k(gb_top, holdout),
            "nn_precision": precision_at_k(nn_top, holdout),
            "gb_recall":    recall_at_k(gb_top,    holdout),
            "nn_recall":    recall_at_k(nn_top,    holdout),
            "gb_ndcg":      ndcg_at_k(gb_top,      holdout),
            "nn_ndcg":      ndcg_at_k(nn_top,      holdout),
        }
        results.append(row)

        if i % 10 == 0:
            print(f"{uid:>6} {n_holdout:>7} {row['gb_precision']:>9.3f} {row['nn_precision']:>9.3f} "
                  f"{row['gb_ndcg']:>9.3f} {row['nn_ndcg']:>9.3f}")

    results_df = pd.DataFrame(results)

    print(f"\n{'='*60}")
    print(f"A/B TEST FINAL RESULTS  (K={K}, users={len(results_df)}, 20% holdout)")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Gradient Boosting':>18} {'Two-Tower Neural':>18} {'Winner':>10}")
    print("-" * 68)

    for label, gc, nc in [
        ("Precision@10", "gb_precision", "nn_precision"),
        ("Recall@10",    "gb_recall",    "nn_recall"),
        ("NDCG@10",      "gb_ndcg",      "nn_ndcg"),
    ]:
        gm = results_df[gc].mean()
        nm = results_df[nc].mean()
        diff = ((nm - gm) / gm * 100) if gm > 0 else 0
        w  = "GB ✅" if gm > nm else f"Neural ✅ (+{diff:.1f}%)"
        print(f"{label:<20} {gm:>18.4f} {nm:>18.4f} {w:>16}")

    t_stat, p_val = stats.ttest_rel(results_df["gb_ndcg"], results_df["nn_ndcg"])
    sig = "✅ Significant" if p_val < 0.05 else "❌ Not significant"
    print(f"\nPaired t-test on NDCG@10:  t={t_stat:.4f}  p={p_val:.4f}  {sig}")

    users_with_hits_gb = (results_df["gb_ndcg"] > 0).sum()
    users_with_hits_nn = (results_df["nn_ndcg"] > 0).sum()
    print(f"Users with ≥1 hit in top10: GB={users_with_hits_gb}  Neural={users_with_hits_nn}")
    print(f"{'='*60}")

    os.makedirs("monitoring", exist_ok=True)
    results_df.to_csv("monitoring/ab_test_results.csv", index=False)
    plot_results(results_df, p_val, t_stat)

    mlflow.set_experiment("two-tower-recsys")
    with mlflow.start_run(run_name="ab_test_20pct_holdout"):
        for label, gc, nc in [
            ("precision","gb_precision","nn_precision"),
            ("recall",   "gb_recall",   "nn_recall"),
            ("ndcg",     "gb_ndcg",     "nn_ndcg"),
        ]:
            mlflow.log_metric(f"gb_{label}_at_{K}", results_df[gc].mean())
            mlflow.log_metric(f"nn_{label}_at_{K}", results_df[nc].mean())
        mlflow.log_metric("t_statistic", float(t_stat) if not np.isnan(t_stat) else 0)
        mlflow.log_metric("p_value",     float(p_val)  if not np.isnan(p_val)  else 1)
        mlflow.log_artifact("monitoring/ab_test_results.csv")
    print(f"\nMLflow run logged.")
    print(f"Plots saved → monitoring/")
    return results_df


def plot_results(results_df, p_val, t_stat):
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(f"A/B Test: GB vs Two-Tower Neural  (K=10, 20% holdout, p={p_val:.4f})", fontsize=12)

    for ax, (title, gc, nc) in zip(axes, [
        ("Precision@10", "gb_precision", "nn_precision"),
        ("Recall@10",    "gb_recall",    "nn_recall"),
        ("NDCG@10",      "gb_ndcg",      "nn_ndcg"),
    ]):
        gm, nm = results_df[gc].mean(), results_df[nc].mean()
        bars = ax.bar(["GB", "Neural"], [gm, nm],
                      color=["#854F0B", "#534AB7"], edgecolor="black", width=0.5)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
        ax.set_title(title)
        ax.set_ylim(0, max(gm, nm) * 1.45 + 0.001)

    plt.tight_layout()
    plt.savefig("monitoring/ab_test_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle("Per-user NDCG@10 distribution", fontsize=12)

    axes2[0].hist(results_df["gb_ndcg"],  bins=15, alpha=0.7, color="#854F0B", label="GB")
    axes2[0].hist(results_df["nn_ndcg"],  bins=15, alpha=0.7, color="#534AB7", label="Neural")
    axes2[0].axvline(results_df["gb_ndcg"].mean(), color="#854F0B", linestyle="--", lw=2, label=f"GB mean={results_df['gb_ndcg'].mean():.4f}")
    axes2[0].axvline(results_df["nn_ndcg"].mean(), color="#534AB7", linestyle="--", lw=2, label=f"NN mean={results_df['nn_ndcg'].mean():.4f}")
    axes2[0].set_xlabel("NDCG@10")
    axes2[0].set_ylabel("Users")
    axes2[0].legend(fontsize=8)

    diff = results_df["nn_ndcg"] - results_df["gb_ndcg"]
    axes2[1].hist(diff, bins=15, color="steelblue", edgecolor="black", alpha=0.8)
    axes2[1].axvline(0,           color="black", linestyle="-",  lw=1.5, label="no difference")
    axes2[1].axvline(diff.mean(), color="red",   linestyle="--", lw=2,   label=f"mean diff={diff.mean():.4f}")
    axes2[1].set_title(f"Neural − GB per user  (t={t_stat:.3f}, p={p_val:.4f})")
    axes2[1].set_xlabel("NDCG difference (Neural − GB)")
    axes2[1].set_ylabel("Users")
    axes2[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("monitoring/ab_test_ndcg_dist.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_ab_test()
