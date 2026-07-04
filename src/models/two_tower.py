import torch
import torch.nn as nn


class UserTower(nn.Module):
    def __init__(self, num_users, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_users, embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(embedding_dim + 3, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32)
        )

    def forward(self, user_idx, mean_rating, rating_count, rating_std):
        emb = self.embedding(user_idx)
        x   = torch.cat([emb, mean_rating, rating_count, rating_std], dim=1)
        return self.network(x)


class MovieTower(nn.Module):
    def __init__(self, num_movies, genre_dim=20, embedding_dim=64):
        super().__init__()
        self.embedding    = nn.Embedding(num_movies, embedding_dim)
        self.genre_encode = nn.Sequential(nn.Linear(genre_dim, 32), nn.ReLU())
        self.network = nn.Sequential(
            nn.Linear(embedding_dim + 32 + 2, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32)
        )

    def forward(self, movie_idx, genre_vector, mean_rating, rating_count):
        emb   = self.embedding(movie_idx)
        genre = self.genre_encode(genre_vector)
        x     = torch.cat([emb, genre, mean_rating, rating_count], dim=1)
        return self.network(x)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_movies, genre_dim=20, embedding_dim=64):
        super().__init__()
        self.user_tower  = UserTower(num_users, embedding_dim)
        self.movie_tower = MovieTower(num_movies, genre_dim, embedding_dim)
        self.output = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, user_idx, user_mean, user_count, user_std,
                movie_idx, genre_vector, movie_mean, movie_count, recency):
        user_vec  = self.user_tower(user_idx, user_mean, user_count, user_std)
        movie_vec = self.movie_tower(movie_idx, genre_vector, movie_mean, movie_count)

        user_vec  = nn.functional.normalize(user_vec,  dim=1)
        movie_vec = nn.functional.normalize(movie_vec, dim=1)

        dot = (user_vec * movie_vec).sum(dim=1, keepdim=True)
        x   = torch.cat([dot, recency], dim=1)
        return self.output(x)


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = TwoTowerModel(610, 9724, 20, 64).to(device)

    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"User tower:       {sum(p.numel() for p in model.user_tower.parameters()):,}")
    print(f"Movie tower:      {sum(p.numel() for p in model.movie_tower.parameters()):,}")
    print("\nTwo-tower model built successfully on", device)
