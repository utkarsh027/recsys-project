import tensorflow as tf
import numpy as np


def build_user_tower(num_users, embedding_dim=64):
    user_input       = tf.keras.Input(shape=(1,),  name="user_idx")
    mean_rating      = tf.keras.Input(shape=(1,),  name="user_mean_rating")
    rating_count     = tf.keras.Input(shape=(1,),  name="user_rating_count")
    rating_std       = tf.keras.Input(shape=(1,),  name="user_rating_std")

    user_embedding = tf.keras.layers.Embedding(
        num_users, embedding_dim, name="user_embedding"
    )(user_input)
    user_embedding = tf.keras.layers.Flatten()(user_embedding)

    user_features = tf.keras.layers.Concatenate()([
        user_embedding, mean_rating, rating_count, rating_std
    ])

    x = tf.keras.layers.Dense(128, activation="relu")(user_features)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    user_vector = tf.keras.layers.Dense(32, activation=None, name="user_vector")(x)

    return tf.keras.Model(
        inputs=[user_input, mean_rating, rating_count, rating_std],
        outputs=user_vector,
        name="user_tower"
    )


def build_movie_tower(num_movies, genre_dim=20, embedding_dim=64):
    movie_input  = tf.keras.Input(shape=(1,),         name="movie_idx")
    genre_input  = tf.keras.Input(shape=(genre_dim,), name="genre_vector")
    mean_rating  = tf.keras.Input(shape=(1,),         name="movie_mean_rating")
    rating_count = tf.keras.Input(shape=(1,),         name="movie_rating_count")

    movie_embedding = tf.keras.layers.Embedding(
        num_movies, embedding_dim, name="movie_embedding"
    )(movie_input)
    movie_embedding = tf.keras.layers.Flatten()(movie_embedding)

    genre_encoded = tf.keras.layers.Dense(32, activation="relu")(genre_input)

    movie_features = tf.keras.layers.Concatenate()([
        movie_embedding, genre_encoded, mean_rating, rating_count
    ])

    x = tf.keras.layers.Dense(128, activation="relu")(movie_features)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    movie_vector = tf.keras.layers.Dense(32, activation=None, name="movie_vector")(x)

    return tf.keras.Model(
        inputs=[movie_input, genre_input, mean_rating, rating_count],
        outputs=movie_vector,
        name="movie_tower"
    )


def build_two_tower_model(num_users, num_movies, genre_dim=20, embedding_dim=64):
    user_tower  = build_user_tower(num_users, embedding_dim)
    movie_tower = build_movie_tower(num_movies, genre_dim, embedding_dim)

    user_idx         = tf.keras.Input(shape=(1,),         name="user_idx")
    user_mean_rating = tf.keras.Input(shape=(1,),         name="user_mean_rating")
    user_rating_count= tf.keras.Input(shape=(1,),         name="user_rating_count")
    user_rating_std  = tf.keras.Input(shape=(1,),         name="user_rating_std")
    movie_idx        = tf.keras.Input(shape=(1,),         name="movie_idx")
    genre_vector     = tf.keras.Input(shape=(genre_dim,), name="genre_vector")
    movie_mean_rating= tf.keras.Input(shape=(1,),         name="movie_mean_rating")
    movie_rating_count=tf.keras.Input(shape=(1,),         name="movie_rating_count")
    recency          = tf.keras.Input(shape=(1,),         name="recency")

    user_vector = user_tower([
        user_idx, user_mean_rating, user_rating_count, user_rating_std
    ])
    movie_vector = movie_tower([
        movie_idx, genre_vector, movie_mean_rating, movie_rating_count
    ])

    dot_product = tf.keras.layers.Dot(axes=1, normalize=True)(
        [user_vector, movie_vector]
    )

    combined = tf.keras.layers.Concatenate()([dot_product, recency])
    x = tf.keras.layers.Dense(16, activation="relu")(combined)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(
        inputs=[
            user_idx, user_mean_rating, user_rating_count, user_rating_std,
            movie_idx, genre_vector, movie_mean_rating, movie_rating_count,
            recency
        ],
        outputs=output,
        name="two_tower_model"
    )

    return model, user_tower, movie_tower


def get_model_summary(num_users=610, num_movies=9742):
    model, user_tower, movie_tower = build_two_tower_model(num_users, num_movies)
    print("=== USER TOWER ===")
    user_tower.summary()
    print("\n=== MOVIE TOWER ===")
    movie_tower.summary()
    print("\n=== FULL TWO-TOWER MODEL ===")
    model.summary()
    return model


if __name__ == "__main__":
    get_model_summary()