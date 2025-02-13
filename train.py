import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout, Multiply
from tensorflow.keras.models import Model

def build_model(num_users, num_movies, num_genres, embedding_size=50, mlp_hidden_units=[128, 64], dropout_rate=0.2):
    """Build the GMF + MLP hybrid model."""
    # Input layers
    user_input = Input(shape=(1,), name='user_input')
    movie_input = Input(shape=(1,), name='movie_input')
    genre_input = Input(shape=(num_genres,), name='genre_input')

    # GMF Branch
    gmf_user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size, name='gmf_user_embedding')(user_input)
    gmf_movie_embedding = Embedding(input_dim=num_movies, output_dim=embedding_size, name='gmf_movie_embedding')(movie_input)
    gmf_user_vector = Flatten()(gmf_user_embedding)
    gmf_movie_vector = Flatten()(gmf_movie_embedding)
    gmf_output = Multiply()([gmf_user_vector, gmf_movie_vector])  # Element-wise product

    # MLP Branch
    mlp_user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size, name='mlp_user_embedding')(user_input)
    mlp_movie_embedding = Embedding(input_dim=num_movies, output_dim=embedding_size, name='mlp_movie_embedding')(movie_input)
    mlp_user_vector = Flatten()(mlp_user_embedding)
    mlp_movie_vector = Flatten()(mlp_movie_embedding)
    mlp_concat = Concatenate()([mlp_user_vector, mlp_movie_vector, genre_input])
    mlp_output = Dense(mlp_hidden_units[0], activation='relu')(mlp_concat)
    mlp_output = Dropout(dropout_rate)(mlp_output)
    mlp_output = Dense(mlp_hidden_units[1], activation='relu')(mlp_output)
    mlp_output = Dropout(dropout_rate)(mlp_output)

    # Combine GMF and MLP
    concat = Concatenate()([gmf_output, mlp_output])
    output = Dense(1, activation='linear')(concat)

    # Build the model
    model = Model(inputs=[user_input, movie_input, genre_input], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def train_model(model, train_users, train_movies, train_genres, train_ratings, test_users, test_movies, test_genres, test_ratings, epochs=10, batch_size=64):
    """Train the GMF + MLP model."""
    history = model.fit(
        [train_users, train_movies, train_genres],
        train_ratings,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([test_users, test_movies, test_genres], test_ratings)
    )
    return history

def save_model(model, file_path):
    """Save the trained model to a file."""
    model.save(file_path)

def main():
    # Load and preprocess data
    ratings, movies = load_data('ml-latest-small/ratings.csv', 'ml-latest-small/movies.csv')
    ratings_with_genres, user_id_map, movie_id_map, genre_columns = preprocess_data(ratings, movies)
    train_data, test_data = split_data(ratings_with_genres)
    train_users, train_movies, train_genres, train_ratings = extract_features_labels(train_data, genre_columns)
    test_users, test_movies, test_genres, test_ratings = extract_features_labels(test_data, genre_columns)

    # Build and train the model
    num_users = len(user_id_map)
    num_movies = len(movie_id_map)
    num_genres = len(genre_columns)
    model = build_model(num_users, num_movies, num_genres)
    hist = train_model(model,
                       train_users, train_movies, train_genres, train_ratings,
                       test_users, test_movies, test_genres, test_ratings
                       )

    # Save the model
    save_model(model, 'model/recommendation_model.keras')