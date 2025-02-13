import json
import os
import logging
from dotenv import load_dotenv
from preprocessing import load_data, split_data, preprocess_data, extract_features_labels
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model  # Import plot_model

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load hyperparameters from .env
EMBEDDING_SIZE = int(os.getenv('EMBEDDING_SIZE', 50))
MLP_HIDDEN_UNITS = list(map(int, os.getenv('MLP_HIDDEN_UNITS', '128,64').split(',')))
DROPOUT_RATE = float(os.getenv('DROPOUT_RATE', 0.2))
EPOCHS = int(os.getenv('EPOCHS', 10))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 64))


def build_model(num_users, num_movies, num_genres, embedding_size=EMBEDDING_SIZE, mlp_hidden_units=MLP_HIDDEN_UNITS,
                dropout_rate=DROPOUT_RATE):
    """Build the GMF + MLP hybrid model."""
    # Input layers
    user_input = Input(shape=(1,), name='user_input')
    movie_input = Input(shape=(1,), name='movie_input')
    genre_input = Input(shape=(num_genres,), name='genre_input')

    # GMF Branch
    gmf_user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size, name='gmf_user_embedding')(
        user_input)
    gmf_movie_embedding = Embedding(input_dim=num_movies, output_dim=embedding_size, name='gmf_movie_embedding')(
        movie_input)
    gmf_user_vector = Flatten()(gmf_user_embedding)
    gmf_movie_vector = Flatten()(gmf_movie_embedding)
    gmf_output = Multiply()([gmf_user_vector, gmf_movie_vector])  # Element-wise product

    # MLP Branch
    mlp_user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size, name='mlp_user_embedding')(
        user_input)
    mlp_movie_embedding = Embedding(input_dim=num_movies, output_dim=embedding_size, name='mlp_movie_embedding')(
        movie_input)
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

    # Print model summary
    model.summary()

    # Save the model architecture as a .png file
    plot_model(model, to_file=os.getenv('MODEL_ARCHITECTURE_PATH'), show_shapes=True, show_layer_names=True)
    logger.info(f"Model architecture saved as {os.getenv('MODEL_ARCHITECTURE_PATH')}")

    return model


def train_model(model, train_users, train_movies, train_genres, train_ratings, test_users, test_movies, test_genres,
                test_ratings, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Train the GMF + MLP model with callbacks."""
    # Define callbacks
    checkpoint = ModelCheckpoint(
        os.getenv('MODEL_PATH'),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    # Train the model
    history = model.fit(
        [train_users, train_movies, train_genres],
        train_ratings,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([test_users, test_movies, test_genres], test_ratings),
        callbacks=[checkpoint, early_stopping]
    )
    return history


def save_model_hist(model, hist):
    """Save the trained model and training history to disk."""
    try:
        # Save the model
        model.save(os.getenv('MODEL_PATH'))

        # Save the training history
        with open(os.getenv('TRAINING_HISTORY_PATH'), 'w') as f:
            json.dump(hist.history, f)
        logger.info(f"Model and training history saved...")
    except Exception as e:
        logger.error(f"Error saving model and history: {e}")


def main():
    logger.info("Loading and preprocessing data...")
    ratings, movies = load_data(os.getenv('RATINGS_PATH'), os.getenv('MOVIES_PATH'))
    ratings_with_genres, user_id_map, movie_id_map, genre_columns = preprocess_data(ratings, movies)
    train_data, test_data = split_data(ratings_with_genres)
    train_users, train_movies, train_genres, train_ratings = extract_features_labels(train_data, genre_columns)
    test_users, test_movies, test_genres, test_ratings = extract_features_labels(test_data, genre_columns)

    logger.info("Building the model...")
    num_users = len(user_id_map)
    num_movies = len(movie_id_map)
    num_genres = len(genre_columns)
    model = build_model(num_users, num_movies, num_genres)

    logger.info("Training the model...")
    hist = train_model(model,
                       train_users, train_movies, train_genres, train_ratings,
                       test_users, test_movies, test_genres, test_ratings
                       )

    logger.info("Saving the model and training history...")
    save_model_hist(model, hist)


if __name__ == '__main__':
    main()