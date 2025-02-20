import io
import json
import os
import logging
import sys
from typing import List
from dotenv import load_dotenv
import numpy as np
from keras.src.callbacks import History
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, Callback
from tensorflow.keras.utils import plot_model
from preprocessing import load_data, split_data, preprocess_data, extract_features_labels

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load hyperparameters from .env
EMBEDDING_SIZE = int(os.getenv('EMBEDDING_SIZE', 50))
MLP_HIDDEN_UNITS = list(map(int, os.getenv('MLP_HIDDEN_UNITS', '128,64').split(',')))
DROPOUT_RATE = float(os.getenv('DROPOUT_RATE', 0.2))
EPOCHS = int(os.getenv('EPOCHS', 50))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 64))


# Custom callback to save epoch logs to a text file
class EpochLogSaver(Callback):
    def __init__(self, log_file):
        super(EpochLogSaver, self).__init__()
        self.log_file = log_file

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {epoch + 1}\n")
            f.write(f" - loss: {logs.get('loss'):.4f}\n")
            f.write(f" - mae: {logs.get('mae'):.4f}\n")
            f.write(f" - mse: {logs.get('mse'):.4f}\n")
            f.write(f" - val_loss: {logs.get('val_loss'):.4f}\n")
            f.write(f" - val_mae: {logs.get('val_mae'):.4f}\n")
            f.write(f" - val_mse: {logs.get('val_mse'):.4f}\n")
            f.write(f" - learning_rate: {logs.get('learning_rate'):.4f}\n")
            f.write("\n")


def build_model(num_users: int, num_movies: int, num_genres: int, embedding_size: int = EMBEDDING_SIZE,
                mlp_hidden_units: List[int] = MLP_HIDDEN_UNITS, dropout_rate: float = DROPOUT_RATE) -> Model:
    """
    Build the GMF + MLP hybrid model.

    Args:
        num_users (int): Number of unique users.
        num_movies (int): Number of unique movies.
        num_genres (int): Number of unique genres.
        embedding_size (int): Size of the embedding layer.
        mlp_hidden_units (List[int]): List of hidden units for the MLP branch.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        Model: The compiled GMF + MLP model.
    """
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
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

    # Print model summary
    model.summary()

    # Capture model summary
    model_summary = io.StringIO()
    sys.stdout = model_summary  # Redirect stdout to capture summary
    model.summary()
    sys.stdout = sys.__stdout__  # Reset stdout

    # Save model summary to training logs
    log_file = os.path.join(os.getenv('LOGS_PATH'), 'training_logs.txt')
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=== Model Summary ===\n")
        f.write(model_summary.getvalue())
        f.write("=====================\n\n")

    # Save the model architecture as a .png file
    plot_model(model, to_file=os.getenv('MODEL_ARCHITECTURE_PATH'), show_shapes=True, show_layer_names=True)
    logger.info(f"Model architecture saved as {os.getenv('MODEL_ARCHITECTURE_PATH')}")

    return model


def train_model(model: Model, train_users: np.ndarray, train_movies: np.ndarray, train_genres: np.ndarray,
                train_ratings: np.ndarray, test_users: np.ndarray, test_movies: np.ndarray, test_genres: np.ndarray,
                test_ratings: np.ndarray, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE) -> History:
    """
    Train the GMF + MLP model with callbacks.

    Args:
        model (Model): The GMF + MLP model.
        train_users (np.ndarray): Training user indices.
        train_movies (np.ndarray): Training movie indices.
        train_genres (np.ndarray): Training genre features.
        train_ratings (np.ndarray): Training ratings.
        test_users (np.ndarray): Testing user indices.
        test_movies (np.ndarray): Testing movie indices.
        test_genres (np.ndarray): Testing genre features.
        test_ratings (np.ndarray): Testing ratings.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        History: Training history.
    """
    # Define callbacks
    tensorboard = TensorBoard(log_dir=os.getenv('LOGS_PATH'), histogram_freq=1)
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
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-5,
        verbose=1
    )

    # Train the model
    history = model.fit(
        [train_users, train_movies, train_genres],
        train_ratings,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([test_users, test_movies, test_genres], test_ratings),
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard, EpochLogSaver(os.getenv('LOGS_PATH')+'/training_logs.txt')]
    )
    return history


def save_model_hist(model: Model, hist: History) -> None:
    """Save the trained model and training history to disk."""
    try:
        # Save the model
        model.save(os.getenv('MODEL_PATH'))
        logger.info(f"Model saved to {os.getenv('MODEL_PATH')}")

        # Save the training history
        with open(os.getenv('TRAINING_HISTORY_PATH'), 'w') as f:
            json.dump(hist.history, f)
        logger.info(f"Training history saved to {os.getenv('TRAINING_HISTORY_PATH')}")
    except Exception as e:
        logger.error(f"Error saving model and history: {e}")
        raise


def main():
    logger.info("Loading and preprocessing data...")
    ratings, movies = load_data(os.getenv('RATINGS_PATH'), os.getenv('MOVIES_PATH'))
    ratings_with_genres, user_id_map, movie_id_map, genre_columns, movies_with_genres = preprocess_data(ratings, movies)
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