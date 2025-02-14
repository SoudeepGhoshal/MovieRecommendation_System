import os
import numpy as np
import matplotlib.pyplot as plt
from keras import Model
from tensorflow.keras.models import load_model
import json
import logging
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, r2_score
from preprocessing import extract_features_labels, preprocess_data, split_data, load_data

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(model, test_users: np.ndarray, test_movies: np.ndarray, test_genres: np.ndarray,
                   test_ratings: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Evaluate the model on the test set.

    Args:
        model: The trained GMF + MLP model.
        test_users (np.ndarray): Test user indices.
        test_movies (np.ndarray): Test movie indices.
        test_genres (np.ndarray): Test genre features.
        test_ratings (np.ndarray): Test ratings.

    Returns:
        Tuple[float, float, float, float]: Test loss, MAE, RMSE, and R².
    """
    test_loss, test_mae, test_mse = model.evaluate([test_users, test_movies, test_genres], test_ratings)
    predictions = model.predict([test_users, test_movies, test_genres])
    test_rmse = np.sqrt(mean_squared_error(test_ratings, predictions))
    test_r2 = r2_score(test_ratings, predictions)
    logger.info(f'Test Loss: {test_loss}, Test MAE: {test_mae}, Test RMSE: {test_rmse}, Test R²: {test_r2}')
    return test_loss, test_mae, test_rmse, test_r2


def plot_training_history(history: Dict[str, List[float]], file_path: str = os.getenv('PLOT_TRAINING_HISTORY_PATH')) -> None:
    """
    Plot training and validation loss and save the plot.

    Args:
        history (Dict[str, List[float]]): Training history.
        file_path (str): Path to save the plot.
    """
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(file_path)
    logger.info(f"Training history plot saved to {file_path}")
    plt.show()


def plot_predictions_vs_actual(test_ratings: np.ndarray, predictions: np.ndarray, file_path: str) -> None:
    """
    Plot predicted vs. actual ratings and save the plot.

    Args:
        test_ratings (np.ndarray): Actual ratings.
        predictions (np.ndarray): Predicted ratings.
        file_path (str): Path to save the plot.
    """
    plt.scatter(test_ratings, predictions, alpha=0.5)
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Predicted vs. Actual Ratings')
    plt.savefig(file_path)
    logger.info(f"Predictions vs. actual plot saved to {file_path}")
    plt.show()


def predict_rating(model, user_idx: int, movie_idx: int, genre_features: np.ndarray) -> float:
    """
    Predict the rating for a given user, movie, and genre features.

    Args:
        model: The trained GMF + MLP model.
        user_idx (int): User index.
        movie_idx (int): Movie index.
        genre_features (np.ndarray): Genre features.

    Returns:
        float: Predicted rating.
    """
    predicted_rating = model.predict([np.array([user_idx]), np.array([movie_idx]), genre_features.reshape(1, -1)])
    return predicted_rating[0][0]


def validate_data(ratings, movies) -> None:
    """
    Validate the dataset.

    Args:
        ratings: Ratings DataFrame.
        movies: Movies DataFrame.
    """
    if ratings.isnull().any().any():
        raise ValueError("Ratings data contains missing values.")
    if movies.isnull().any().any():
        raise ValueError("Movies data contains missing values.")
    if ratings.duplicated().any():
        raise ValueError("Ratings data contains duplicate entries.")
    if movies.duplicated().any():
        raise ValueError("Movies data contains duplicate entries.")


def load_model_and_history(model_path: str, history_path: str) -> Tuple[Model, Dict[str, List[float]]]:
    """
    Load the trained model and training history.

    Args:
        model_path (str): Path to the saved model.
        history_path (str): Path to the training history.

    Returns:
        Tuple[Model, Dict[str, List[float]]]: Loaded model and training history.
    """
    try:
        model = load_model(model_path)
        with open(history_path, 'r') as f:
            history = json.load(f)
        return model, history
    except Exception as e:
        logger.error(f"Error loading model or history: {e}")
        raise


def main():
    logger.info("Loading model and training history...")
    model, history = load_model_and_history(os.getenv('MODEL_PATH'), os.getenv('TRAINING_HISTORY_PATH'))

    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    ratings, movies = load_data(os.getenv('RATINGS_PATH'), os.getenv('MOVIES_PATH'))
    validate_data(ratings, movies)
    ratings_with_genres, user_id_map, movie_id_map, genre_columns, movies_with_genres = preprocess_data(ratings, movies)
    train_data, test_data = split_data(ratings_with_genres)
    train_users, train_movies, train_genres, train_ratings = extract_features_labels(train_data, genre_columns)
    test_users, test_movies, test_genres, test_ratings = extract_features_labels(test_data, genre_columns)

    # Evaluate the model
    logger.info("Evaluating the model...")
    test_loss, test_mae, test_rmse, test_r2 = evaluate_model(model, test_users, test_movies, test_genres, test_ratings)

    # Plot training history
    logger.info("Plotting training history...")
    plot_training_history(history)

    # Plot predictions vs. actual ratings
    logger.info("Plotting predictions vs. actual ratings...")
    predictions = model.predict([test_users, test_movies, test_genres])
    plot_predictions_vs_actual(test_ratings, predictions, os.getenv('PREDICTIONS_VS_ACTUAL_PATH'))

    # Example prediction
    user_id = 10
    movie_id = 95
    user_idx = user_id_map.get(user_id, -1)
    movie_idx = movie_id_map.get(movie_id, -1)
    if user_idx != -1 and movie_idx != -1:
        movie_data = test_data[test_data['movieId'] == movie_id]
        if movie_data.empty:
            logger.error(f"Movie with ID {movie_id} not found in the dataset.")
        else:
            genre_features = movie_data[genre_columns].iloc[0].values
            predicted_rating = predict_rating(model, user_idx, movie_idx, genre_features)
            logger.info(f'Predicted Rating for User {user_id} and Movie {movie_id}: {predicted_rating}')
    else:
        logger.error('User or movie not found in the dataset.')


if __name__ == '__main__':
    main()