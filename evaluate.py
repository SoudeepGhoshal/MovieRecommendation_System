import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import json
import logging
from dotenv import load_dotenv
from preprocessing import extract_features_labels, preprocess_data, split_data, load_data

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model, test_users, test_movies, test_genres, test_ratings):
    """Evaluate the model on the test set."""
    test_loss, test_mae = model.evaluate([test_users, test_movies, test_genres], test_ratings)
    logger.info(f'Test Loss: {test_loss}, Test MAE: {test_mae}')
    return test_loss, test_mae

def plot_training_history(history, file_path=os.getenv('PLOT_TRAINING_HISTORY_PATH')):
    """Plot training and validation loss and save the plot."""
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(file_path)
    logger.info(f"Training history plot saved to {file_path}")
    plt.show()

def predict_rating(model, user_idx, movie_idx, genre_features):
    """Predict the rating for a given user, movie, and genre features."""
    predicted_rating = model.predict([np.array([user_idx]), np.array([movie_idx]), genre_features.reshape(1, -1)])
    return predicted_rating[0][0]

def validate_data(ratings, movies):
    """Validate the dataset."""
    if ratings.isnull().any().any():
        raise ValueError("Ratings data contains missing values.")
    if movies.isnull().any().any():
        raise ValueError("Movies data contains missing values.")
    if ratings.duplicated().any():
        raise ValueError("Ratings data contains duplicate entries.")
    if movies.duplicated().any():
        raise ValueError("Movies data contains duplicate entries.")

def main():
    logger.info("Loading model and training history...")
    try:
        model = load_model(os.getenv('MODEL_PATH'))
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    try:
        with open(os.getenv('TRAINING_HISTORY_PATH'), 'r') as f:
            history = json.load(f)
    except Exception as e:
        logger.error(f"Error loading training history: {e}")
        return

    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    ratings, movies = load_data(os.getenv('RATINGS_PATH'), os.getenv('MOVIES_PATH'))
    validate_data(ratings, movies)
    ratings_with_genres, user_id_map, movie_id_map, genre_columns = preprocess_data(ratings, movies)
    train_data, test_data = split_data(ratings_with_genres)
    train_users, train_movies, train_genres, train_ratings = extract_features_labels(train_data, genre_columns)
    test_users, test_movies, test_genres, test_ratings = extract_features_labels(test_data, genre_columns)

    # Evaluate the model
    logger.info("Evaluating the model...")
    evaluate_model(model, test_users, test_movies, test_genres, test_ratings)

    # Plot training history
    logger.info("Plotting training history...")
    plot_training_history(history)

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