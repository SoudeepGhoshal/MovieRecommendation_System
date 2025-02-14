import os
import logging
import pickle
import numpy as np
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load file paths from .env
RATINGS_PATH = os.getenv('RATINGS_PATH')
MOVIES_PATH = os.getenv('MOVIES_PATH')
PREPROCESSED_DATA_PATH = os.getenv('PREPROCESSED_DATA_PATH')
TEST_SIZE = float(os.getenv('TEST_SIZE', 0.2))
RANDOM_STATE = int(os.getenv('RANDOM_STATE', 42))


def load_data(ratings_file: str, movies_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the MovieLens dataset."""
    if not os.path.exists(ratings_file):
        raise FileNotFoundError(f"Ratings file not found: {ratings_file}")
    if not os.path.exists(movies_file):
        raise FileNotFoundError(f"Movies file not found: {movies_file}")

    ratings = pd.read_csv(ratings_file)
    movies = pd.read_csv(movies_file)
    return ratings, movies


def validate_data(ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
    """Validate the dataset."""
    if ratings.isnull().any().any():
        raise ValueError("Ratings data contains missing values.")
    if movies.isnull().any().any():
        raise ValueError("Movies data contains missing values.")
    if ratings.duplicated().any():
        raise ValueError("Ratings data contains duplicate entries.")
    if movies.duplicated().any():
        raise ValueError("Movies data contains duplicate entries.")


def preprocess_data(ratings: pd.DataFrame, movies: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int], List[str], pd.DataFrame]:
    """Preprocess the dataset by mapping user and movie IDs to indices and adding genres."""
    # Map user and movie IDs to indices
    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()

    user_id_map = {id: i for i, id in enumerate(user_ids)}
    movie_id_map = {id: i for i, id in enumerate(movie_ids)}

    ratings['user_idx'] = ratings['userId'].map(user_id_map)
    ratings['movie_idx'] = ratings['movieId'].map(movie_id_map)

    # Preprocess genres
    movies['genres'] = movies['genres'].str.split('|')
    mlb = MultiLabelBinarizer()
    genre_features = pd.DataFrame(mlb.fit_transform(movies['genres']), columns=mlb.classes_)
    genre_features['movieId'] = movies['movieId']

    # Merge genre features with movies
    movies_with_genres = movies.merge(genre_features, on='movieId', how='left')

    # Merge genre features with ratings
    ratings_with_genres = ratings.merge(genre_features, on='movieId', how='left')

    return ratings_with_genres, user_id_map, movie_id_map, mlb.classes_, movies_with_genres


def split_data(ratings: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into training and testing sets."""
    train_data, test_data = train_test_split(ratings, test_size=test_size, random_state=random_state)
    return train_data, test_data


def extract_features_labels(data: pd.DataFrame, genre_columns: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract features (user and movie indices, genres) and labels (ratings)."""
    users = data['user_idx'].values
    movies = data['movie_idx'].values
    genres = data[genre_columns].values
    ratings = data['rating'].values
    return users, movies, genres, ratings


def save_preprocessed_data(data: dict, file_path: str) -> None:
    """Save preprocessed data to a file."""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Preprocessed data saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save preprocessed data: {e}")
        raise


def load_preprocessed_data(file_path: str) -> dict:
    """Load preprocessed data from a file."""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load preprocessed data: {e}")
        raise


def main():
    # Load data
    ratings, movies = load_data(RATINGS_PATH, MOVIES_PATH)
    logger.info('Ratings DataFrame:\n%s', ratings.head())
    logger.info('Movies DataFrame:\n%s', movies.head())

    # Validate data
    validate_data(ratings, movies)

    # Preprocess data
    ratings_with_genres, user_id_map, movie_id_map, genre_columns, movies_with_genres = preprocess_data(ratings, movies)
    logger.info('Preprocessed Ratings DataFrame:\n%s', ratings_with_genres.head())
    logger.info('User ID Map:\n%s', user_id_map)
    logger.info('Movie ID Map:\n%s', movie_id_map)
    logger.info('Genre Columns:\n%s', genre_columns)

    # Save preprocessed data
    preprocessed_data = {
        'ratings_with_genres': ratings_with_genres,
        'user_id_map': user_id_map,
        'movie_id_map': movie_id_map,
        'genre_columns': genre_columns
    }
    save_preprocessed_data(preprocessed_data, PREPROCESSED_DATA_PATH)

    # Split data
    train_data, test_data = split_data(ratings_with_genres)
    train_users, train_movies, train_genres, train_ratings = extract_features_labels(train_data, genre_columns)
    test_users, test_movies, test_genres, test_ratings = extract_features_labels(test_data, genre_columns)


if __name__ == '__main__':
    main()