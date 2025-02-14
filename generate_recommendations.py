import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
import logging
from typing import List, Tuple, Dict
from sklearn.preprocessing import MultiLabelBinarizer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(ratings_path: str, movies_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the MovieLens dataset.

    Args:
        ratings_path (str): Path to the ratings CSV file.
        movies_path (str): Path to the movies CSV file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Ratings and movies DataFrames.
    """
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"Ratings file not found: {ratings_path}")
    if not os.path.exists(movies_path):
        raise FileNotFoundError(f"Movies file not found: {movies_path}")

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    return ratings, movies


def preprocess_data(ratings: pd.DataFrame, movies: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int], List[str], pd.DataFrame]:
    """
    Preprocess the dataset by mapping user and movie IDs to indices and adding genres.

    Args:
        ratings (pd.DataFrame): Ratings DataFrame.
        movies (pd.DataFrame): Movies DataFrame.

    Returns:
        Tuple[pd.DataFrame, Dict[int, int], Dict[int, int], List[str], pd.DataFrame]:
            - Processed ratings DataFrame with genres.
            - User ID to index mapping.
            - Movie ID to index mapping.
            - List of genre columns.
            - Movies DataFrame with genre features.
    """
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


def get_recent_movies_ratings(movies: pd.DataFrame) -> Tuple[List[int], List[float]]:
    """
    Ask the user to input recently watched movies and their ratings.

    Args:
        movies (pd.DataFrame): DataFrame containing movie information.

    Returns:
        Tuple[List[int], List[float]]: List of movie IDs and corresponding ratings.
    """
    recent_movies = []
    recent_ratings = []
    print("\nSince you're a new user, please help us understand your preferences.")
    print("Enter the names of a few movies you've recently watched and rate them (out of 5).")
    print("Press Enter on movie name to finish.")

    while True:
        movie_name = input("Enter a movie name: ").strip()
        if not movie_name:
            if not recent_movies:
                print("Please enter at least one movie.")
                continue
            else:
                break
        # Find the movie ID from the movies dataset
        movie_match = movies[movies['title'].str.contains(movie_name, case=False, na=False)]
        if movie_match.empty:
            print(f"Movie '{movie_name}' not found in the dataset. Please try again.")
            continue
        # If multiple matches, ask the user to select one
        if len(movie_match) > 1:
            print(f"Multiple matches found for '{movie_name}'. Please select one:")
            for i, row in movie_match.iterrows():
                print(f"{row['movieId']}: {row['title']} ({row['genres']})")
            movie_id = int(input("Enter the movieId of the correct movie: "))
        else:
            movie_id = movie_match.iloc[0]['movieId']
        while True:
            try:
                rating = float(input(f"Rate '{movie_name}' (out of 5): "))
                if 0 <= rating <= 5:
                    break
                else:
                    print("Rating must be between 0 and 5. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        recent_movies.append(movie_id)
        recent_ratings.append(rating)

    return recent_movies, recent_ratings


def create_temp_user_profile(model, recent_movies: List[int], recent_ratings: List[float],
                             movies_with_genres: pd.DataFrame, genre_columns: List[str],
                             movie_id_map: Dict[int, int]) -> np.ndarray:
    """
    Create a temporary user profile based on recently watched movies and ratings.

    Args:
        model: The trained GMF + MLP model.
        recent_movies (List[int]): List of recently watched movie IDs.
        recent_ratings (List[float]): List of corresponding ratings.
        movies_with_genres (pd.DataFrame): DataFrame containing movie information with genres.
        genre_columns (List[str]): List of genre columns.
        movie_id_map (Dict[int, int]): Mapping of movie IDs to indices.

    Returns:
        np.ndarray: Temporary user embedding.
    """
    # Get genre features for the recently watched movies
    recent_movie_indices = [movie_id_map[movie_id] for movie_id in recent_movies]
    recent_genre_features = movies_with_genres[movies_with_genres['movieId'].isin(recent_movies)][genre_columns].values

    # Get the output shape of the GMF user embedding layer
    gmf_user_embedding_layer = model.get_layer('gmf_user_embedding')
    embedding_size = gmf_user_embedding_layer.output.shape[-1]  # Correctly access output shape

    # Initialize user embedding
    user_embedding = np.zeros((1, embedding_size))

    # Predict user embedding based on recent movies and ratings
    for movie_idx, genre_features, rating in zip(recent_movie_indices, recent_genre_features, recent_ratings):
        # Use the model to predict the user embedding (simplified approach)
        user_embedding += model.predict([np.array([0]), np.array([movie_idx]), genre_features.reshape(1, -1)]) * rating
    user_embedding /= len(recent_movies)  # Average the embeddings

    # Normalize the user embedding
    user_embedding = user_embedding / np.linalg.norm(user_embedding)
    return user_embedding


def generate_top_recommendations(model, user_embedding: np.ndarray, movie_id_map: Dict[int, int],
                                 movies_with_genres: pd.DataFrame, genre_columns: List[str],
                                 recent_movies: List[int], top_k: int = 10) -> pd.DataFrame:
    """
    Generate top-k movie recommendations for a user.

    Args:
        model: The trained GMF + MLP model.
        user_embedding (np.ndarray): Temporary user embedding.
        movie_id_map (Dict[int, int]): Mapping of movie IDs to indices.
        movies_with_genres (pd.DataFrame): DataFrame containing movie information with genres.
        genre_columns (List[str]): List of genre columns.
        recent_movies (List[int]): List of recently watched movie IDs.
        top_k (int): Number of recommendations to generate.

    Returns:
        pd.DataFrame: DataFrame containing top-k recommendations.
    """
    # Get all movies and their genre features
    movie_ids = movies_with_genres['movieId'].unique()
    movie_indices = []
    valid_movie_ids = []

    # Filter out movie IDs that are not in movie_id_map
    for movie_id in movie_ids:
        if movie_id in movie_id_map:
            movie_indices.append(movie_id_map[movie_id])
            valid_movie_ids.append(movie_id)

    genre_features = movies_with_genres[movies_with_genres['movieId'].isin(valid_movie_ids)][genre_columns].values

    # Predict ratings for all movies
    predicted_ratings = np.array(
        [np.dot(user_embedding, model.get_layer('gmf_movie_embedding').get_weights()[0][movie_idx]) for movie_idx in
         movie_indices])
    predicted_ratings = predicted_ratings.flatten()  # Flatten to 1-dimensional array

    # Scale predicted ratings to the range of 0 to 5
    predicted_ratings = 1 + 4 * (predicted_ratings - np.min(predicted_ratings)) / (
                np.max(predicted_ratings) - np.min(predicted_ratings))

    # Create a DataFrame with movie IDs and predicted ratings
    recommendations = pd.DataFrame({
        'movieId': valid_movie_ids,
        'predicted_rating': predicted_ratings
    })

    # Merge with movie titles and genres
    recommendations = recommendations.merge(movies_with_genres[['movieId', 'title', 'genres']], on='movieId',
                                            how='left')

    # Remove movies the user has already entered
    recommendations = recommendations[~recommendations['movieId'].isin(recent_movies)]

    # Sort by predicted rating and get top-k recommendations
    top_recommendations = recommendations.sort_values(by='predicted_rating', ascending=False).head(top_k)
    return top_recommendations


def main():
    # Load the trained model
    logger.info("Loading the trained model...")
    try:
        model = load_model(os.getenv('MODEL_PATH'))
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    ratings, movies = load_data(os.getenv('RATINGS_PATH'), os.getenv('MOVIES_PATH'))
    ratings_with_genres, user_id_map, movie_id_map, genre_columns, movies_with_genres = preprocess_data(ratings, movies)

    # Get recently watched movies and ratings from the user
    recent_movies, recent_ratings = get_recent_movies_ratings(movies_with_genres)

    # Create a temporary user profile
    logger.info("Creating a temporary user profile...")
    user_embedding = create_temp_user_profile(model, recent_movies, recent_ratings, movies_with_genres, genre_columns,
                                              movie_id_map)

    # Generate top recommendations
    logger.info("Generating top 10 recommendations...")
    top_recommendations = generate_top_recommendations(model, user_embedding, movie_id_map, movies_with_genres,
                                                       genre_columns, recent_movies, top_k=10)

    # Display the recommendations
    print("\nTop 10 Recommendations:")
    for i, row in top_recommendations.iterrows():
        print(f"{i + 1}. {row['title']} (Predicted Rating: {row['predicted_rating']:.2f}, Genres: {row['genres']})")


if __name__ == '__main__':
    main()