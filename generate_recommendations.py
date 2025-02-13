import numpy as np
import pandas as pd
import tensorflow as tf
from preprocessing import load_data, preprocess_data


def load_trained_model(model_path):
    """Load the trained GMF + MLP model."""
    model = tf.keras.models.load_model(model_path)
    return model


def get_user_input():
    """Get user input for userId and optional genre filter."""
    user_id = input("Enter your userId (or press Enter if you're a new user): ").strip()
    genre_filter = input("Enter a genre to filter recommendations (or press Enter to skip): ").strip().lower()
    return user_id, genre_filter


def get_recent_movies_ratings(movies):
    """Ask the user to input recently watched movies and their ratings."""
    recent_movies = []
    recent_ratings = []
    print("\nSince you're a new user, please help us understand your preferences.")
    print("Enter the names of a few movies you've recently watched and rate them (out of 5).")
    print("Press Enter on movie name to finish.")

    while True:
        movie_name = input("Enter a movie name: ").strip()
        if not movie_name:
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
        rating = float(input(f"Rate '{movie_name}' (out of 5): "))
        recent_movies.append(movie_id)
        recent_ratings.append(rating)

    return recent_movies, recent_ratings


def create_temp_user_profile(model, recent_movies, recent_ratings, movies, genre_columns, movie_id_map):
    """Create a temporary user profile based on recently watched movies and ratings."""
    # Get genre features for the recently watched movies
    recent_movie_indices = [movie_id_map[movie_id] for movie_id in recent_movies]
    recent_genre_features = movies[movies['movieId'].isin(recent_movies)][genre_columns].values

    # Predict user embedding based on recent movies and ratings
    user_embedding = np.zeros((1, model.get_layer('gmf_user_embedding').output_shape[-1]))
    for movie_idx, genre_features, rating in zip(recent_movie_indices, recent_genre_features, recent_ratings):
        # Use the model to predict the user embedding (simplified approach)
        user_embedding += model.predict([np.array([0]), np.array([movie_idx]), genre_features.reshape(1, -1)]) * rating
    user_embedding /= len(recent_movies)  # Average the embeddings

    return user_embedding


def generate_recommendations(model, user_id, genre_filter, ratings_file, movies_file, top_k=10):
    """Generate top-k movie recommendations for the user."""
    # Load and preprocess data
    ratings, movies = load_data(ratings_file, movies_file)
    ratings_with_genres, user_id_map, movie_id_map, genre_columns = preprocess_data(ratings, movies)

    # Check if user exists
    if user_id and int(user_id) in user_id_map:
        # Existing user
        user_idx = user_id_map[int(user_id)]
        user_indices = np.array([user_idx] * len(movie_id_map))
        recent_movies = []  # No recent movies for existing users
    else:
        # New user: Ask for recently watched movies and ratings
        recent_movies, recent_ratings = get_recent_movies_ratings(movies)
        if not recent_movies:
            print("No movies entered. Displaying popular movies instead.")
            return get_popular_movies(ratings, movies, genre_filter, top_k)
        # Create a temporary user profile
        user_embedding = create_temp_user_profile(model, recent_movies, recent_ratings, movies, genre_columns,
                                                  movie_id_map)
        user_indices = np.array([0] * len(movie_id_map))  # Use a dummy user index

    # Get all movies and their genre features
    movie_ids = ratings_with_genres['movieId'].unique()
    movie_indices = [movie_id_map[movie_id] for movie_id in movie_ids]
    genre_features = ratings_with_genres.drop_duplicates('movieId')[genre_columns].values

    # Predict ratings for all movies
    if user_id and int(user_id) in user_id_map:
        # Existing user: Use the model directly
        predicted_ratings = model.predict([user_indices, movie_indices, genre_features]).flatten()
    else:
        # New user: Use the temporary user embedding
        predicted_ratings = np.array(
            [np.dot(user_embedding, model.get_layer('gmf_movie_embedding').get_weights()[0][movie_idx]) for movie_idx in
             movie_indices])

    # Create a DataFrame with movie IDs and predicted ratings
    recommendations = pd.DataFrame({
        'movieId': movie_ids,
        'predicted_rating': predicted_ratings
    })

    # Merge with movie titles and genres
    recommendations = recommendations.merge(movies[['movieId', 'title', 'genres']], on='movieId', how='left')

    # Remove movies the user has already entered (for new users)
    if recent_movies:
        recommendations = recommendations[~recommendations['movieId'].isin(recent_movies)]

    # Filter by genre if specified
    if genre_filter:
        recommendations = recommendations[recommendations['genres'].str.lower().str.contains(genre_filter)]

    # Sort by predicted rating and get top-k recommendations
    top_recommendations = recommendations.sort_values(by='predicted_rating', ascending=False).head(top_k)
    return top_recommendations


def display_recommendations(recommendations):
    """Display the top-k recommendations."""
    if recommendations.empty:
        print("No recommendations found.")
    else:
        print("\nTop Recommendations:")
        for i, row in recommendations.iterrows():
            print(f"{i + 1}. {row['title']} (Predicted Rating: {row['predicted_rating']:.2f}, Genres: {row['genres']})")


def main():
    # Load the trained model
    model_path = 'gmf_mlp_movie_recommendation_model.h5'
    model = load_trained_model(model_path)

    # Get user input
    user_id, genre_filter = get_user_input()

    # Generate and display recommendations
    ratings_file = 'ml-latest-small/ratings.csv'
    movies_file = 'ml-latest-small/movies.csv'
    recommendations = generate_recommendations(model, user_id, genre_filter, ratings_file, movies_file)
    display_recommendations(recommendations)


if __name__ == '__main__':
    main()