from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
import logging
from generate_recommendations import load_data, preprocess_data, create_temp_user_profile, generate_top_recommendations
from tensorflow.keras.models import load_model

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
logger.info("Loading the trained model...")
try:
    model = load_model(os.getenv('MODEL_PATH'))
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Load and preprocess data
logger.info("Loading and preprocessing data...")
try:
    ratings, movies = load_data(os.getenv('RATINGS_PATH'), os.getenv('MOVIES_PATH'))
    ratings_with_genres, user_id_map, movie_id_map, genre_columns, movies_with_genres = preprocess_data(ratings, movies)
    logger.info("Data loaded and preprocessed successfully.")
except Exception as e:
    logger.error(f"Error loading or preprocessing data: {e}")
    raise

@app.route('/')
def home():
    """Serve the frontend HTML page with movies data."""
    movies_data = movies[['movieId', 'title']].to_dict('records')
    logger.info(f"Movies data sent to frontend: {movies_data[:5]}")  # Log first 5 entries
    return render_template('index.html', movies_data=movies_data)


@app.route('/status', methods=['GET'])
def status():
    """
    Status check endpoint to verify that the server is running and ready.

    Response JSON format:
    {
        "status": "ok",
        "message": "Server is running and ready to handle requests."
    }
    """
    return jsonify({
        "status": "ok",
        "message": "Server is running and ready to handle requests."
    })


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    API endpoint to generate movie recommendations.

    Request JSON format:
    {
        "recent_movies": [{"movieId": 1, "rating": 4.5}, {"movieId": 2, "rating": 3.0}]
    }

    Response JSON format:
    {
        "recommendations": [
            {"movieId": 3, "title": "Toy Story", "genres": "Adventure|Animation", "predicted_rating": 4.8},
            {"movieId": 4, "title": "Jumanji", "genres": "Adventure|Fantasy", "predicted_rating": 4.5}
        ]
    }
    """
    try:
        # Log the incoming request
        logger.info("Received recommendation request.")
        data = request.get_json()
        logger.info(f"Request data: {data}")

        # Parse request JSON
        recent_movies = [item['movieId'] for item in data['recent_movies']]
        recent_ratings = [item['rating'] for item in data['recent_movies']]

        # Log parsed data
        logger.info(f"Recent movies: {recent_movies}")
        logger.info(f"Recent ratings: {recent_ratings}")

        # Create a temporary user profile
        user_embedding = create_temp_user_profile(model, recent_movies, recent_ratings, movies_with_genres, genre_columns,
                                                 movie_id_map)

        # Generate top recommendations
        top_recommendations = generate_top_recommendations(model, user_embedding, movie_id_map, movies_with_genres,
                                                           genre_columns, recent_movies, top_k=10)

        # Format recommendations as JSON
        recommendations = []
        for _, row in top_recommendations.iterrows():
            recommendations.append({
                "movieId": int(row['movieId']),
                "title": row['title'],
                "genres": row['genres'],
                "predicted_rating": float(row['predicted_rating'])
            })

        return jsonify({"recommendations": recommendations})
    except KeyError as e:
        logger.error(f"KeyError: {e}")
        return jsonify({"error": "Invalid request format. Ensure 'movieId' and 'rating' keys are provided."}), 400
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": "Failed to generate recommendations. Please try again."}), 500


if __name__ == '__main__':
    # Run the Flask server
    app.run(host='0.0.0.0', port=5000)