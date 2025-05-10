import pandas as pd
import numpy as np
import torch
from torch import nn
import logging
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from hybrid_recommender import HybridRecommender
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovieRecommender:
    def __init__(self, movies_df=None, ratings_df=None, metadata_df=None):
        """Initialize the MovieRecommender with data loading and validation."""
        try:
            # Get the directory containing the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Load data if not provided
            if movies_df is None:
                movies_path = os.path.join(current_dir, 'movies.csv')
                if not os.path.exists(movies_path):
                    raise FileNotFoundError(f"movies.csv not found at {movies_path}")
                movies_df = pd.read_csv(movies_path)
                
            if ratings_df is None:
                ratings_path = os.path.join(current_dir, 'ratings.csv')
                if not os.path.exists(ratings_path):
                    raise FileNotFoundError(f"ratings.csv not found at {ratings_path}")
                ratings_df = pd.read_csv(ratings_path)
                
            if metadata_df is None:
                # First try to load the mapping file
                mapping_path = os.path.join(current_dir, 'movie_mapping.csv')
                if os.path.exists(mapping_path):
                    logger.info("Loading movie mapping file...")
                    metadata_df = pd.read_csv(mapping_path)
                else:
                    # Fall back to original metadata file
                    metadata_path = os.path.join(current_dir, 'movie_metadata.csv')
                    if not os.path.exists(metadata_path):
                        raise FileNotFoundError(f"movie_metadata.csv not found at {metadata_path}")
                    metadata_df = pd.read_csv(metadata_path)
                    
                    # Create mapping file if it doesn't exist
                    logger.info("Creating movie mapping file...")
                    from create_movie_mapping import create_movie_mapping
                    create_movie_mapping()
                    metadata_df = pd.read_csv(mapping_path)
                
                # Log column names for debugging
                logger.info(f"Metadata columns: {metadata_df.columns.tolist()}")
            
            # Validate and preprocess data
            self._validate_and_preprocess_data(movies_df, ratings_df, metadata_df)
            
            # Initialize hybrid recommender
            self.hybrid_recommender = HybridRecommender(movies_df, ratings_df, metadata_df)
            
            # Store dataframes
            self.movies_df = movies_df
            self.ratings_df = ratings_df
            self.metadata_df = metadata_df
            
            # Train the model
            self.train_model()
            
        except Exception as e:
            logger.error(f"Failed to initialize MovieRecommender: {str(e)}")
            raise
    
    def _validate_and_preprocess_data(self, movies_df, ratings_df, metadata_df):
        """Validate and preprocess the input dataframes."""
        try:
            # Validate required columns for movies and ratings
            required_movies_cols = {'movieId', 'title', 'genres'}
            required_ratings_cols = {'userId', 'movieId', 'rating'}
            
            missing_cols = []
            if not required_movies_cols.issubset(movies_df.columns):
                missing_cols.append(f"movies_df: {required_movies_cols - set(movies_df.columns)}")
            if not required_ratings_cols.issubset(ratings_df.columns):
                missing_cols.append(f"ratings_df: {required_ratings_cols - set(ratings_df.columns)}")
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
            # Validate data types
            if not pd.api.types.is_numeric_dtype(movies_df['movieId']):
                raise ValueError("movieId in movies_df must be numeric")
            if not pd.api.types.is_numeric_dtype(ratings_df['movieId']):
                raise ValueError("movieId in ratings_df must be numeric")
            if not pd.api.types.is_numeric_dtype(ratings_df['userId']):
                raise ValueError("userId in ratings_df must be numeric")
            if not pd.api.types.is_numeric_dtype(ratings_df['rating']):
                raise ValueError("rating in ratings_df must be numeric")
            
            # Validate rating range
            if not (0 <= ratings_df['rating'].min() <= ratings_df['rating'].max() <= 5):
                raise ValueError("Ratings must be between 0 and 5")
            
            # Handle missing values
            for df, name in [(movies_df, 'movies_df'), (ratings_df, 'ratings_df'), (metadata_df, 'metadata_df')]:
                null_cols = df.columns[df.isnull().any()].tolist()
                if null_cols:
                    logger.warning(f"Found null values in {name} columns: {null_cols}")
                    # Fill missing values appropriately
                    if name == 'movies_df':
                        df['genres'] = df['genres'].fillna('')
                    elif name == 'ratings_df':
                        df['rating'] = df['rating'].fillna(0)
                    elif name == 'metadata_df':
                        df = df.fillna('')
            
            # Clean and normalize titles
            def clean_title(title):
                if pd.isna(title):
                    return ''
                # Remove year in parentheses and clean up
                title = str(title).lower().strip()
                title = title.split('(')[0].strip()
                return title
            
            movies_df['clean_title'] = movies_df['title'].apply(clean_title)
            metadata_df['clean_title'] = metadata_df['movie_title'].apply(clean_title)
            
            # Create a mapping dictionary from movie titles to IDs
            title_to_id = dict(zip(movies_df['clean_title'], movies_df['movieId']))
            
            # Log some debug information
            logger.info(f"Number of movies in movies_df before mapping: {len(movies_df)}")
            logger.info(f"Number of movies in metadata_df before mapping: {len(metadata_df)}")
            
            # Map movie IDs using the clean titles
            metadata_df.loc[:, 'movieId'] = metadata_df['clean_title'].map(title_to_id)
            
            # Log mapping results
            matched_count = metadata_df['movieId'].notna().sum()
            logger.info(f"Number of movies matched: {matched_count}")
            
            if matched_count == 0:
                raise ValueError("No movies were matched between metadata and movies datasets")
            
            # Drop rows where movieId is NaN (no match found)
            metadata_df = metadata_df.dropna(subset=['movieId'])
            
            # Convert movieId to integer
            metadata_df.loc[:, 'movieId'] = metadata_df['movieId'].astype(int)
            
            # Clean up temporary columns
            metadata_df = metadata_df.drop('clean_title', axis=1)
            movies_df = movies_df.drop('clean_title', axis=1)
            
            # Ensure consistent movie IDs across dataframes
            movies_set = set(movies_df['movieId'])
            ratings_set = set(ratings_df['movieId'])
            metadata_set = set(metadata_df['movieId'].dropna())
            
            # Log intersection information
            logger.info(f"Number of movies in movies_df: {len(movies_set)}")
            logger.info(f"Number of movies in ratings_df: {len(ratings_set)}")
            logger.info(f"Number of movies in metadata_df: {len(metadata_set)}")
            logger.info(f"Number of movies in intersection: {len(movies_set & ratings_set & metadata_set)}")
            
            # Remove ratings for movies that don't exist in movies_df
            invalid_movie_ids = ratings_set - movies_set
            if invalid_movie_ids:
                logger.warning(f"Found {len(invalid_movie_ids)} ratings for non-existent movies")
                ratings_df = ratings_df[~ratings_df['movieId'].isin(invalid_movie_ids)]
            
            # Remove ratings from users with too few ratings
            user_rating_counts = ratings_df['userId'].value_counts()
            min_ratings = 5  # Minimum number of ratings per user
            valid_users = user_rating_counts[user_rating_counts >= min_ratings].index
            if len(valid_users) < len(user_rating_counts):
                logger.warning(f"Removing {len(user_rating_counts) - len(valid_users)} users with fewer than {min_ratings} ratings")
                ratings_df = ratings_df[ratings_df['userId'].isin(valid_users)]
            
            return movies_df, ratings_df, metadata_df
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise
    
    def train_model(self, epochs=5):
        """Train the hybrid recommender model."""
        try:
            self.hybrid_recommender.train(epochs=epochs)
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
    
    def get_recommendations(self, user_id, top_k=10, exclude_movies=None):
        """Get movie recommendations for a user."""
        try:
            if exclude_movies is None:
                exclude_movies = []
            
            # Get recommendations from hybrid recommender
            recommendations = self.hybrid_recommender.get_recommendations(user_id, top_k=top_k)
            
            # Filter out excluded movies
            recommendations = [rec for rec in recommendations if rec['movieId'] not in exclude_movies]
            
            # Convert scores to percentages
            for rec in recommendations:
                rec['score'] = round(rec['score'] * 100, 2)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get recommendations: {str(e)}")
            return []
    
    def get_similar_movies(self, movie_id, top_k=5):
        """Get similar movies based on genre similarity."""
        try:
            # Get movie genres
            movie_genres = self.metadata_df[self.metadata_df['movieId'] == movie_id]['genres'].iloc[0]
            
            # Calculate genre similarity
            genre_similarity = cosine_similarity(
                self.hybrid_recommender.genre_matrix.loc[movie_id].values.reshape(1, -1),
                self.hybrid_recommender.genre_matrix.values
            ).flatten()
            
            # Get top similar movies
            similar_indices = np.argsort(genre_similarity)[::-1][1:top_k+1]
            similar_movies = []
            
            for idx in similar_indices:
                movie_id = self.metadata_df.iloc[idx]['movieId']
                movie_title = self.metadata_df.iloc[idx]['title']
                similarity = genre_similarity[idx]
                
                similar_movies.append({
                    'movieId': movie_id,
                    'title': movie_title,
                    'similarity': round(similarity * 100, 2)
                })
            
            return similar_movies
            
        except Exception as e:
            logger.error(f"Failed to get similar movies: {str(e)}")
            return []

# Initialize recommender with data loading
try:
    recommender = MovieRecommender()
except Exception as e:
    logger.error(f"Failed to initialize recommender: {str(e)}")
    raise
