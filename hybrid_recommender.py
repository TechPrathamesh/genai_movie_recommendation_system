import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Dict, List, Optional, Union
import traceback
import os
import re
from genai_enhancer import genai_enhancer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRecommenderError(Exception):
    """Base exception class for HybridRecommender errors."""
    pass

class DataValidationError(HybridRecommenderError):
    """Raised when data validation fails."""
    pass

class HybridRecommender:
    def __init__(self, metadata_df: pd.DataFrame):
        """Initialize the hybrid recommender with metadata."""
        if metadata_df is None or metadata_df.empty:
            raise HybridRecommenderError("Metadata DataFrame is empty or None")
            
        # Ensure required columns exist
        required_columns = ['movieId', 'movie_title', 'genres', 'title_year']
        missing_columns = [col for col in required_columns if col not in metadata_df.columns]
        if missing_columns:
            raise HybridRecommenderError(f"Missing required columns: {missing_columns}")
            
        # Optimize DataFrame memory usage
        self.metadata_df = self._optimize_dataframe(metadata_df.copy())
        self.genai = genai_enhancer
        
        try:
            # Calculate popularity scores
            self._calculate_popularity_scores()
            
            # Pre-compute genre vectors for faster matching
            self._precompute_genre_vectors()
            
            logger.info("Recommender initialized successfully")
        except Exception as e:
            raise HybridRecommenderError(f"Failed to initialize recommender: {str(e)}")
    
    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        for col in df.columns:
            if df[col].dtype == 'object':
                # Convert string columns to categorical if they have low cardinality
                if df[col].nunique() < len(df) * 0.5:  # If less than 50% unique values
                    df[col] = df[col].astype('category')
            elif df[col].dtype == 'float64':
                # Convert float64 to float32 if possible
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                # Convert int64 to int32 if possible
                df[col] = df[col].astype('int32')
        return df
    
    def _precompute_genre_vectors(self):
        """Pre-compute genre vectors for faster matching."""
        # Create a set of all unique genres
        all_genres = set()
        for genres in self.metadata_df['genres'].dropna():
            all_genres.update(genres.split('|'))
        self.all_genres = sorted(list(all_genres))
        
        # Create genre vectors for each movie
        genre_vectors = []
        for genres in self.metadata_df['genres']:
            if pd.isna(genres):
                genre_vectors.append(np.zeros(len(self.all_genres)))
            else:
                vector = np.zeros(len(self.all_genres))
                for genre in genres.split('|'):
                    if genre in self.all_genres:
                        vector[self.all_genres.index(genre)] = 1
                genre_vectors.append(vector)
        
        self.genre_vectors = np.array(genre_vectors)
        
    def _calculate_popularity_scores(self):
        """Calculate popularity scores for movies."""
        try:
            # Calculate popularity based on IMDB score and number of votes
            if 'imdb_score' in self.metadata_df.columns and 'num_voted_users' in self.metadata_df.columns:
                self.metadata_df['popularity_score'] = (
                    self.metadata_df['imdb_score'] * 
                    np.log1p(self.metadata_df['num_voted_users'])
                )
            else:
                # Fallback to simple popularity based on movie ID (assuming newer movies are more popular)
                self.metadata_df['popularity_score'] = self.metadata_df['movieId'].rank(ascending=False)
            
            # Normalize popularity scores
            self.metadata_df['popularity_score'] = (
                self.metadata_df['popularity_score'] - self.metadata_df['popularity_score'].min()
            ) / (self.metadata_df['popularity_score'].max() - self.metadata_df['popularity_score'].min())
            
            logger.info("Calculated popularity scores successfully")
            
        except Exception as e:
            raise HybridRecommenderError(f"Failed to calculate popularity scores: {str(e)}")
    
    def get_recommendations(self, user_genres=None, top_k=10, rated_movies=None, user_profile=None):
        """Get movie recommendations using a hybrid approach."""
        try:
            # Get content-based recommendations
            content_recs = self._get_content_recommendations(user_genres, top_k=top_k*2)
            
            # Get popularity-based recommendations
            popularity_recs = self._get_popularity_recommendations(top_k=top_k*2)
            
            # Combine and diversify recommendations
            recommendations = self._diversify_recommendations(
                content_recs=content_recs,
                popularity_recs=popularity_recs,
                rated_movies=rated_movies or [],
                top_k=top_k
            )
            
            # Enhance recommendations with GenAI insights if user profile is provided
            if user_profile:
                recommendations = self.genai.enhance_recommendations(recommendations, user_profile)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            return []

    def _get_content_recommendations(self, user_genres, top_k=20):
        """Get content-based recommendations based on user genres."""
        try:
            if self.metadata_df is None or self.metadata_df.empty:
                return []
                
            # Start with all movies
            recommendations = self.metadata_df.copy()
            
            # Filter by user genres if provided
            if user_genres:
                # Convert user genres to vector
                user_vector = np.zeros(len(self.all_genres))
                for genre in user_genres:
                    if genre in self.all_genres:
                        user_vector[self.all_genres.index(genre)] = 1
                
                # Calculate genre similarity using pre-computed vectors
                similarities = cosine_similarity([user_vector], self.genre_vectors)[0]
                recommendations['genre_match_score'] = similarities
                
                # Filter out movies with no genre match
                recommendations = recommendations[recommendations['genre_match_score'] > 0]
            
            if recommendations.empty:
                return []
            
            # Calculate content similarity score
            recommendations['content_score'] = recommendations['popularity_score'] * 0.3 + recommendations['genre_match_score'] * 0.7
            
            # Sort by content score
            recommendations = recommendations.sort_values('content_score', ascending=False)
            
            # Get top K recommendations
            top_recommendations = recommendations.head(top_k)
            
            # Convert to list of dictionaries
            result = []
            for _, row in top_recommendations.iterrows():
                if pd.notna(row['movieId']):
                    result.append({
                        'movieId': int(row['movieId']),
                        'title': str(row['movie_title']).strip() if pd.notna(row['movie_title']) else 'Unknown Title',
                        'genres': str(row['genres']).strip() if pd.notna(row['genres']) else 'Unknown',
                        'year': int(row['title_year']) if pd.notna(row['title_year']) else None,
                        'final_score': float(row['content_score'])
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            return []

    def _get_popularity_recommendations(self, top_k=20):
        """Get popularity-based recommendations."""
        try:
            if self.metadata_df is None or self.metadata_df.empty:
                return []
                
            # Sort by popularity score
            recommendations = self.metadata_df.sort_values('popularity_score', ascending=False)
            
            # Get top K recommendations
            top_recommendations = recommendations.head(top_k)
            
            # Convert to list of dictionaries
            result = []
            for _, row in top_recommendations.iterrows():
                if pd.notna(row['movieId']):
                    result.append({
                        'movieId': int(row['movieId']),
                        'title': str(row['movie_title']).strip() if pd.notna(row['movie_title']) else 'Unknown Title',
                        'genres': str(row['genres']).strip() if pd.notna(row['genres']) else 'Unknown',
                        'year': int(row['title_year']) if pd.notna(row['title_year']) else None,
                        'final_score': float(row['popularity_score'])
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in popularity-based recommendations: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            return []

    def _diversify_recommendations(self, content_recs, popularity_recs, rated_movies, top_k):
        """Diversify recommendations by combining different sources and adding randomization."""
        try:
            # Remove already rated movies
            content_recs = [r for r in content_recs if r['movieId'] not in rated_movies]
            popularity_recs = [r for r in popularity_recs if r['movieId'] not in rated_movies]
            
            # Add some randomization to scores
            for rec in content_recs:
                rec['final_score'] = float(rec.get('final_score', 0)) * (0.8 + 0.4 * np.random.random())
                
            for rec in popularity_recs:
                rec['final_score'] = float(rec.get('final_score', 0)) * (0.8 + 0.4 * np.random.random())
            
            # Combine recommendations with weights
            combined_recs = []
            
            # Add content-based recommendations (60% weight)
            for rec in content_recs:
                combined_recs.append({
                    'movieId': rec['movieId'],
                    'title': rec['title'],
                    'genres': rec['genres'],
                    'year': rec['year'],
                    'final_score': float(rec['final_score']) * 0.6
                })
            
            # Add popularity-based recommendations (40% weight)
            for rec in popularity_recs:
                combined_recs.append({
                    'movieId': rec['movieId'],
                    'title': rec['title'],
                    'genres': rec['genres'],
                    'year': rec['year'],
                    'final_score': float(rec['final_score']) * 0.4
                })
            
            # Sort by final score
            combined_recs.sort(key=lambda x: x['final_score'], reverse=True)
            
            # Ensure genre diversity
            diverse_recs = []
            used_genres = set()
            
            for rec in combined_recs:
                movie_genres = set(rec['genres'].split('|'))
                
                # If this movie adds new genres or we haven't reached top_k yet
                if movie_genres - used_genres or len(diverse_recs) < top_k:
                    diverse_recs.append(rec)
                    used_genres.update(movie_genres)
                
                if len(diverse_recs) >= top_k:
                    break
            
            # If we don't have enough diverse recommendations, add remaining top scored movies
            if len(diverse_recs) < top_k:
                remaining = [r for r in combined_recs if r not in diverse_recs]
                diverse_recs.extend(remaining[:top_k - len(diverse_recs)])
            
            # Shuffle the final recommendations slightly
            np.random.shuffle(diverse_recs)
            
            # Ensure all recommendations have a final_score
            for rec in diverse_recs:
                if 'final_score' not in rec:
                    rec['final_score'] = 0.0
            
            return diverse_recs[:top_k]
            
        except Exception as e:
            logger.error(f"Error diversifying recommendations: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            return [] 