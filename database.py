import sqlite3
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import logging
from contextlib import contextmanager
import traceback
import os
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path='movie_recommender.db'):
        self.db_path = db_path
        self.init_db()
        self.load_movies()
        self.users_file = 'users.json'
        self.users = self._load_users()
        # Enhanced caching system
        self._cache = {}
        self._cache_timeout = 3600  # 1 hour cache timeout
        self._last_cache_update = {}
        self._request_count = 0
        self._last_request_reset = datetime.now()
        self._max_requests_per_minute = 80  # Setting safe limit below 100
        
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            yield conn
        finally:
            if conn:
                conn.close()
    
    def init_db(self):
        """Initialize the database with required tables."""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                
                # Movies table
                c.execute('''
                CREATE TABLE IF NOT EXISTS movies (
                    movie_id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    genres TEXT,
                    year INTEGER,
                    director TEXT,
                    actors TEXT,
                    content_rating TEXT,
                    budget REAL,
                    revenue REAL
                )
                ''')
                
                # User watch history table
                c.execute('''
                CREATE TABLE IF NOT EXISTS user_watch_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    movie_id INTEGER NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    watch_duration INTEGER,  -- in seconds
                    completed BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
                )
                ''')
                
                # User ratings table
                c.execute('''
                CREATE TABLE IF NOT EXISTS user_ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    movie_id INTEGER NOT NULL,
                    rating FLOAT NOT NULL,
                    rating_time TIMESTAMP NOT NULL,
                    watch_duration INTEGER,  -- in seconds
                    FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
                )
                ''')
                
                # User behavior table
                c.execute('''
                CREATE TABLE IF NOT EXISTS user_behavior (
                    username TEXT PRIMARY KEY,
                    genre_preferences TEXT,  -- JSON string of genre weights
                    time_patterns TEXT,      -- JSON string of time-based patterns
                    content_preferences TEXT, -- JSON string of content preferences
                    last_updated TIMESTAMP
                )
                ''')
                
                # Pseudo user behavior table
                c.execute('''
                CREATE TABLE IF NOT EXISTS pseudo_user_behavior (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    behavior_pattern TEXT,    -- JSON string of behavior pattern
                    source TEXT,             -- Source of the behavior pattern
                    created_at TIMESTAMP
                )
                ''')
                
                conn.commit()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    def validate_user(self, username: str) -> bool:
        """Validate user credentials."""
        try:
            if not username:
                return False
                
            # Create user if doesn't exist
            if username not in self.users:
                self.users[username] = {
                    'genre_preferences': [],
                    'rated_movies': {},
                    'watch_history': {}
                }
                self._save_users()
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating user: {str(e)}")
            return False
            
    def get_user_genre_preferences(self, username: str) -> List[str]:
        """Get user's genre preferences."""
        try:
            return self.users.get(username, {}).get('genre_preferences', [])
        except Exception as e:
            logger.error(f"Error getting genre preferences: {str(e)}")
            return []
            
    def update_user_genre_preferences(self, username: str, genres: List[str]):
        """Update user's genre preferences."""
        try:
            if username in self.users:
                self.users[username]['genre_preferences'] = genres
                self._save_users()
        except Exception as e:
            logger.error(f"Error updating genre preferences: {str(e)}")
            
    def get_user_rated_movies(self, username: str) -> List[int]:
        """Get list of movies rated by user."""
        try:
            return list(self.users.get(username, {}).get('rated_movies', {}).keys())
        except Exception as e:
            logger.error(f"Error getting rated movies: {str(e)}")
            return []
            
    def add_rating(self, username: str, movie_id: int, rating: float, watch_duration: float = 0):
        """Add or update a movie rating."""
        try:
            if username in self.users:
                self.users[username]['rated_movies'][str(movie_id)] = {
                    'rating': rating,
                    'watch_duration': watch_duration
                }
                self._save_users()
        except Exception as e:
            logger.error(f"Error adding rating: {str(e)}")
            
    def get_movie(self, movie_id: int) -> Optional[Dict]:
        """Get movie details from metadata."""
        try:
            # Load metadata if not already loaded
            if not hasattr(self, 'metadata_df'):
                self.metadata_df = pd.read_csv('movie_metadata_updated.csv')
                
            movie_data = self.metadata_df[self.metadata_df['movieId'] == movie_id]
            if movie_data.empty:
                return None
                
            return {
                'movieId': int(movie_data.iloc[0]['movieId']),
                'title': movie_data.iloc[0]['movie_title'],
                'genres': movie_data.iloc[0]['genres'],
                'year': int(movie_data.iloc[0]['title_year']) if pd.notna(movie_data.iloc[0]['title_year']) else None
            }
            
        except Exception as e:
            logger.error(f"Error getting movie: {str(e)}")
            return None
    
    def get_watched_movies(self, user_id):
        """Get movies watched by a user."""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                
                c.execute('''
                SELECT DISTINCT m.* 
                FROM movies m
                JOIN user_watch_history w ON m.movie_id = w.movie_id
                WHERE w.username = ?
                ''', (user_id,))
                
                movies = []
                for movie in c.fetchall():
                    movies.append({
                        'movieId': movie[0],
                        'title': movie[1],
                        'genres': movie[2],
                        'year': movie[3],
                        'director': movie[4],
                        'actors': movie[5],
                        'content_rating': movie[6],
                        'budget': movie[7],
                        'revenue': movie[8]
                    })
            return movies
            
        except Exception as e:
            logger.error(f"Error getting watched movies: {str(e)}")
            return []
    
    def get_ratings(self, user_id):
        """Get ratings by a user."""
        try:
            # Check cache first
            cache_key = f"ratings_{user_id}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data

            with self.get_connection() as conn:
                c = conn.cursor()
                
                # Optimize query to get only necessary fields
                c.execute('''
                SELECT 
                    r.movie_id,
                    r.rating,
                    r.rating_time,
                    m.title,
                    m.genres
                FROM user_ratings r
                JOIN movies m ON r.movie_id = m.movie_id
                WHERE r.username = ?
                ORDER BY r.rating_time DESC
                ''', (user_id,))
                
                ratings = []
                for row in c.fetchall():
                    ratings.append({
                        'movieId': row[0],
                        'rating': float(row[1]),
                        'rating_time': row[2],
                        'title': row[3],
                        'genres': row[4]
                    })

                # Cache the results
                self._set_cached_data(cache_key, ratings)
                return ratings
            
        except Exception as e:
            logger.error(f"Error getting ratings: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            return []
    
    def get_watch_history(self, user_id):
        """Get watch history for a user."""
        try:
            # Check cache first
            cache_key = f"watch_history_{user_id}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data

            with self.get_connection() as conn:
                c = conn.cursor()
                
                # Optimize query to get only necessary fields
                c.execute('''
                SELECT 
                    w.movie_id,
                    w.start_time,
                    w.end_time,
                    w.watch_duration,
                    w.completed,
                    m.title,
                    m.genres
                FROM user_watch_history w
                JOIN movies m ON w.movie_id = m.movie_id
                WHERE w.username = ?
                ORDER BY w.start_time DESC
                ''', (user_id,))
                
                history = []
                for row in c.fetchall():
                    history.append({
                        'movieId': row[0],
                        'start_time': row[1],
                        'end_time': row[2],
                        'watch_duration': row[3],
                        'completed': bool(row[4]),
                        'title': row[5],
                        'genres': row[6]
                    })

                # Cache the results
                self._set_cached_data(cache_key, history)
                return history
            
        except Exception as e:
            logger.error(f"Error getting watch history: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            return []
    
    def record_watch_session(self, username, movie_id, start_time=None):
        """Record the start of a movie watching session."""
        if start_time is None:
            start_time = datetime.now()
            
        with self.get_connection() as conn:
            c = conn.cursor()
            
            c.execute('''
            INSERT INTO user_watch_history (username, movie_id, start_time)
            VALUES (?, ?, ?)
            ''', (username, movie_id, start_time))
            
            conn.commit()
            return c.lastrowid
    
    def end_watch_session(self, session_id, end_time=None):
        """End a movie watching session and calculate duration."""
        if end_time is None:
            end_time = datetime.now()
            
        with self.get_connection() as conn:
            c = conn.cursor()
            
            # Get start time
            c.execute('SELECT start_time FROM user_watch_history WHERE id = ?', (session_id,))
            start_time = datetime.fromisoformat(c.fetchone()[0])
            
            # Calculate duration
            duration = int((end_time - start_time).total_seconds())
            
            # Update session
            c.execute('''
            UPDATE user_watch_history 
            SET end_time = ?, watch_duration = ?, completed = TRUE
            WHERE id = ?
            ''', (end_time, duration, session_id))
            
            conn.commit()
    
    def record_rating(self, username, movie_id, rating, watch_duration=None):
        """Record a user's rating for a movie."""
        with self.get_connection() as conn:
            c = conn.cursor()
            
            c.execute('''
            INSERT INTO user_ratings (username, movie_id, rating, rating_time, watch_duration)
            VALUES (?, ?, ?, ?, ?)
            ''', (username, movie_id, rating, datetime.now(), watch_duration))
            
            conn.commit()
    
    def update_user_behavior(self, username, metadata_df):
        """Update user behavior based on watch history and ratings."""
        try:
            with self.get_connection() as conn:
                # Get user's watch history and ratings
                watch_history = pd.read_sql_query(
                    "SELECT * FROM user_watch_history WHERE username = ?", 
                    conn, params=(username,)
                )
                ratings = pd.read_sql_query(
                    "SELECT * FROM user_ratings WHERE username = ?", 
                    conn, params=(username,)
                )
                
                # Calculate genre preferences
                genre_preferences = self._calculate_genre_preferences(ratings, metadata_df)
                
                # Calculate time patterns
                time_patterns = self._calculate_time_patterns(watch_history)
                
                # Calculate content preferences
                content_preferences = self._calculate_content_preferences(ratings, metadata_df)
                
                # Store behavior
                behavior_data = {
                    'genre_preferences': genre_preferences,
                    'time_patterns': time_patterns,
                    'content_preferences': content_preferences
                }
                
                c = conn.cursor()
                c.execute('''
                INSERT OR REPLACE INTO user_behavior 
                (username, genre_preferences, time_patterns, content_preferences, last_updated)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    username, 
                    json.dumps(genre_preferences), 
                    json.dumps(time_patterns), 
                    json.dumps(content_preferences),
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                logger.info(f"Updated behavior for user {username}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating user behavior: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            return False
    
    def _calculate_genre_preferences(self, ratings, metadata_df):
        """Calculate genre preferences based on ratings and watch duration."""
        if ratings.empty:
            return {}
            
        # Merge ratings with movie metadata
        rated_movies = pd.merge(
            ratings, 
            metadata_df[['movieId', 'genres']], 
            left_on='movie_id',  # From ratings table
            right_on='movieId',  # From metadata
            how='left'
        )
        
        # Calculate genre weights
        genre_weights = {}
        for _, row in rated_movies.iterrows():
            if pd.notna(row['genres']):
                genres = row['genres'].split('|')
                weight = row['rating'] * (row['watch_duration'] / 3600 if pd.notna(row['watch_duration']) else 1)
                
                for genre in genres:
                    genre_weights[genre] = genre_weights.get(genre, 0) + weight
        
        # Normalize weights
        total = sum(genre_weights.values())
        if total > 0:
            genre_weights = {k: v/total for k, v in genre_weights.items()}
            
        return genre_weights
    
    def _calculate_time_patterns(self, watch_history):
        """Calculate time-based watching patterns."""
        try:
            if watch_history.empty:
                return {
                    'hour_of_day': {},
                    'day_of_week': {},
                    'avg_duration': 0,
                    'completion_rate': 0,
                    'total_movies': 0
                }
            
            # Convert timestamps to datetime
            watch_history['start_time'] = pd.to_datetime(watch_history['start_time'])
            
            # Calculate patterns
            patterns = {
                'hour_of_day': watch_history['start_time'].dt.hour.value_counts().to_dict(),
                'day_of_week': watch_history['start_time'].dt.day_name().value_counts().to_dict(),
                'avg_duration': float(watch_history['watch_duration'].mean()) if 'watch_duration' in watch_history else 0,
                'completion_rate': float(watch_history['completed'].mean() * 100) if 'completed' in watch_history else 0,
                'total_movies': len(watch_history)
            }
            
            return patterns
        except Exception as e:
            logger.error(f"Error calculating time patterns: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            return {
                'hour_of_day': {},
                'day_of_week': {},
                'avg_duration': 0,
                'completion_rate': 0,
                'total_movies': 0
            }
    
    def _calculate_content_preferences(self, ratings, metadata_df):
        """Calculate content preferences based on ratings and metadata."""
        try:
            # Merge ratings with metadata
            rated_movies = pd.merge(
                ratings,
                metadata_df,
                left_on='movie_id',  # From ratings table
                right_on='movieId',  # From metadata
                how='left'
            )
            
            # Calculate preferences for different aspects
            preferences = {
                'genres': self._calculate_weighted_preferences(rated_movies, 'genres'),
                'director': self._calculate_weighted_preferences(rated_movies, 'director_name'),
                'actors': self._calculate_weighted_preferences(rated_movies, 'actor_1_name'),
                'content_rating': self._calculate_weighted_preferences(rated_movies, 'content_rating'),
                'budget_level': self._calculate_budget_preferences(rated_movies),
                'year_range': self._calculate_year_preferences(rated_movies)
            }
            
            return preferences
        except Exception as e:
            logger.error(f"Error calculating content preferences: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            # Return empty preferences if there's an error
            return {
                'genres': {},
                'director': {},
                'actors': {},
                'content_rating': {},
                'budget_level': {},
                'year_range': {}
            }
    
    def _calculate_weighted_preferences(self, df, column):
        """Calculate weighted preferences for a specific column."""
        if column not in df.columns:
            return {}
            
        preferences = {}
        for _, row in df.iterrows():
            if pd.notna(row[column]):
                values = row[column].split('|') if column == 'genres' else [row[column]]
                # Calculate weight based on rating only if watch_duration is not available
                try:
                    weight = row['rating'] * (row['watch_duration'] / 3600 if 'watch_duration' in row and pd.notna(row['watch_duration']) else 1)
                except (KeyError, TypeError):
                    weight = row['rating']  # Fallback to just using the rating
                
                for value in values:
                    preferences[value] = preferences.get(value, 0) + weight
        
        # Normalize weights
        total = sum(preferences.values())
        if total > 0:
            preferences = {k: v/total for k, v in preferences.items()}
            
        return preferences
    
    def _calculate_budget_preferences(self, df):
        """Calculate preferences for different budget levels."""
        if 'budget' not in df.columns:
            return {}
            
        # Define budget levels
        df['budget_level'] = pd.qcut(
            df['budget'].fillna(df['budget'].median()), 
            q=5, 
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        )
        
        return self._calculate_weighted_preferences(df, 'budget_level')
    
    def _calculate_year_preferences(self, df):
        """Calculate preferences for different year ranges."""
        if 'title_year' not in df.columns:
            return {}
            
        # Define year ranges
        df['year_range'] = pd.cut(
            df['title_year'].fillna(df['title_year'].median()),
            bins=[1900, 1950, 1970, 1990, 2010, 2024],
            labels=['1900-1950', '1951-1970', '1971-1990', '1991-2010', '2011-present']
        )
        
        return self._calculate_weighted_preferences(df, 'year_range')
    
    def compare_with_pseudo_users(self, username):
        """Compare user behavior with pseudo user behaviors."""
        with self.get_connection() as conn:
            # Get user behavior
            user_behavior = pd.read_sql_query(
                "SELECT * FROM user_behavior WHERE username = ?", 
                conn, params=(username,)
            )
            
            if user_behavior.empty:
                return []
            
            # Get pseudo user behaviors
            pseudo_behaviors = pd.read_sql_query(
                "SELECT * FROM pseudo_user_behavior", 
                conn
            )
            
            if pseudo_behaviors.empty:
                return []
            
            # Compare behaviors
            similarities = []
            user_data = json.loads(user_behavior.iloc[0]['genre_preferences'])
            
            for _, pseudo in pseudo_behaviors.iterrows():
                pseudo_data = json.loads(pseudo['behavior_pattern'])
                similarity = self._calculate_behavior_similarity(user_data, pseudo_data)
                similarities.append({
                    'pseudo_id': pseudo['id'],
                    'similarity': similarity,
                    'source': pseudo['source']
                })
            
        return sorted(similarities, key=lambda x: x['similarity'], reverse=True)
    
    def _calculate_behavior_similarity(self, user_data, pseudo_data):
        """Calculate similarity between user and pseudo user behaviors."""
        # Combine all features
        all_features = set(list(user_data.keys()) + list(pseudo_data.keys()))
        
        # Create feature vectors
        user_vector = np.array([user_data.get(f, 0) for f in all_features])
        pseudo_vector = np.array([pseudo_data.get(f, 0) for f in all_features])
        
        # Calculate cosine similarity
        return cosine_similarity([user_vector], [pseudo_vector])[0][0]

    def _load_users(self) -> Dict:
        """Load user data from JSON file."""
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading users: {str(e)}")
            return {}
            
    def _save_users(self):
        """Save user data to JSON file."""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f)
        except Exception as e:
            logger.error(f"Error saving users: {str(e)}")

    def load_movies(self):
        """Load movies from CSV files into the database."""
        try:
            # Read movies data
            movies_df = pd.read_csv('movies.csv')
            metadata_df = pd.read_csv('movie_metadata.csv')
            
            with self.get_connection() as conn:
                c = conn.cursor()
                
                # Clear existing movies
                c.execute('DELETE FROM movies')
                
                # Insert movies
                for _, row in movies_df.iterrows():
                    # Get metadata for this movie
                    metadata = metadata_df[metadata_df['movie_title'] == row['title']]
                    
                    c.execute('''
                    INSERT INTO movies (
                        movie_id, title, genres, year, director, actors, 
                        content_rating, budget, revenue
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        int(row['movieId']),
                        row['title'],
                        row['genres'],
                        metadata['title_year'].iloc[0] if not metadata.empty else None,
                        metadata['director_name'].iloc[0] if not metadata.empty else None,
                        metadata['actor_1_name'].iloc[0] if not metadata.empty else None,
                        metadata['content_rating'].iloc[0] if not metadata.empty else None,
                        metadata['budget'].iloc[0] if not metadata.empty else None,
                        metadata['gross'].iloc[0] if not metadata.empty else None
                    ))
                
                conn.commit()
                logger.info(f"Loaded {len(movies_df)} movies into database")
                
        except Exception as e:
            logger.error(f"Failed to load movies: {str(e)}")
            raise

    def get_watch_statistics(self, username):
        """Get comprehensive watch statistics for a user."""
        try:
            with self.get_connection() as conn:
                # Get watch history
                watch_history = pd.read_sql_query(
                    "SELECT * FROM user_watch_history WHERE username = ?", 
                    conn, params=(username,)
                )
                
                # Get ratings
                ratings = pd.read_sql_query(
                    "SELECT * FROM user_ratings WHERE username = ?", 
                    conn, params=(username,)
                )
                
                # Calculate statistics
                stats = {
                    'total_movies_watched': len(watch_history),
                    'average_watch_duration': float(watch_history['watch_duration'].mean()) if not watch_history.empty else 0,
                    'completion_rate': float(watch_history['completed'].mean() * 100) if not watch_history.empty else 0,
                    'average_rating': float(ratings['rating'].mean()) if not ratings.empty else 0,
                    'rating_distribution': ratings['rating'].value_counts().to_dict() if not ratings.empty else {},
                    'time_patterns': {
                        'hour_of_day': watch_history['start_time'].dt.hour.value_counts().to_dict() if not watch_history.empty else {},
                        'day_of_week': watch_history['start_time'].dt.day_name().value_counts().to_dict() if not watch_history.empty else {},
                        'avg_duration': float(watch_history['watch_duration'].mean()) if not watch_history.empty else 0,
                        'completion_rate': float(watch_history['completed'].mean() * 100) if not watch_history.empty else 0,
                        'total_movies': len(watch_history)
                    }
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting watch statistics: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            return {
                'total_movies_watched': 0,
                'average_watch_duration': 0,
                'completion_rate': 0,
                'average_rating': 0,
                'rating_distribution': {},
                'time_patterns': {
                    'hour_of_day': {},
                    'day_of_week': {},
                    'avg_duration': 0,
                    'completion_rate': 0,
                    'total_movies': 0
                }
            }

    def _check_rate_limit(self):
        """Check if we're within rate limits."""
        current_time = datetime.now()
        if (current_time - self._last_request_reset).total_seconds() >= 60:
            self._request_count = 0
            self._last_request_reset = current_time
            
        if self._request_count >= self._max_requests_per_minute:
            raise Exception("Rate limit exceeded. Please try again in a minute.")
            
        self._request_count += 1
        
    def _get_cached_data(self, key):
        """Get data from cache if it exists and is not expired."""
        if key in self._cache:
            if key in self._last_cache_update:
                if (datetime.now() - self._last_cache_update[key]).total_seconds() < self._cache_timeout:
                    return self._cache[key]
        return None
        
    def _set_cached_data(self, key, data):
        """Store data in cache with timestamp."""
        self._cache[key] = data
        self._last_cache_update[key] = datetime.now()
        
    def get_user_data(self, user_id: str) -> Dict:
        """Get all user data in a single query."""
        try:
            # Check cache first
            cache_key = f"user_data_{user_id}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data

            # Get user preferences
            preferences = self.get_user_genre_preferences(user_id)
            
            # Get user ratings
            ratings = self.get_ratings(user_id)
            
            # Get user watch history
            watch_history = self.get_watch_history(user_id)
            
            # Convert to serializable format
            user_data = {
                'genre_preferences': preferences,
                'rated_movies': [int(r['movieId']) for r in ratings],
                'rating_history': [
                    {
                        'movieId': int(r['movieId']),
                        'rating': float(r['rating']),
                        'timestamp': r['rating_time'].isoformat() if isinstance(r['rating_time'], datetime) else r['rating_time']
                    }
                    for r in ratings
                ],
                'watch_history': [
                    {
                        'movieId': int(w['movieId']),
                        'timestamp': w['start_time'].isoformat() if isinstance(w['start_time'], datetime) else w['start_time']
                    }
                    for w in watch_history
                ]
            }
            
            # Cache the results
            self._set_cached_data(cache_key, user_data)
            
            return user_data
            
        except Exception as e:
            logger.error(f"Error getting user data: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            return {}

    def find_similar_users(self, user_id: str, top_k: int = 5) -> List[Dict]:
        """
        Find users with similar preferences and ratings.
        Returns a list of similar users with their similarity scores.
        """
        try:
            # Get target user's data
            user_data = self.get_user_data(user_id)
            if not user_data:
                return []
                
            with self.get_connection() as conn:
                c = conn.cursor()
                
                # Get all users' ratings
                c.execute('''
                SELECT username, movie_id, rating
                FROM user_ratings
                WHERE username != ?
                ''', (user_id,))
                
                # Group ratings by user
                user_ratings = {}
                for username, movie_id, rating in c.fetchall():
                    if username not in user_ratings:
                        user_ratings[username] = {}
                    user_ratings[username][movie_id] = rating
                
                # Calculate similarity scores
                similarities = []
                target_ratings = {r['movieId']: r['rating'] for r in user_data['rating_history']}
                
                for other_user, ratings in user_ratings.items():
                    # Get common movies
                    common_movies = set(target_ratings.keys()) & set(ratings.keys())
                    if len(common_movies) < 3:  # Require at least 3 common movies
                        continue
                    
                    # Calculate genre similarity
                    other_preferences = self.get_user_genre_preferences(other_user)
                    genre_similarity = len(set(user_data['genre_preferences']) & set(other_preferences)) / \
                                    max(len(set(user_data['genre_preferences']) | set(other_preferences)), 1)
                    
                    # Calculate rating similarity
                    rating_diffs = [abs(target_ratings[m] - ratings[m]) for m in common_movies]
                    rating_similarity = 1 - (sum(rating_diffs) / (len(rating_diffs) * 5))  # 5 is max rating
                    
                    # Combined similarity score (weighted average)
                    similarity = (0.4 * genre_similarity) + (0.6 * rating_similarity)
                    
                    similarities.append({
                        'user_id': other_user,
                        'similarity': similarity,
                        'common_movies': len(common_movies),
                        'genre_similarity': genre_similarity,
                        'rating_similarity': rating_similarity
                    })
                
                # Sort by similarity and return top_k
                return sorted(similarities, key=lambda x: x['similarity'], reverse=True)[:top_k]
                
        except Exception as e:
            logger.error(f"Error finding similar users: {str(e)}")
            return []
            
    def get_similar_user_recommendations(self, user_id: str, top_k: int = 10) -> List[Dict]:
        """
        Get movie recommendations based on similar users' preferences.
        """
        try:
            # Check cache first
            cache_key = f"similar_recs_{user_id}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
                
            # Check rate limit
            self._check_rate_limit()
            
            # Find similar users
            similar_users = self.find_similar_users(user_id)
            if not similar_users:
                return []
                
            # Get target user's watched movies
            user_data = self.get_user_data(user_id)
            watched_movies = set(user_data['rated_movies'])
            
            with self.get_connection() as conn:
                c = conn.cursor()
                
                # Get movies rated highly by similar users in one query
                similar_user_ids = [u['user_id'] for u in similar_users]
                placeholders = ','.join(['?' for _ in similar_user_ids])
                
                c.execute(f'''
                WITH similar_user_ratings AS (
                    SELECT 
                        m.movie_id,
                        m.title,
                        m.genres,
                        m.year,
                        AVG(r.rating) as avg_rating,
                        COUNT(DISTINCT r.username) as num_raters
                    FROM movies m
                    JOIN user_ratings r ON m.movie_id = r.movie_id
                    WHERE r.username IN ({placeholders})
                    AND r.rating >= 4.0
                    AND m.movie_id NOT IN (
                        SELECT movie_id 
                        FROM user_ratings 
                        WHERE username = ?
                    )
                    GROUP BY m.movie_id
                    HAVING num_raters >= 2
                )
                SELECT * FROM similar_user_ratings
                ORDER BY avg_rating DESC, num_raters DESC
                LIMIT ?
                ''', similar_user_ids + [user_id, top_k])
                
                recommendations = []
                for row in c.fetchall():
                    movie_id, title, genres, year, avg_rating, num_raters = row
                    recommendations.append({
                        'movieId': movie_id,
                        'title': title,
                        'genres': genres,
                        'year': year,
                        'avg_rating': avg_rating,
                        'num_raters': num_raters,
                        'similarity_score': next(
                            (u['similarity'] for u in similar_users if u['user_id'] in similar_user_ids),
                            0
                        )
                    })
                
                # Cache the results
                self._set_cached_data(cache_key, recommendations)
                
                return recommendations
                
        except Exception as e:
            logger.error(f"Error getting similar user recommendations: {str(e)}")
            return []

# Create a singleton instance
db = Database() 