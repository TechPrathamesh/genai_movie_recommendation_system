from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import json
import os
from hybrid_recommender import HybridRecommender
from database import Database
from behavior_analyzer import BehaviorAnalyzer
import pandas as pd
import logging
from functools import wraps
from datetime import datetime
import traceback
from flask_sqlalchemy import SQLAlchemy
from typing import Dict, List, Optional
from genai_enhancer import genai_enhancer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-here')  # Use environment variable in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movies.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Get the directory containing the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# File paths
USER_DATA_FILE = os.path.join(current_dir, 'users.json')
METADATA_FILE = os.path.join(current_dir, 'movie_metadata_updated.csv')
RATINGS_FILE = os.path.join(current_dir, 'ratings.csv')

# Ensure required directories exist
os.makedirs(os.path.dirname(USER_DATA_FILE), exist_ok=True)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def handle_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            flash(f'An error occurred: {str(e)}', 'error')
            return redirect(url_for('index'))
    return decorated_function

# Initialize database and recommender
try:
    database = Database()
    logger.info("Database initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize database: {str(e)}")
    raise

try:
    # Load data files
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(f"movie_metadata_updated.csv not found at {METADATA_FILE}")
    if not os.path.exists(RATINGS_FILE):
        raise FileNotFoundError(f"ratings.csv not found at {RATINGS_FILE}")
    
    # Load metadata and ratings
    metadata_df = pd.read_csv(METADATA_FILE)
    ratings_df = pd.read_csv(RATINGS_FILE)
    
    # Create user and movie mappings
    unique_users = ratings_df['userId'].unique()
    unique_movies = metadata_df['movieId'].dropna().unique()
    
    user2idx = {str(user): idx for idx, user in enumerate(unique_users)}
    movie2idx = {str(movie): idx for idx, movie in enumerate(unique_movies)}
    
    # Initialize recommender with metadata only
    recommender = HybridRecommender(metadata_df=metadata_df)
    logger.info("Recommender initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize recommender: {str(e)}")
    logger.error(f"Error details: {traceback.format_exc()}")
    recommender = None

# Initialize behavior analyzer
try:
    behavior_analyzer = BehaviorAnalyzer()
    logger.info("Behavior analyzer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize behavior analyzer: {str(e)}")
    behavior_analyzer = None

# Helper to load/save user data
def load_users():
    if not os.path.exists(USER_DATA_FILE):
        return {}
    try:
        with open(USER_DATA_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load users: {str(e)}")
        return {}

def save_users(users):
    try:
        with open(USER_DATA_FILE, 'w') as f:
            json.dump(users, f)
    except Exception as e:
        logger.error(f"Failed to save users: {str(e)}")

# Get available genres from metadata
AVAILABLE_GENRES = sorted(set(genre for genres in metadata_df['genres'].dropna() 
                            for genre in genres.split('|')))

class Movie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    movieId = db.Column(db.Integer, unique=True, nullable=False)
    title = db.Column(db.String(200), nullable=False)
    genres = db.Column(db.String(200))
    year = db.Column(db.Integer)
    rating = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Movie {self.title}>'

class UserBehavior(db.Model):
    __tablename__ = 'user_activity'  # Use a unique table name for activity log
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    movieId = db.Column(db.Integer, nullable=False)
    action = db.Column(db.String(50), nullable=False)  # 'rate' or 'watch'
    rating = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<UserBehavior {self.username} {self.action} for movie {self.movieId}>'

@app.route('/')
@handle_errors
def index():
    if 'user_id' in session:
        return redirect(url_for('recommend'))
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
@handle_errors
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not username or not password:
            flash('Please fill in all fields', 'error')
            return render_template('signup.html')
            
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('signup.html')
            
        users = load_users()
        if username in users:
            flash('Username already exists', 'error')
            return render_template('signup.html')
            
        try:
            # Store user in users.json
            users[username] = {
                'password': password,  # In production, use proper password hashing
                'created_at': datetime.now().isoformat(),
                'genre_preferences': [],
                'rated_movies': {},
                'watch_history': {}
            }
            save_users(users)
            
            # Create user in user_behavior table (not user_activity)
            with database.get_connection() as conn:
                c = conn.cursor()
                c.execute('''
                INSERT INTO user_behavior (username, genre_preferences, time_patterns, content_preferences, last_updated)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    username,
                    json.dumps([]),  # Empty genre preferences
                    json.dumps({}),  # Empty time patterns
                    json.dumps({}),  # Empty content preferences
                    datetime.now().isoformat()
                ))
                conn.commit()
            
            # Set session and redirect
            session['user_id'] = username
            flash('Account created successfully! Please select your favorite genres.', 'success')
            return redirect(url_for('select_genres'))
            
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            flash('Failed to create user account. Please try again.', 'error')
            return render_template('signup.html')
            
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
@handle_errors
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Please fill in all fields', 'error')
            return render_template('login.html')
            
        users = load_users()
        if username not in users or users[username]['password'] != password:
            flash('Invalid username or password', 'error')
            return render_template('login.html')
        
        try:
            # Validate user in database
            if database.validate_user(username):
                session['user_id'] = username
                flash('Successfully logged in!', 'success')
                
                # Check if user has genre preferences
                preferences = database.get_user_genre_preferences(username)
                if not preferences:
                    return redirect(url_for('select_genres'))
                    
                return redirect(url_for('recommend'))
            else:
                flash('Failed to login', 'error')
        except Exception as e:
            flash('An error occurred during login', 'error')
            
    return render_template('login.html')

@app.route('/recommend')
@login_required
@handle_errors
def recommend():
    if recommender is None:
        flash('Recommender system is not available. Please try again later.', 'error')
        return redirect(url_for('index'))
    
    user_id = session['user_id']
    
    try:
        # Get all user data in a single database query
        user_data = database.get_user_data(user_id)
        if not user_data:
            flash('Please select your favorite genres first', 'warning')
            return redirect(url_for('select_genres'))
            
        user_genres = user_data.get('genre_preferences', [])
        if not user_genres:
            flash('Please select your favorite genres first', 'warning')
            return redirect(url_for('select_genres'))
            
        try:
            # Get hybrid recommendations (top 10)
            hybrid_recs = recommender.get_recommendations(
                user_genres=user_genres,
                top_k=10,
                rated_movies=user_data.get('rated_movies', []),
                user_profile=user_data
            )
            
            # Get similar user recommendations
            similar_user_recs = database.get_similar_user_recommendations(user_id, top_k=5)
            
            # Combine and deduplicate recommendations
            all_recs = []
            seen_movies = set()
            
            # Add hybrid recommendations first
            for rec in hybrid_recs:
                if rec['movieId'] not in seen_movies:
                    all_recs.append({
                        'movieId': rec['movieId'],
                        'title': rec['title'],
                        'genres': rec['genres'],
                        'year': rec['year'],
                        'source': 'hybrid',
                        'ai_insight': rec.get('reason', '')
                    })
                    seen_movies.add(rec['movieId'])
            
            # Add similar user recommendations
            for rec in similar_user_recs:
                if rec['movieId'] not in seen_movies:
                    all_recs.append({
                        'movieId': rec['movieId'],
                        'title': rec['title'],
                        'genres': rec['genres'],
                        'year': rec['year'],
                        'source': 'similar_users',
                        'ai_insight': f"Recommended by {rec['num_raters']} similar users with {rec['avg_rating']:.1f} average rating"
                    })
                    seen_movies.add(rec['movieId'])
            
            if not all_recs:
                flash('No recommendations available. Please try rating some movies first.', 'warning')
                return redirect(url_for('index'))
                
            return render_template('recommend.html', 
                                 recommendations=all_recs,
                                 show_similar_users=True)
                                 
        except Exception as e:
            if "Rate limit exceeded" in str(e):
                # If rate limit exceeded, return cached recommendations if available
                cache_key = f"cached_recs_{user_id}"
                cached_recs = database._get_cached_data(cache_key)
                if cached_recs:
                    flash('Showing cached recommendations due to high traffic. Please try again in a minute for fresh recommendations.', 'info')
                    return render_template('recommend.html', 
                                         recommendations=cached_recs,
                                         show_similar_users=True)
                else:
                    flash('System is currently busy. Please try again in a minute.', 'warning')
                    return redirect(url_for('index'))
            else:
                raise e
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        logger.error(f"Error details: {traceback.format_exc()}")
        flash('Failed to get recommendations. Please try again later.', 'error')
        return redirect(url_for('index'))

@app.route('/rate/<int:movie_id>', methods=['GET', 'POST'])
@login_required
@handle_errors
def rate_movie(movie_id):
    if request.method == 'POST':
        try:
            rating = float(request.form.get('rating', 0))
            
            if not (0 <= rating <= 5):
                flash('Rating must be between 0 and 5', 'error')
                return redirect(url_for('rate_movie', movie_id=movie_id))
            
            # Add rating to database
            movie = Movie.query.filter_by(movieId=movie_id).first()
            if movie:
                movie.rating = rating
            else:
                # Get movie details from metadata
                movie_data = metadata_df[metadata_df['movieId'] == movie_id]
                if movie_data.empty:
                    flash('Movie not found in database', 'error')
                    return redirect(url_for('recommend'))
                
                movie = Movie(
                    movieId=movie_id,
                    title=movie_data.iloc[0]['movie_title'],
                    genres=movie_data.iloc[0]['genres'],
                    year=int(movie_data.iloc[0]['title_year']) if pd.notna(movie_data.iloc[0]['title_year']) else None,
                    rating=rating
                )
                db.session.add(movie)
            
            # Record user behavior with username
            behavior = UserBehavior(
                username=session['user_id'],
                movieId=movie_id,
                action='rate',
                rating=rating
            )
            db.session.add(behavior)
            db.session.commit()
            
            flash('Rating submitted successfully!', 'success')
            return redirect(url_for('recommend'))
            
        except ValueError as e:
            flash('Invalid input values', 'error')
            return redirect(url_for('rate_movie', movie_id=movie_id))
    
    # Get movie details from metadata
    movie_data = metadata_df[metadata_df['movieId'] == movie_id]
    if movie_data.empty:
        flash('Movie not found', 'error')
        return redirect(url_for('recommend'))
    
    # Convert to dictionary with proper keys
    movie = {
        'movieId': int(movie_data.iloc[0]['movieId']),
        'title': movie_data.iloc[0]['movie_title'],
        'genres': movie_data.iloc[0]['genres'],
        'year': int(movie_data.iloc[0]['title_year']) if pd.notna(movie_data.iloc[0]['title_year']) else None
    }
    
    return render_template('rate.html', movie=movie)

@app.route('/watch/<int:movie_id>')
@login_required
@handle_errors
def watch_movie(movie_id):
    try:
        # Record watch behavior with username
        behavior = UserBehavior(
            username=session['user_id'],
            movieId=movie_id,
            action='watch'
        )
        db.session.add(behavior)
        db.session.commit()
        flash('Movie marked as watched!', 'success')
    except Exception as e:
        logger.error(f"Error recording watch: {str(e)}")
        flash('Error recording watch. Please try again.', 'error')
        db.session.rollback()
    
    return redirect(url_for('recommend'))

@app.route('/behavior')
@login_required
@handle_errors
def view_behavior():
    try:
        user_id = session['user_id']
        # Only show behaviors for the current user
        behaviors = UserBehavior.query.filter_by(username=user_id).order_by(UserBehavior.timestamp.desc()).all()
        behavior_data = []
        for behavior in behaviors:
            movie = Movie.query.filter_by(movieId=behavior.movieId).first()
            if movie:
                behavior_data.append({
                    'movie_title': movie.title,
                    'action': behavior.action,
                    'rating': behavior.rating,
                    'timestamp': behavior.timestamp
                })
        
        # Get GenAI-enhanced behavior analysis
        analysis = behavior_analyzer.analyze_user_behavior(user_id, database)
        
        return render_template('behavior.html', 
                             behaviors=behavior_data,
                             analysis=analysis)
    except Exception as e:
        logger.error(f"Error viewing behavior: {str(e)}")
        flash('Error loading behavior data. Please try again.', 'error')
        return redirect(url_for('index'))

@app.route('/genres')
@login_required
@handle_errors
def view_genres():
    user_id = session['user_id']
    genre_preferences = database.get_user_genre_preferences(user_id)
    return render_template('genres.html', genre_preferences=genre_preferences)

@app.route('/select-genres', methods=['GET', 'POST'])
@login_required
@handle_errors
def select_genres():
    if request.method == 'POST':
        selected_genres = request.form.getlist('genres')
        if not selected_genres:
            flash('Please select at least one genre', 'error')
            return redirect(url_for('select_genres'))
            
        try:
            # Update user's genre preferences
            user_id = session['user_id']
            database.update_user_genre_preferences(user_id, selected_genres)
            
            # Clear any cached data for this user
            cache_key = f"user_data_{user_id}"
            database._cache.pop(cache_key, None)
            
            flash('Genre preferences updated successfully!', 'success')
            return redirect(url_for('recommend'))
        except Exception as e:
            logger.error(f"Error updating genre preferences: {str(e)}")
            flash('Failed to update genre preferences. Please try again.', 'error')
            return redirect(url_for('select_genres'))
        
    # Get user's current genre preferences
    user_id = session['user_id']
    current_preferences = database.get_user_genre_preferences(user_id)
    
    # If user already has genre preferences, redirect to recommendations
    if current_preferences:
        return redirect(url_for('recommend'))
    
    return render_template('select_genres.html',
                         available_genres=AVAILABLE_GENRES,
                         selected_genres=current_preferences)

@app.route('/logout')
@handle_errors
def logout():
    session.clear()
    flash('Successfully logged out!', 'success')
    return redirect(url_for('login'))

# Create database tables
with app.app_context():
    try:
        db.create_all()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        raise

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    print("\nStarting Flask application...")
    print("Access the website at: http://localhost:5000")
    print("Press Ctrl+C to stop the server\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

    # Print GROQ_API_KEY
    api_key = os.environ.get('GROQ_API_KEY')
    if api_key:
        print("Loaded GROQ_API_KEY:", api_key) 