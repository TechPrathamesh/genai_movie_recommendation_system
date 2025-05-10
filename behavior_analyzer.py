import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
import os
import traceback
import logging
from genai_enhancer import genai_enhancer
from typing import Dict, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BehaviorAnalyzer:
    def __init__(self):
        """Initialize the behavior analyzer."""
        try:
            self.genai = genai_enhancer
            logger.info("Behavior analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing behavior analyzer: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            raise
        
    def analyze_user_behavior(self, username: str, db) -> Dict:
        """Analyze user behavior and return insights."""
        try:
            # Get watch statistics
            stats = db.get_watch_statistics(username)
            
            # Get genre preferences
            genre_preferences = db.get_user_genre_preferences(username)
            
            # Prepare data for GenAI analysis
            user_data = {
                'total_movies_watched': stats.get('total_movies_watched', 0),
                'average_watch_duration': stats.get('average_watch_duration', 0),
                'completion_rate': stats.get('completion_rate', 0),
                'average_rating': stats.get('average_rating', 0),
                'rating_distribution': stats.get('rating_distribution', {}),
                'time_patterns': stats.get('time_patterns', {}),
                'genre_preferences': genre_preferences or []
            }
            
            # Get GenAI analysis
            genai_analysis = self.genai.analyze_user_behavior(user_data)
            
            # Combine traditional and GenAI insights
            insights = []
            
            # Add GenAI insights
            if genai_analysis.get('watching_patterns'):
                insights.append(genai_analysis['watching_patterns'])
            if genai_analysis.get('preference_insights'):
                insights.append(genai_analysis['preference_insights'])
            if genai_analysis.get('genre_suggestions'):
                insights.append(genai_analysis['genre_suggestions'])
            if genai_analysis.get('improvement_recommendations'):
                insights.append(genai_analysis['improvement_recommendations'])
            
            # Add traditional insights as fallback
            if not insights:
                # Watch completion insight
                if stats.get('completion_rate', 0) < 50:
                    insights.append("You often don't complete movies, suggesting you're selective about what you watch.")
                elif stats.get('completion_rate', 0) > 80:
                    insights.append("You usually complete the movies you start, showing strong commitment to your choices.")
                
                # Genre preferences insight
                if genre_preferences:
                    insights.append(f"You enjoy {', '.join(genre_preferences)} movies, showing diverse genre preferences.")
                
                # Rating pattern insight
                if stats.get('average_rating', 0) > 4:
                    insights.append("You tend to rate movies highly, indicating you're good at selecting movies you'll enjoy.")
                elif stats.get('average_rating', 0) < 3:
                    insights.append("You're quite critical in your ratings, suggesting you have high standards for movies.")
                
                # Time pattern insight
                if stats.get('time_patterns', {}).get('hour_of_day'):
                    most_active_hour = max(stats['time_patterns']['hour_of_day'].items(), key=lambda x: x[1])[0]
                    insights.append(f"You're most active during {most_active_hour}:00 hours.")
            
            return {
                'statistics': stats,
                'genre_preferences': genre_preferences,
                'insights': insights,
                'genai_analysis': genai_analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing user behavior: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            return {
                'statistics': {
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
                },
                'genre_preferences': [],
                'insights': [],
                'genai_analysis': self.genai._get_default_analysis()
            }
    
    def _analyze_time_patterns(self, watch_history_df):
        """Analyze time-based watching patterns."""
        if watch_history_df.empty:
            return {
                'hour_distribution': {},
                'day_distribution': {},
                'avg_duration_by_hour': {}
            }
            
        try:
            watch_history_df['start_time'] = pd.to_datetime(watch_history_df['start_time'])
            
            return {
                'hour_distribution': watch_history_df['start_time'].dt.hour.value_counts().to_dict(),
                'day_distribution': watch_history_df['start_time'].dt.day_name().value_counts().to_dict(),
                'avg_duration_by_hour': watch_history_df.groupby(watch_history_df['start_time'].dt.hour)['watch_duration'].mean().to_dict()
            }
        except Exception as e:
            print(f"Error analyzing time patterns: {str(e)}")
            return {
                'hour_distribution': {},
                'day_distribution': {},
                'avg_duration_by_hour': {}
            }
    
    def _analyze_rating_patterns(self, ratings_df):
        """Analyze rating patterns."""
        if ratings_df.empty:
            return {
                'avg_rating': 0,
                'rating_distribution': {},
                'rating_trend': []
            }
            
        try:
            ratings_df['rating_time'] = pd.to_datetime(ratings_df['rating_time'])
            ratings_df = ratings_df.sort_values('rating_time')
            
            return {
                'avg_rating': float(ratings_df['rating'].mean()),
                'rating_distribution': ratings_df['rating'].value_counts().to_dict(),
                'rating_trend': ratings_df['rating'].rolling(window=5).mean().fillna(0).tolist()
            }
        except Exception as e:
            print(f"Error analyzing rating patterns: {str(e)}")
            return {
                'avg_rating': 0,
                'rating_distribution': {},
                'rating_trend': []
            }
    
    def _generate_insights(self, total_movies, avg_watch_duration, completion_rate, 
                         time_patterns, rating_patterns, genre_preferences):
        """Generate insights about user behavior."""
        insights = []
        
        # Watch duration insights
        if avg_watch_duration > 0:
            if avg_watch_duration > 120:
                insights.append("You tend to watch longer movies, suggesting you enjoy immersive storytelling.")
            elif avg_watch_duration < 90:
                insights.append("You prefer shorter movies, indicating you enjoy concise storytelling.")
        
        # Completion rate insights
        if completion_rate > 0.8:
            insights.append("You have a high completion rate, showing strong engagement with the content.")
        elif completion_rate < 0.5:
            insights.append("You often don't complete movies, suggesting you're selective about what you watch.")
        
        # Time pattern insights
        if time_patterns['hour_distribution']:
            peak_hour = max(time_patterns['hour_distribution'].items(), key=lambda x: x[1])[0]
            if 18 <= peak_hour <= 22:
                insights.append("You're a prime-time viewer, typically watching in the evening.")
            elif 22 <= peak_hour <= 4:
                insights.append("You're a night owl, preferring late-night viewing sessions.")
        
        # Rating pattern insights
        if rating_patterns['avg_rating'] > 4:
            insights.append("You tend to rate movies highly, showing an appreciation for the content.")
        elif rating_patterns['avg_rating'] < 3:
            insights.append("You're a critical viewer, with lower average ratings.")
        
        # Genre preference insights
        if genre_preferences and len(genre_preferences) > 0:
            insights.append(f"You enjoy {', '.join(genre_preferences)} movies, showing diverse genre preferences.")
        
        return insights

# Create a singleton instance
behavior_analyzer = BehaviorAnalyzer() 