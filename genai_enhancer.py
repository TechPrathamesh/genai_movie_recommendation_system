from groq import Groq
import json
from typing import List, Dict, Optional
import logging
from datetime import datetime
import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ai_insight_cache = {}

class GenAIEnhancer:
    def __init__(self):
        """Initialize the GenAI enhancer with Gemma 2 9B model using config.json or fallback to hardcoded key."""
        api_key = None
        # Try to load from config.json
        try:
            with open("config.json") as f:
                config = json.load(f)
                api_key = config.get("GROQ_API_KEY")
                logger.info("Loaded GROQ_API_KEY from config.json")
        except Exception as e:
            logger.warning(f"Could not load config.json: {e}")
        # Fallback to hardcoded key
        if not api_key:
            api_key = "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"
            logger.info("Using hardcoded GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        if not self._validate_api_key(api_key):
            raise ValueError("Invalid GROQ_API_KEY format")
        self.client = Groq(api_key=api_key)
        self.model = "gemma2-9b-it"
        self.max_retries = 3
        self.timeout = 30
        self._cache = {}
        self._cache_timeout = 3600  # 1 hour cache timeout
        self._last_cache_update = {}
        self._request_count = 0
        self._last_request_reset = datetime.now()
        self._max_requests_per_minute = 25  # Safe limit below 30 RPM
        logger.info("Groq API client initialized successfully with Gemma 2 9B model")
    
    def _validate_api_key(self, api_key: str) -> bool:
        """Validate the format of the API key."""
        # Basic validation - adjust based on actual Groq API key format
        return bool(api_key and len(api_key) > 20 and api_key.startswith('gsk_'))
    
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
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _make_api_call(self, messages: List[Dict], max_tokens: int = 500) -> str:
        """Make an API call with retry logic."""
        try:
            # Check rate limit
            self._check_rate_limit()
            
            # Check cache for exact message match
            cache_key = f"api_call_{hash(str(messages))}"
            cached_response = self._get_cached_data(cache_key)
            if cached_response:
                return cached_response
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=max_tokens,
                timeout=self.timeout
            )
            
            result = response.choices[0].message.content
            self._set_cached_data(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise
        
    def analyze_user_behavior(self, user_data: Dict) -> Dict:
        """Analyze user behavior using Gemma 2 9B model."""
        try:
            # Validate input data
            if not isinstance(user_data, dict):
                raise ValueError("user_data must be a dictionary")
            
            # Prepare the prompt
            prompt = self._create_behavior_analysis_prompt(user_data)
            
            # Generate response using Groq API with retry logic
            response_text = self._make_api_call([
                {"role": "system", "content": "You are an AI assistant that analyzes user behavior patterns and provides insights. You are powered by Gemma 2 9B."},
                {"role": "user", "content": prompt}
            ])
            
            # Parse and structure the response
            analysis = self._parse_behavior_analysis(response_text)
            return analysis
            
        except Exception as e:
            logger.error(f"Error in behavior analysis: {str(e)}")
            return self._get_default_analysis()
    
    def generate_movie_insights(self, movie_data: Dict) -> str:
        """Generate personalized movie insights using Gemma 2 9B model."""
        try:
            # Validate input data
            if not isinstance(movie_data, dict):
                raise ValueError("movie_data must be a dictionary")
            
            prompt = self._create_movie_insights_prompt(movie_data)
            
            response_text = self._make_api_call([
                {"role": "system", "content": "You are an AI assistant that provides personalized movie insights and recommendations. You are powered by Gemma 2 9B."},
                {"role": "user", "content": prompt}
            ], max_tokens=300)
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating movie insights: {str(e)}")
            return "Unable to generate movie insights at this time."
    
    def _create_behavior_analysis_prompt(self, user_data: Dict) -> str:
        """Create a prompt for behavior analysis."""
        return f"""Analyze the following user behavior data and provide insights:
        User Data: {json.dumps(user_data, indent=2)}
        
        Please provide a JSON response with the following structure:
        {{
            "watching_patterns": "Analysis of user's watching habits",
            "preference_insights": "Insights about user preferences",
            "genre_suggestions": "Suggested genres based on behavior",
            "improvement_recommendations": "Recommendations for better experience"
        }}
        
        Keep the analysis concise and focused on actionable insights."""
    
    def _create_movie_insights_prompt(self, movie_data: Dict) -> str:
        """Create a prompt for movie insights."""
        return f"""Based on the following movie data, provide personalized insights:
        Movie Data: {json.dumps(movie_data, indent=2)}
        
        Please provide:
        1. Key highlights of the movie
        2. Why this movie might appeal to the user
        3. Similar movies they might enjoy
        4. Any potential concerns or content warnings
        
        Keep the insights concise and engaging."""
    
    def _parse_behavior_analysis(self, response_text: str) -> Dict:
        """Parse the behavior analysis response into a structured format."""
        try:
            # Clean the response text
            response_text = response_text.strip()
            
            # Try to extract JSON if it's wrapped in markdown code blocks
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            # Try to parse as JSON first
            try:
                analysis = json.loads(response_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract key-value pairs
                analysis = {}
                lines = response_text.split('\n')
                current_key = None
                current_value = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    if ':' in line and not current_key:
                        parts = line.split(':', 1)
                        current_key = parts[0].strip().strip('"\'')
                        current_value = [parts[1].strip().strip('"\'')]
                    elif current_key:
                        current_value.append(line.strip().strip('"\''))
                    else:
                        continue
                
                if current_key:
                    analysis[current_key] = ' '.join(current_value)
            
            # Ensure required keys exist
            required_keys = {
                'watching_patterns',
                'preference_insights',
                'genre_suggestions',
                'improvement_recommendations'
            }
            
            # Add missing keys with default values
            for key in required_keys:
                if key not in analysis:
                    analysis[key] = f"No {key.replace('_', ' ')} available"
            
            # Add timestamp
            analysis['timestamp'] = datetime.now().isoformat()
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing behavior analysis: {str(e)}")
            logger.error(f"Raw response: {response_text}")
            return self._get_default_analysis()
    
    def _get_default_analysis(self) -> Dict:
        """Return default analysis structure when errors occur."""
        return {
            'watching_patterns': "Unable to analyze watching patterns at this time.",
            'preference_insights': "Unable to generate preference insights at this time.",
            'genre_suggestions': "Unable to suggest new genres at this time.",
            'improvement_recommendations': "Unable to provide recommendations at this time.",
            'timestamp': datetime.now().isoformat()
        }

    def enhance_recommendations(self, recommendations: list, user_profile: dict) -> list:
        """Enhance recommendations with AI-generated insights and cache them."""
        try:
            enhanced = []
            for i, rec in enumerate(recommendations):
                # Create a cache key based on movie and user profile
                cache_key = f"insight_{rec.get('movieId')}_{hash(str(user_profile.get('genre_preferences', [])))}"
                
                if i < 3:  # Only enhance top 3 recommendations
                    cached_insight = self._get_cached_data(cache_key)
                    if cached_insight:
                        rec['ai_insight'] = cached_insight
                    else:
                        try:
                            movie_data = {
                                'title': rec.get('title'),
                                'genres': rec.get('genres'),
                                'year': rec.get('year'),
                                'user_genres': user_profile.get('genre_preferences', []),
                                'rating_history': user_profile.get('rating_history', []),
                                'watch_history': user_profile.get('watch_history', [])
                            }
                            insight = self.generate_movie_insights(movie_data)
                            rec['ai_insight'] = insight
                            self._set_cached_data(cache_key, insight)
                        except Exception as e:
                            logger.error(f"Error generating insight for movie {rec.get('movieId')}: {str(e)}")
                            rec['ai_insight'] = "AI insight temporarily unavailable."
                else:
                    rec['ai_insight'] = "AI insight available for top picks only."
                enhanced.append(rec)
            return enhanced
        except Exception as e:
            logger.error(f"Error enhancing recommendations: {str(e)}")
            return recommendations  # Return original recommendations if enhancement fails

    def get_genai_recommendations(self, user_profile, hybrid_recs, other_user_behaviors, available_movies, top_k=10):
        """
        Use GenAI to generate collaborative recommendations, blending with hybrid recommender picks.
        Returns a list of dicts: [{movieId, reason}]
        """
        prompt = (
            "You are a movie recommendation assistant.\n"
            "User profile: " + json.dumps(user_profile) + "\n"
            "Other user behaviors: " + json.dumps(other_user_behaviors) + "\n"
            "Hybrid recommender top picks: " + json.dumps([m['movieId'] for m in hybrid_recs]) + "\n"
            "Available movies: " + json.dumps(available_movies) + "\n"
            f"Task: Based on the current user's profile, find the most similar users from the list above. "
            f"Suggest {top_k//2} movies that those similar users liked, but the current user hasn't seen or rated. "
            f"Blend these with {top_k//2} from the hybrid recommender's top picks for a total of {top_k} recommendations. "
            "Return ONLY a valid JSON list of objects: [{\"movieId\": ..., \"reason\": ...}]. Do not include any extra text or explanation. If you cannot answer, return []."
        )
        response_text = self._make_api_call([
            {"role": "system", "content": "You are a movie recommendation assistant."},
            {"role": "user", "content": prompt}
        ], max_tokens=1000)
        logger.error(f"GenAI raw response: {response_text}")
        try:
            recs = json.loads(response_text)
            return recs
        except Exception as e:
            logger.error(f"Failed to parse GenAI recommendations: {e}")
            return []

# Create a singleton instance
genai_enhancer = GenAIEnhancer() 