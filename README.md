# Movie Recommendation System

A Flask-based movie recommendation system that uses hybrid filtering to provide personalized movie recommendations.

## Features

- User authentication and session management
- Personalized movie recommendations
- Movie rating and watch history tracking
- User behavior analysis
- Genre-based preferences
- Responsive web interface

## Requirements

- Python 3.8 or higher
- Flask
- PyTorch
- Pandas
- NumPy
- scikit-learn
- SciPy

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd movie-recommender
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare the data files:
   - Place `movies.csv` in the root directory
   - Place `ratings.csv` in the root directory
   - Place `movie_metadata.csv` in the root directory

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Access the application at: http://localhost:5000

3. To stop the server, press Ctrl+C

## Project Structure

- `app.py`: Main Flask application
- `recommender.py`: Movie recommendation engine
- `hybrid_recommender.py`: Hybrid filtering implementation
- `database.py`: Database operations
- `behavior_analyzer.py`: User behavior analysis
- `templates/`: HTML templates
- `static/`: CSS and JavaScript files

## Usage

1. Log in with your user ID
2. Browse movie recommendations
3. Rate movies and track watch history
4. View personalized insights and genre preferences

## Data Files

The system requires the following data files:

- `movies.csv`: Contains movie information (ID, title, genres)
- `ratings.csv`: Contains user ratings
- `movie_metadata.csv`: Contains additional movie metadata

## Error Handling

The application includes comprehensive error handling:
- Input validation
- Database connection management
- Data type validation
- User authentication
- API error handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request 