# AI-Powered Movie Recommendation System

A full-stack, AI-powered movie recommender web application built with Flask, SQLAlchemy, and modern hybrid recommendation techniques. The system combines collaborative filtering, content-based filtering, and GenAI to provide personalized movie recommendations and insights.

## 🌟 Features

- **Hybrid Recommendation Engine**
  - Content-based filtering using movie metadata
  - Collaborative filtering using user ratings
  - Similar user analysis for better personalization

- **GenAI Integration**
  - Behavior analysis and insights
  - Personalized movie descriptions
  - User preference analysis

- **User Features**
  - Secure authentication system
  - Genre preference selection
  - Movie rating and watch history
  - Personalized recommendations
  - Activity tracking and insights

## 📊 Architecture

### System Overview
```mermaid
graph TD
    A[User Interface] --> B[Flask Application]
    B --> C[Hybrid Recommender]
    B --> D[GenAI Enhancer]
    B --> E[Database]
    C --> F[Content Filtering]
    C --> G[Collaborative Filtering]
    D --> H[Behavior Analysis]
    D --> I[Insight Generation]
```

### Recommendation Flow
```mermaid
graph LR
    A[User Input] --> B[Genre Selection]
    B --> C[Initial Recommendations]
    C --> D[User Ratings]
    D --> E[Hybrid Filtering]
    E --> F[Final Recommendations]
    F --> G[GenAI Enhancement]
```

### GenAI Integration Flow
```mermaid
graph TD
    A[User Activity] --> B[Data Collection]
    B --> C[Behavior Analysis]
    C --> D[Pattern Recognition]
    D --> E[Insight Generation]
    
    F[Movie Data] --> G[Content Analysis]
    G --> H[Feature Extraction]
    H --> I[Semantic Understanding]
    
    E --> J[Personalized Insights]
    I --> J
    J --> K[User Interface]
    
    subgraph "Groq API Integration"
        L[API Request] --> M[LLM Processing]
        M --> N[Response Generation]
    end
    
    C --> L
    N --> E
```

### Project flow
```mermaid
flowchart TD
    Start([Start]) --> Init[Initialize HybridRecommender]
    
    subgraph Initialization
        Init --> ValidateData{Validate Metadata}
        ValidateData -->|Invalid| Error[Raise Error]
        ValidateData -->|Valid| CopyData[Copy Metadata DataFrame]
        CopyData --> CalcPopScores[Calculate Popularity Scores]
    end
    
    subgraph GetRecommendations
        GetRecs[Get Recommendations] --> InputParams{Check Input Parameters}
        InputParams -->|Invalid| ReturnEmpty[Return Empty List]
        InputParams -->|Valid| GetContent[Get Content-Based Recommendations]
        InputParams -->|Valid| GetPopular[Get Popularity-Based Recommendations]
    end
    
    subgraph ContentBasedFiltering
        GetContent --> CheckMetadata{Check Metadata}
        CheckMetadata -->|Empty| ReturnEmpty
        CheckMetadata -->|Valid| FilterGenres[Filter by User Genres]
        FilterGenres --> CalcGenreMatch[Calculate Genre Match Score]
        CalcGenreMatch --> CalcContentScore[Calculate Content Score]
        CalcContentScore --> SortContent[Sort by Content Score]
        SortContent --> GetTopContent[Get Top K Recommendations]
    end
    
    subgraph PopularityBasedFiltering
        GetPopular --> CheckMetadata2{Check Metadata}
        CheckMetadata2 -->|Empty| ReturnEmpty
        CheckMetadata2 -->|Valid| SortPop[Sort by Popularity Score]
        SortPop --> GetTopPop[Get Top K Recommendations]
    end
    
    subgraph Diversification
        GetTopContent --> Diversify[Diversify Recommendations]
        GetTopPop --> Diversify
        Diversify --> RemoveRated[Remove Rated Movies]
        RemoveRated --> Randomize[Add Randomization to Scores]
        Randomize --> Combine[Combine Recommendations]
        
        Combine -->|60% Weight| ContentWeight[Content-Based Weight]
        Combine -->|40% Weight| PopWeight[Popularity Weight]
        
        ContentWeight --> SortFinal[Sort by Final Score]
        PopWeight --> SortFinal
        
        SortFinal --> EnsureDiversity[Ensure Genre Diversity]
        EnsureDiversity --> Shuffle[Shuffle Final Recommendations]
    end
    
    subgraph GenAIEnhancement
        Shuffle --> CheckProfile{Check User Profile}
        CheckProfile -->|Available| Enhance[Enhance with GenAI]
        CheckProfile -->|Not Available| SkipEnhance[Skip Enhancement]
        Enhance --> FinalRecs[Final Recommendations]
        SkipEnhance --> FinalRecs
    end
    
    subgraph PopularityScoreCalculation
        CalcPopScores --> CheckIMDB{Check IMDB Data}
        CheckIMDB -->|Available| CalcIMDB[Calculate IMDB-based Score]
        CheckIMDB -->|Not Available| CalcSimple[Calculate Simple Score]
        CalcIMDB --> Normalize[Normalize Scores]
        CalcSimple --> Normalize
    end
    
    FinalRecs --> End([End])
    ReturnEmpty --> End
    
    classDef process fill:#f9f,stroke:#333,stroke-width:2px
    classDef decision fill:#bbf,stroke:#333,stroke-width:2px
    classDef error fill:#fbb,stroke:#333,stroke-width:2px
    classDef start_end fill:#bfb,stroke:#333,stroke-width:2px
    
    class Start,End start_end
    class ValidateData,InputParams,CheckMetadata,CheckMetadata2,CheckIMDB,CheckProfile decision
    class Error,ReturnEmpty error
    class Init,GetRecs,GetContent,GetPopular,Diversify,Enhance process
```

## 🔄 Technical Stack

- **Backend**: Python, Flask
- **Database**: SQLite with SQLAlchemy
- **AI/ML**: 
  - Hybrid recommender system
  - Groq API for GenAI features
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: Pandas, NumPy

## 📥 Installation

1. Clone the repository:
```bash
git clone https://github.com/prathameshmohod174/genai_movie_recommendation_system.git
cd genai_movie_recommendation_system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root and add:
```
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

4. Download required data files:
Download the zip from the link - [Dataset](https://grouplens.org/datasets/movielens/32m/)
- Extract the the zip.
- Copy the three files 
   - movies.csv
   - movie_metadata.csv
   - ratings.csv
   to the project directory.



Place these files in the project root directory.

## 🚀 Running the Application

1. Initialize the database:
```bash
python database.py
```

2. Start the Flask server:
```bash
python app.py
```

3. Access the application at: http://localhost:5000


## 🔄 Project Flow

1. **User Onboarding**
   - Sign up/login
   - Select preferred genres
   - Initial preference analysis

2. **Recommendation Process**
   - Hybrid filtering combines multiple approaches
   - Real-time preference updates
   - Continuous learning from user behavior

3. **GenAI Enhancement**
   - Behavior pattern analysis
   - Personalized insights
   - Dynamic content adaptation

## 📝 API Documentation

### Groq API Integration
The system uses Groq API for GenAI features. Replace `YOUR_GROQ_API_KEY` in the `.env`, `config.json` and `genai_enhancer.py` file with your actual API key.

### Endpoints
- `/signup` - User registration
- `/login` - User authentication
- `/select_genres` - Genre preference selection
- `/rate_movies` - Movie rating interface
- `/watch_movies` - Watch history tracking
- `/recommendations` - Get personalized recommendations
- `/activity` - View user activity and insights

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Personal Use License - see the [LICENSE](LICENSE) file for details.
This license allows personal, non-commercial use only. Commercial use is strictly prohibited.

## 👥 Authors

- **Prathamesh Mohod** - *Initial work* - [GitHub Profile](https://github.com/prathameshmohod174)

## 🙏 Acknowledgments

- MovieLens dataset for training data
- Groq API for GenAI capabilities
- Flask and SQLAlchemy communities 