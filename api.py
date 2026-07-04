from fastapi import FastAPI
from database import init_db, save_search
import pandas as pd
import pickle

init_db()
# ==========================================
# Create FastAPI App
# ==========================================

app = FastAPI(
    title="🎬 Movie Recommendation API",
    description="Content-Based Movie Recommendation System",
    version="1.0.0"
)

# ==========================================
# Load Files
# ==========================================

movies = pd.read_pickle("movies.pkl")

model = pickle.load(open("recommendation_model.pkl", "rb"))

tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))

# ==========================================
# Create Movie Index
# ==========================================

indices = pd.Series(
    movies.index,
    index=movies["title"]
).drop_duplicates()

# ==========================================
# Home API
# ==========================================

@app.get("/")
def home():
    return {
        "message": "Movie Recommendation API is Running 🚀"
    }

# ==========================================
# Get All Movies
# ==========================================

@app.get("/movies")
def get_movies():

    return movies["title"].tolist()

# ==========================================
# Recommend Movies
# ==========================================

@app.get("/recommend/{movie_name}")
def recommend(movie_name: str):

    if movie_name not in indices:

        return {
            "error": "Movie not found."
        }

    idx = indices[movie_name]

    distances, neighbors = model.kneighbors(
        tfidf_matrix[idx],
        n_neighbors=11
    )

    recommendations = []

    for i in neighbors[0][1:]:

        movie = movies.iloc[i]

        recommendations.append({

            "id": int(movie["id"]),

            "title": movie["title"],

            "genre": movie.get("genres", "Unknown"),

            "rating": float(movie.get("vote_average", 0)),

            "release_date": str(movie.get("release_date", "")),

            "language": movie.get("original_language", ""),

            "overview": movie.get("overview", ""),

            "poster_path": movie.get("poster_path", "")

        })

    return recommendations

# ==========================================
# Search Movie
# ==========================================

@app.get("/search/{movie_name}")
def search_movie(movie_name: str):

    result = movies[
        movies["title"].str.contains(
            movie_name,
            case=False,
            na=False
        )
    ][["title"]].head(20)

    return result.to_dict(orient="records")

@app.get("/recommend/{movie}")
def recommend(movie: str):

    save_search(movie)

    return {
        "movie": movie,
        "recommendations": ["Titanic", "Inception"]
    }