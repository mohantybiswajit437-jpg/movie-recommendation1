from fastapi import FastAPI, HTTPException
from database import init_db, save_search
import pandas as pd
import pickle

# ==========================================
# Initialize Database
# ==========================================

init_db()

# ==========================================
# FastAPI App
# ==========================================

app = FastAPI(
    title="Movie Recommendation API",
    description="Content-Based Movie Recommendation System",
    version="2.0.0"
)

# ==========================================
# Load ML Files (Loads only once)
# ==========================================

movies = pd.read_pickle("movies.pkl")

with open("recommendation_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

indices = pd.Series(
    movies.index,
    index=movies["title"]
).drop_duplicates()

# ==========================================
# Home
# ==========================================

@app.get("/")
def home():
    return {
        "status": "success",
        "message": "Movie Recommendation API is running 🚀",
        "version": "2.0.0"
    }

# ==========================================
# Get All Movies
# ==========================================

@app.get("/movies")
def get_movies():

    return {
        "count": len(movies),
        "movies": sorted(movies["title"].tolist())
    }

# ==========================================
# Search Movies
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

    return {
        "count": len(result),
        "results": result.to_dict(orient="records")
    }

# ==========================================
# Recommend Movies
# ==========================================

@app.get("/recommend/{movie_name}")
def recommend(movie_name: str):

    if movie_name not in indices:

        raise HTTPException(
            status_code=404,
            detail="Movie not found."
        )

    save_search(movie_name)

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

            "genres": movie.get("genres", ""),

            "rating": float(movie.get("vote_average", 0)),

            "release_date": str(movie.get("release_date", "")),

            "language": movie.get("original_language", ""),

            "overview": movie.get("overview", ""),

            "poster_path": movie.get("poster_path", "")

        })

    return {

        "status": "success",

        "searched_movie": movie_name,

        "total_recommendations": len(recommendations),

        "recommendations": recommendations

    }