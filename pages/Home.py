import streamlit as st
import pandas as pd
import pickle

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="🎬 AI Movie Recommendation",
    page_icon="🎬",
    layout="wide"
)

# ============================================
# Load Data
# ============================================

movies = pd.read_pickle("movies.pkl")

model = pickle.load(open("recommendation_model.pkl", "rb"))

tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))

# ============================================
# Create Movie Index
# ============================================

indices = pd.Series(
    movies.index,
    index=movies["title"]
).drop_duplicates()

# ============================================
# Recommendation Function
# ============================================

def recommend(movie_name):

    idx = indices[movie_name]

    distances, neighbors = model.kneighbors(
        tfidf_matrix[idx],
        n_neighbors=11
    )

    return movies.iloc[neighbors[0][1:]]

# ============================================
# Hero Banner
# ============================================

st.title("🎬 AI Movie Recommendation System")

st.markdown("""
### Discover Your Next Favorite Movie 🍿

Find movies similar to your favorite movie using Machine Learning.
""")

st.divider()

# ============================================
# Filters
# ============================================

left, right = st.columns(2)

with left:

    genres = sorted(movies["genres"].dropna().unique())

    selected_genre = st.selectbox(
        "🎭 Genre",
        ["All"] + genres
    )

with right:

    min_rating = st.slider(
        "⭐ Minimum Rating",
        0.0,
        10.0,
        5.0
    )

# ============================================
# Movie Search
# ============================================

selected_movie = st.selectbox(
    "🔍 Search Movie",
    sorted(movies["title"].unique())
)

# ============================================
# Featured Movie
# ============================================

movie = movies[movies["title"] == selected_movie].iloc[0]

st.markdown("---")

st.subheader("🔥 Featured Movie")

col1, col2 = st.columns([1, 3])

with col1:

    st.image(
        "https://via.placeholder.com/250x370.png?text=Movie+Poster",
        use_container_width=True
    )

with col2:

    st.header(movie["title"])

    st.write(f"⭐ Rating : {movie['vote_average']}")

    st.write(f"🎭 Genre : {movie['genres']}")

    st.write(f"🌍 Language : {movie['original_language']}")

    st.write(f"📅 Release Date : {movie['release_date']}")

    st.write(movie["overview"])

st.markdown("---")

# ============================================
# Recommendation Button
# ============================================

if st.button("🎥 Recommend Movies", use_container_width=True):

    recommendations = recommend(selected_movie)

    if selected_genre != "All":

        recommendations = recommendations[
            recommendations["genres"].str.contains(
                selected_genre,
                case=False,
                na=False
            )
        ]

    recommendations = recommendations[
        recommendations["vote_average"] >= min_rating
    ]

    st.subheader("🍿 Recommended Movies")

    cols = st.columns(5)

    for i, (_, row) in enumerate(recommendations.iterrows()):

        with cols[i % 5]:

            st.image(
                "https://via.placeholder.com/200x300.png?text=Poster",
                use_container_width=True
            )

            st.markdown(f"### {row['title']}")

            st.write(f"⭐ {row['vote_average']}")

            st.write(f"🎭 {row['genres']}")

            st.write(f"🌍 {row['original_language']}")

            if pd.notna(row["release_date"]):
                st.write(f"📅 {row['release_date']}")

            overview = str(row["overview"])

            if len(overview) > 120:
                overview = overview[:120] + "..."
            st.caption(overview)