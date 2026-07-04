import streamlit as st
import pandas as pd
import pickle

# --------------------------
# Page Configuration
# --------------------------

st.set_page_config(
    page_title="AI Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# --------------------------
# Load Files
# --------------------------

movies = pd.read_pickle("movies.pkl")

model = pickle.load(open("recommendation_model.pkl", "rb"))

tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))

# --------------------------
# Create Index
# --------------------------

indices = pd.Series(
    movies.index,
    index=movies["title"]
).drop_duplicates()

# --------------------------
# Recommendation Function
# --------------------------

def recommend(movie):

    idx = indices[movie]

    distances, neighbors = model.kneighbors(
        tfidf_matrix[idx],
        n_neighbors=11
    )

    return movies.iloc[
        neighbors[0][1:]
    ]

# --------------------------
# Sidebar
# --------------------------

st.sidebar.title("🎬 Movie Recommender")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dashboard",
        "About"
    ]
)

# ==========================
# HOME
# ==========================

if page == "Home":

    st.title("🎬 AI Movie Recommendation System")

    st.write(
        "Find similar movies using Machine Learning."
    )

    movie = st.selectbox(
        "Choose a Movie",
        sorted(movies["title"].unique())
    )

    if st.button("Recommend"):

        result = recommend(movie)

        st.success("Top Recommendations")

        cols = st.columns(2)

        for i, (_, row) in enumerate(result.iterrows()):

            with cols[i % 2]:

                st.subheader(row["title"])

                if "genres" in result.columns:
                    st.write("🎭", row["genres"])

                if "vote_average" in result.columns:
                    st.write("⭐", row["vote_average"])

                if "overview" in result.columns:
                    st.write(row["overview"][:200] + "...")

                st.divider()

# ==========================
# DASHBOARD
# ==========================

elif page == "Dashboard":

    st.title("📊 Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Movies",
        len(movies)
    )

    if "original_language" in movies.columns:
        col2.metric(
            "Languages",
            movies["original_language"].nunique()
        )

    if "genres" in movies.columns:
        col3.metric(
            "Genres",
            movies["genres"].nunique()
        )

    st.subheader("Top Rated Movies")

    if "vote_average" in movies.columns:

        st.dataframe(

            movies[
                ["title", "vote_average"]
            ]

            .sort_values(
                "vote_average",
                ascending=False
            )

            .head(20)

        )

# ==========================
# ABOUT
# ==========================

else:

    st.title("About")

    st.markdown("""

## AI Movie Recommendation System

### Built Using

- Python
- Pandas
- Scikit-Learn
- TF-IDF
- Nearest Neighbors
- Streamlit

### Features

- Movie Recommendation
- Dashboard
- Machine Learning
- NLP

""")