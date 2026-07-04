import streamlit as st
import requests

# ==========================================================
# Configuration
# ==========================================================

API_URL = "https://movie-recommendation1-7.onrender.com"

st.set_page_config(
    page_title="🎬 AI Movie Recommendation",
    page_icon="🎬",
    layout="wide"
)

# ==========================================================
# Load Movies from API
# ==========================================================

@st.cache_data
def load_movies():
    try:
        response = requests.get(f"{API_URL}/movies", timeout=30)
        response.raise_for_status()
        return response.json()["movies"]
    except Exception:
        return []

movies = load_movies()

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("🎬 Movie Recommendation")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "About"
    ]
)

# ==========================================================
# HOME
# ==========================================================

if page == "Home":

    st.title("🎬 AI Movie Recommendation System")

    st.write(
        "Get movie recommendations using Machine Learning powered by FastAPI."
    )

    if not movies:
        st.error("Unable to connect to the API.")
        st.stop()

    movie = st.selectbox(
        "Choose a Movie",
        movies
    )

    if st.button("Recommend"):

        with st.spinner("Finding similar movies..."):

            try:

                response = requests.get(
                    f"{API_URL}/recommend/{movie}",
                    timeout=60
                )

                response.raise_for_status()

                data = response.json()

                recommendations = data["recommendations"]

                st.success(
                    f"Top {len(recommendations)} Recommendations"
                )

                cols = st.columns(2)

                for i, rec in enumerate(recommendations):

                    with cols[i % 2]:

                        poster = rec.get("poster_path", "")

                        if poster:

                            st.image(
                                f"https://image.tmdb.org/t/p/w500{poster}",
                                use_container_width=True
                            )

                        st.subheader(rec["title"])

                        st.write(
                            f"⭐ Rating : {rec['rating']}"
                        )

                        st.write(
                            f"🎭 Genre : {rec['genres']}"
                        )

                        st.write(
                            f"🌍 Language : {rec['language']}"
                        )

                        st.write(
                            f"📅 Release : {rec['release_date'][:10]}"
                        )

                        st.write(rec["overview"])

                        st.divider()

            except Exception as e:

                st.error("Unable to get recommendations.")

                st.exception(e)

# ==========================================================
# ABOUT
# ==========================================================

else:

    st.title("About")

    st.markdown("""

## 🎬 AI Movie Recommendation System

### Architecture

Streamlit Frontend

⬇

FastAPI Backend (Render)

⬇

Machine Learning Model

### Technologies

- Python
- FastAPI
- Streamlit
- Scikit-Learn
- TF-IDF
- Nearest Neighbors
- Render
- GitHub

""")