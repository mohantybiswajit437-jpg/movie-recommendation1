import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

st.title("📊 Movie Dashboard")

movies = pd.read_pickle("movies.pkl")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Movies",
    len(movies)
)

col2.metric(
    "Languages",
    movies["original_language"].nunique()
)

col3.metric(
    "Genres",
    movies["genres"].nunique()
)

st.divider()

st.subheader("⭐ Top Rated Movies")

top_movies = movies.sort_values(
    by="vote_average",
    ascending=False
).head(10)

st.dataframe(
    top_movies[
        ["title", "vote_average"]
    ],
    use_container_width=True
)

st.divider()

st.subheader("🎬 Most Popular Movies")

popular = movies.sort_values(
    by="popularity",
    ascending=False
).head(10)

fig = px.bar(
    popular,
    x="title",
    y="popularity"
)

st.plotly_chart(fig, use_container_width=True)