import streamlit as st
import pandas as pd
import pickle
import google.generativeai as genai

# ===========================================
# PAGE CONFIG
# ===========================================

st.set_page_config(
    page_title="🎬 AI Movie Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Movie Recommendation Assistant")

st.write(
    "Ask anything about movies. This chatbot uses your movie database (RAG) with Gemini AI."
)

# ===========================================
# GEMINI API
# ===========================================

genai.configure(api_key="Your API")

gemini = genai.GenerativeModel("gemini-2.5-flash")

# ===========================================
# LOAD FILES
# ===========================================

movies = pd.read_pickle("movies.pkl")

knn_model = pickle.load(open("recommendation_model.pkl", "rb"))

tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# ===========================================
# CHAT HISTORY
# ===========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ===========================================
# RAG FUNCTION
# ===========================================

def rag(question):

    # Convert question to TF-IDF vector
    vector = tfidf.transform([question])

    # Retrieve top 5 similar movies
    distances, indices = knn_model.kneighbors(
        vector,
        n_neighbors=5
    )

    retrieved_movies = movies.iloc[indices[0]]

    # Build context
    context = ""

    for _, movie in retrieved_movies.iterrows():

        context += f"""
Movie Title: {movie['title']}

Genre: {movie.get('genres','Unknown')}

Language: {movie.get('original_language','Unknown')}

Rating: {movie.get('vote_average','N/A')}

Overview:
{movie.get('overview','No overview available.')}

-----------------------------------------
"""

    # Prompt for Gemini
    prompt = f"""
You are an expert movie recommendation assistant.

Use ONLY the movie information below.

{context}

User Question:
{question}

Instructions:
1. Recommend the best matching movies.
2. Explain why each movie matches.
3. Mention genre and rating.
4. Do NOT invent movies.
5. Only use the provided context.
"""

    response = gemini.generate_content(prompt)

    return response.text

# ===========================================
# USER INPUT
# ===========================================

question = st.chat_input("Ask about movies...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Searching movies and asking Gemini..."):

        answer = rag(question)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):
        st.markdown(answer)