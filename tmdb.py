import requests

API_KEY = "your api key"

BASE_URL = "https://api.themoviedb.org/3/movie/"
IMAGE_URL = "https://image.tmdb.org/t/p/w500"


def fetch_poster(movie_id):

    url = f"{BASE_URL}{movie_id}?api_key={API_KEY}"

    data = requests.get(url).json()

    if data.get("poster_path"):

        return IMAGE_URL + data["poster_path"]

    return None