import requests

movie = "Avatar"

url = f"http://127.0.0.1:8000/recommend/{movie}"

response = requests.get(url)

print("Status Code:", response.status_code)

data = response.json()

print("\nRecommended Movies:\n")

for m in data:
    print(m["title"], "-", m["rating"])