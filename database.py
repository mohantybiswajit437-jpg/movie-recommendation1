import sqlite3

def get_connection():
    conn = sqlite3.connect("movies.db")
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS search_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        movie TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def save_search(movie):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("INSERT INTO search_history (movie) VALUES (?)", (movie,))
    
    conn.commit()
    conn.close()