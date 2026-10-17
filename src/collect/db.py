import sqlite3
from pathlib import Path

# data/raw/news.db in the repo root (ignored by Git)
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "news.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            source       TEXT NOT NULL,
            url          TEXT NOT NULL UNIQUE,
            published_at TEXT,
            title        TEXT,
            summary      TEXT,
            text         TEXT,
            section      TEXT,
            collected_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return conn


def save_articles(articles):
    """Save a list of article dicts. Articles whose URL is already stored are skipped.
    Returns how many new articles were added."""
    conn = get_connection()
    before = conn.total_changes
    conn.executemany("""
        INSERT OR IGNORE INTO articles (source, url, published_at, title, summary, text, section)
        VALUES (:source, :url, :published_at, :title, :summary, :text, :section)
    """, articles)
    conn.commit()
    added = conn.total_changes - before
    conn.close()
    return added
