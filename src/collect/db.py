import sqlite3
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# data/raw/news.db in the repo root (ignored by Git)
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "news.db"

# Query parameters that only track where a visitor came from (not part of the article address)
TRACKING_PREFIXES = ("utm_", "at_")  # utm_source=..., BBC: at_medium=RSS, at_campaign=rss
TRACKING_NAMES = {"traffic_source", "fbclid", "gclid", "ocid", "cmp", "intcmp", "smid", "taid", "xtor"}


def clean_url(url):
    """Remove tracking parameters and the #fragment from a URL, so one article has one address."""
    if not url:
        return url
    parts = urlsplit(url.strip())
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith(TRACKING_PREFIXES) and k.lower() not in TRACKING_NAMES]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def normalise_date(value):
    """Return a date as UTC text in one format: 2026-10-17T14:28:07Z.
    Dates without a time zone are taken as UTC. Text we cannot read is kept unchanged."""
    if not value:
        return value
    try:
        dt = datetime.fromisoformat(value.strip())  # 2026-10-17T14:28:07Z, ...+00:00, ...18.167Z
    except ValueError:
        try:
            dt = parsedate_to_datetime(value)  # RSS style: Sat, 17 Oct 2026 14:28:07 GMT
        except (TypeError, ValueError):
            return value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    """Save a list of article dicts. Dates are stored as UTC and URLs without tracking parameters.
    Articles whose URL is already stored are skipped. Returns how many new articles were added."""
    articles = [dict(a, url=clean_url(a["url"]), published_at=normalise_date(a.get("published_at")))
                for a in articles]
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


def normalise_existing():
    """One-time cleanup of rows saved before this normalising existed (python src\\collect\\db.py).
    Returns (rows changed, rows left unchanged because the cleaned URL already exists)."""
    conn = get_connection()
    changed = 0
    conflicts = []
    for row_id, url, published_at in conn.execute("SELECT id, url, published_at FROM articles").fetchall():
        new_url, new_date = clean_url(url), normalise_date(published_at)
        if (new_url, new_date) == (url, published_at):
            continue
        try:
            conn.execute("UPDATE articles SET url = ?, published_at = ? WHERE id = ?",
                         (new_url, new_date, row_id))
            changed += 1
        except sqlite3.IntegrityError:  # another row already has this cleaned URL: do not delete anything
            conflicts.append(url)
    conn.commit()
    conn.close()
    return changed, conflicts


if __name__ == "__main__":
    changed, conflicts = normalise_existing()
    print(f"{changed} rows normalised, {len(conflicts)} left unchanged (duplicate URL after cleaning).")
    for url in conflicts:
        print("  duplicate:", url)
