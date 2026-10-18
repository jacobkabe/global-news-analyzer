import logging
import os
import re
import sys
import time
import traceback
from datetime import date, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

import guardian
import nyt
import rss
from db import get_connection, save_articles

load_dotenv()

# data/raw/collect.log in the repo root (ignored by Git)
LOG_FILE = Path(__file__).resolve().parents[2] / "data" / "raw" / "collect.log"


def scrub(text):
    """Hide API keys in a text before it is written to the log."""
    text = re.sub(r"(api-key=)[^&\s'\")]+", r"\1***", text)
    for name in ("GUARDIAN_API_KEY", "NYT_API_KEY"):
        key = os.getenv(name)
        if key:
            text = text.replace(key, "***")
    return text


class ScrubFormatter(logging.Formatter):
    def format(self, record):
        return scrub(super().format(record))


def setup_log():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    formatter = ScrubFormatter("%(asctime)s %(levelname)s %(message)s")
    # the log file stops growing at about 1 MB (3 older copies are kept)
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    console = logging.StreamHandler()
    for handler in (file_handler, console):
        handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console])


# A gap longer than this is filled in steps: this many days per run, oldest first, so the next run
# continues where this one stopped. NYT: about 17 requests per day, and NYT allows 500 per day.
MAX_GUARDIAN_DAYS = 90
MAX_NYT_DAYS = 14


def newest_date(source):
    """Day of the newest article of this source in the database (UTC), or None if there is none."""
    conn = get_connection()
    newest = conn.execute("SELECT MAX(published_at) FROM articles WHERE source = ?", (source,)).fetchone()[0]
    conn.close()
    return date.fromisoformat(newest[:10]) if newest else None


def catch_up_range(source, max_days):
    """First and last day to fetch: from the newest article we already have up to today.
    The newest day is fetched again because it may have been incomplete, and the range is
    at least yesterday + today. Duplicates are skipped when saving."""
    today = date.today()
    newest = newest_date(source)
    start = min(newest, today - timedelta(days=1)) if newest else today - timedelta(days=1)
    end = min(today, start + timedelta(days=max_days))
    if end < today:
        logging.warning("%s: gap since %s is longer than %d days - fetching %s to %s now, the rest on the next runs",
                        source, start, max_days, start, end)
    else:
        logging.info("%s: newest article is from %s - fetching %s to %s", source, newest, start, end)
    return start, end


def collect_guardian():
    """Guardian articles since the newest one we have."""
    start, end = catch_up_range("guardian", MAX_GUARDIAN_DAYS)
    articles = guardian.fetch_guardian(start.isoformat(), end.isoformat())
    return len(articles), save_articles(articles)


def collect_nyt():
    """NYT news since the newest one we have (Article Search, about 3-4 minutes per day because of the rate limit)."""
    start, end = catch_up_range("nyt", MAX_NYT_DAYS)
    return nyt.fetch_search((end - start).days, last_day=end)


def run_source(name, function):
    """Run one source. A failure is logged and does not stop the other sources. Returns True if it worked."""
    logging.info("%s: start", name)
    started = time.time()
    try:
        downloaded, added = function()
    except Exception as e:
        logging.error("%s: FAILED - %s: %s", name, type(e).__name__, e)
        logging.error("%s", traceback.format_exc())
        return False
    logging.info("%s: downloaded %d, %d new saved (%.0f s)", name, downloaded, added, time.time() - started)
    return True


if __name__ == "__main__":
    # python src\collect\collect.py                  all sources
    # python src\collect\collect.py guardian nyt     only these sources (names: guardian, nyt, bbc, ...)
    setup_log()
    known = rss.known_urls()
    sources = {"guardian": collect_guardian}
    for source, feed_url in rss.FEEDS.items():
        sources[source] = lambda s=source, u=feed_url: rss.collect_feed(s, u, known)
    sources["nyt"] = collect_nyt  # last, because it is the slowest

    wanted = sys.argv[1:] or list(sources)
    unknown = [name for name in wanted if name not in sources]
    if unknown:
        sys.exit(f"Unknown source(s): {', '.join(unknown)}. Choose from: {', '.join(sources)}")

    logging.info("=== collection started (%s) ===", ", ".join(wanted))
    failed = [name for name in wanted if not run_source(name, sources[name])]
    if failed:
        logging.error("=== finished with errors, failed: %s ===", ", ".join(failed))
        sys.exit(1)
    logging.info("=== finished, all %d sources ok ===", len(wanted))
