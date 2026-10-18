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
from db import save_articles

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


def collect_guardian():
    """Guardian articles of yesterday and today."""
    today = date.today()
    articles = guardian.fetch_guardian((today - timedelta(days=1)).isoformat(), today.isoformat())
    return len(articles), save_articles(articles)


def collect_nyt():
    """NYT news of yesterday and today (Article Search, about 7 minutes because of the rate limit)."""
    return nyt.fetch_search(1)


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
