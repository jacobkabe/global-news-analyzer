"""
merge_db.py - add the articles that a team member collected on their own computer to our main database.

    python src\\collect\\merge_db.py path\\to\\their_news.db       their SQLite database
    python src\\collect\\merge_db.py path\\to\\their_news.csv      or their export (src\\collect\\export_csv.py)

Rules
- Always merge INTO the main database (data/raw/news.db on Jakub's computer), never the other way round:
  our hand labels and all pipeline outputs refer to the article id in the main database. New articles get
  new ids after the existing ones, so the existing ids never change.
- An article whose URL is already in the main database is skipped (the same rule as save_articles()).
  URLs and dates are cleaned the same way as when collecting.
- A backup of the main database is made first: data/raw/news_backup_before_merge_<date>.db

After merging, run the pipeline again; every step only does the new articles:
    python pipeline\\clean.py
    python pipeline\\sentiment.py run
    python pipeline\\sentiment.py explain
    python pipeline\\ner.py run
    python pipeline\\ner.py normalise
"""
import csv
import re
import shutil
import sqlite3
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from db import DB_PATH, clean_url, get_connection, normalise_date

COLUMNS = ["source", "url", "published_at", "title", "summary", "text", "section", "collected_at"]


def read_excel_damaged_csv(path):
    """Read a CSV that was opened and saved in Excel with Czech settings. Excel then adds ';;;' to the lines,
    puts most lines into ONE quoted field, and splits a line where a summary contains a line break.
    Here every line is unpacked again, the ';' are removed and split lines are joined back together."""
    rows = list(csv.reader(open(path, encoding="utf-8-sig", newline="")))
    header = [h.rstrip(";") for h in rows[0]]
    n = len(header)
    out, waiting = [], None
    for r in rows[1:]:
        if not any(x.strip(" ;") for x in r):
            continue                                      # an empty line
        if len(r) == 1 or re.match(r"\d+,[^,]*,https?://", r[0]):
            # the line (or its start) was put into quotes as one field: read the text once more
            r = next(csv.reader([",".join(r)]))
        r = [x.rstrip(";") for x in r]
        if waiting is not None:                           # second half of a line split by a line break
            joined = (waiting[-1] + " " + r[0]).replace('""', '"')
            if joined.count('"') % 2:                     # a quote left over from Excel's quoting
                joined = joined.rstrip('"')
            r = waiting[:-1] + [joined] + r[1:]
            waiting = None
        if len(r) < n:
            waiting = r                                   # first half: wait for the rest on the next line
            continue
        out.append(r[:n])
    print(f"repaired an Excel-damaged CSV: {len(out)} articles read")
    return pd.DataFrame(out, columns=header)


def read_other(path):
    if path.suffix == ".db":
        conn = sqlite3.connect(path)
        df = pd.read_sql("SELECT * FROM articles", conn)
        conn.close()
    elif open(path, encoding="utf-8-sig").readline().rstrip().endswith(";"):
        df = read_excel_damaged_csv(path)
    else:
        df = pd.read_csv(path, keep_default_na=False, encoding="utf-8-sig")
    for col in COLUMNS:          # an export without the full text has no "text" column
        if col not in df.columns:
            df[col] = None
    df = df[COLUMNS].replace({"": None})
    df["url"] = df["url"].map(clean_url)
    df["published_at"] = df["published_at"].map(normalise_date)
    return df[df["url"].notna() & df["source"].notna()]


def merge(path):
    other = read_other(path)
    backup = DB_PATH.with_name(f"news_backup_before_merge_{date.today():%Y%m%d}.db")
    shutil.copy(DB_PATH, backup)
    print(f"backup: {backup.name}")

    conn = get_connection()
    known = {u for (u,) in conn.execute("SELECT url FROM articles")}
    new = other[~other["url"].isin(known)].drop_duplicates("url")
    rows = new.where(new.notna(), None).to_dict("records")
    for r in rows:
        r["collected_at"] = r["collected_at"] or date.today().isoformat()
    conn.executemany(f"""
        INSERT OR IGNORE INTO articles ({', '.join(COLUMNS)})
        VALUES ({', '.join(':' + c for c in COLUMNS)})
    """, rows)
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    conn.close()

    print(f"{len(other)} articles in {path.name}: {len(new)} new, {len(other) - len(new)} already in our database")
    if len(new):
        print(new["source"].value_counts().rename("new articles").to_string())
    print(f"the main database now has {total} articles")


if __name__ == "__main__":
    if len(sys.argv) != 2 or not Path(sys.argv[1]).exists():
        sys.exit("usage: python src\\collect\\merge_db.py path\\to\\their_news.db (or .csv)")
    merge(Path(sys.argv[1]))
