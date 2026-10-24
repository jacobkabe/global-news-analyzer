"""Export the articles table to a CSV file that opens in Excel.

python src\collect\export_csv.py             -> data\raw\news.csv (all articles, with full text)
python src\collect\export_csv.py --no-text   -> without the long 'text' column (easier to read in Excel)
The CSV goes to data/raw/ (ignored by Git): do not commit it.
"""
import csv
import sqlite3
import sys

from db import DB_PATH

CSV_PATH = DB_PATH.with_name("news.csv")
COLUMNS = ["id", "source", "url", "published_at", "title", "summary", "text", "section", "collected_at"]


def export_csv(include_text=True):
    columns = [c for c in COLUMNS if include_text or c != "text"]
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(f"SELECT {', '.join(columns)} FROM articles ORDER BY published_at DESC").fetchall()
    conn.close()
    # utf-8-sig adds a marker so Excel shows accents and quotes correctly
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    count = export_csv(include_text="--no-text" not in sys.argv)
    print(f"{count} articles written to {CSV_PATH}")
