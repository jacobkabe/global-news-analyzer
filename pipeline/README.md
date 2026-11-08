# Pipeline (upgraded NLP version)

The new NLP pipeline. The first version in `nlp/` stays as it is.
Every step reads and writes CSV files in `data/raw/outputs/` (ignored by Git), joined by `article_id`.
Only small results (metrics, plots, samples) are committed, to `results/`, `failures/` and `data/sample/`.

Run each step from the project folder, in this order:

| Step | Command | Output | Status |
|---|---|---|---|
| 1. Clean | `python pipeline\clean.py` | `articles_clean.csv` | done |
| 2. Sentiment | `python pipeline\sentiment.py run` / `evaluate` / `explain` | `sentiment.csv`, `results/sentiment_*` | planned |
| 3. Entities | `python pipeline\ner.py ...` (with name normalisation: "Trump" = "Donald Trump") | `entities.csv` | planned |
| 4. Stories | `python pipeline\cluster.py ...` (same event across outlets) | `stories.csv` | later |
| 5. Dashboard | `streamlit run app\streamlit_app.py` | | planned |

## 1. Clean (`clean.py`)
Reads `data/raw/news.db` (or `--input news.csv`) and writes one row per article:

- `headline`: what the models read. HTML, invisible characters and the Guardian `| Author` ending removed,
  curly quotes made straight (VADER only recognises words next to straight quotes), series labels
  such as `Country diary:` or `LIVE:` removed.
- `headline_raw`: the title as collected, for the dashboard.
- `headline_sentcase`: Title Case (NYT) turned into sentence case, for the NER casing experiment.
- `content_type`: `news`, `opinion`, `briefing`, `column`, `letter`, `live`, `media`, `callout`.
  A briefing or a reader call-out is not one news event, so later steps can skip anything that is not `news`.
- `topic`: the outlets' sections merged into one list (`U.S.` and `US news` -> `US`).
- `outlet`: readable outlet name (`The Guardian`).
- `lead` (summary), `n_text_chars` (length of the full text; only the Guardian has it), `date`, `url`.
