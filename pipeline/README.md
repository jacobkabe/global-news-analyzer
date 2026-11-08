# Pipeline (upgraded NLP version)

The new NLP pipeline. The first version in `nlp/` stays as it is.
Every step reads and writes CSV files in `data/raw/outputs/` (ignored by Git), joined by `article_id`.
Only small results (metrics, plots, samples) are committed, to `results/`, `failures/` and `data/sample/`.

Run each step from the project folder, in this order:

| Step | Command | Output | Status |
|---|---|---|---|
| 1. Clean | `python pipeline\clean.py` | `articles_clean.csv` | done |
| 2. Sentiment | `python pipeline\sentiment.py run` / `evaluate` / `explain` | `sentiment.csv`, `sentiment_explain.csv`, `results/sentiment_*` | done |
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

## 2. Sentiment (`sentiment.py`)
Headline only (full text is for the story summary). Gold labels: `data/sample/sentiment_gold.csv`
(codebook in `data/sample/README.md`). Details of the models and commands: the top of `sentiment.py`.

**Results on the 150 random headlines** (`results/sentiment_metrics.csv`; "tuned" = neutral cut-offs tuned
with 5-fold cross-validation):

| Model | Labels | Accuracy | Macro-F1 (95% interval) |
|---|---|---|---|
| VADER (baseline, word list) | own / tuned | 57% / 70% | 0.51 / 0.58 (0.46-0.70) |
| Twitter RoBERTa | own / tuned | 73% / 77% | 0.55 (0.46-0.66) / 0.52 |
| BART-MNLI zero-shot | own / tuned | 35% / 70% | 0.28 / 0.58 (0.49-0.67) |
| DistilBERT SST-2 (failure example) | own / tuned | 32% / 51% | 0.24 / 0.44 |

**Which model does what, and why**
- **RoBERTa gives the label and score.** Highest accuracy, best macro-F1 on all 230 gold headlines (0.63),
  a real neutral class (63% of our headlines are neutral), and about 20x faster than BART-MNLI.
  VADER-tuned and BART-tuned have a similar macro-F1; the intervals overlap, so 150 headlines cannot
  separate them and the practical reasons decide.
- **Each label is explained by the model that gave it** (`sentiment.py explain`). RoBERTa: occlusion
  (remove one word, see how much RoBERTa's own score changes). VADER: its word list values.
  Showing VADER's words next to RoBERTa's label would mix two models and could contradict the label.
- **VADER** is the baseline the course asks for, and the most transparent model (fixed word values).
- **BART-MNLI** is a third, independent opinion: never trained on sentiment, only asked our codebook question.
  Its own labels almost never say "neutral" (1 of 230), so only its tuned score is useful.
- **DistilBERT** is kept as a documented failure: trained on movie reviews, it has no neutral class.

**Known limits**
- Positive news is rare (8 of 150 random headlines) and under-detected by every model. Typical failure:
  good news told with bad words ("Rescuers find two more people alive after flash flood disaster").
- The 80 extra "positive candidate" headlines were picked by the models, not at random (only 17 were positive).
- One annotator. The Guardian is the hardest outlet for every model (55-68% accuracy).
- All failures with every model's label and driving words: `failures/sentiment_gold_vs_models.csv`.
