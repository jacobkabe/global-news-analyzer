# Results

Small result files made by the pipeline (`pipeline/README.md` says which command makes which file).
What the numbers mean and their limits: `docs/findings.md`.

| File | What it shows | Made by |
|---|---|---|
| `sentiment_metrics.csv` | Accuracy, macro-F1 (95% bootstrap interval) and F1 per class for each model; `eval_set` = `random150` (main result), `all230`, `team_random` / `team_all` (held-out test on the teammates' labels) | `sentiment.py evaluate` |
| `sentiment_cutoffs.csv` | The cut-offs the dashboard uses for VADER and BART (tuned on the 230 hand labels) | `sentiment.py evaluate` |
| `sentiment_by_outlet.csv` | Accuracy of each model per outlet (random 150) | `sentiment.py evaluate` |
| `sentiment_confusion.png` | Confusion matrices, tuned labels, random 150 | `sentiment.py evaluate` |
| `sentiment_agreement.csv` | Agreement and Cohen's kappa between annotators and with our reviewed labels | `labels.py collect` |
| `sentiment_explain_comparison.csv` | Driving words: deleting vs masking a word, compared on the 230 | `sentiment.py compare-explain` |
| `ner_metrics.csv` | Entities: precision, recall, F1 of spaCy sm vs trf; name rules before vs after | `ner.py evaluate` |
| `ner_top_entities.csv` | The 100 most mentioned entities with their name variants | `ner.py normalise` |
