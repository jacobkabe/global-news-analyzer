# Pipeline (upgraded NLP version)

The new NLP pipeline. The first version in `nlp/` stays as it is.
Every step reads and writes CSV files in `data/raw/outputs/` (ignored by Git), joined by `article_id`.
Only small results (metrics, plots, samples) are committed, to `results/`, `failures/` and `data/sample/`.

Run each step from the project folder, in this order:

| Step | Command | Output | Status |
|---|---|---|---|
| 1. Clean | `python pipeline\clean.py` | `articles_clean.csv` | done |
| 2. Sentiment | `python pipeline\sentiment.py run` / `evaluate` / `explain` | `sentiment.csv`, `sentiment_explain.csv`, `results/sentiment_*` | done |
| 3. Entities | `python pipeline\ner.py run` / `normalise` / `sheets` / `evaluate` | `entities.csv`, `results/ner_*` | built; waiting for labels |
| 4. Stories | `python pipeline\cluster.py ...` (same event across outlets) | `stories.csv` | later |
| 5. Dashboard | `streamlit run app\streamlit_app.py` | (reads the files above) | v2 built; stories come with step 4 |

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
with 5-fold cross-validation). The 150 are random within each outlet (20 each for the small RSS outlets),
while most of our data is Guardian + NYT. Labels after the codebook review of 12 Nov 2026 (see below).

| Model | Labels | Accuracy | Macro-F1 (95% interval) |
|---|---|---|---|
| Always "neutral" (no model) | – | 53% | 0.23 |
| VADER (baseline, word list) | own / tuned | 61% / 71% | 0.54 / 0.57 (0.47-0.68) |
| Twitter RoBERTa | own / tuned | 71% / 73% | 0.58 (0.46-0.68) / 0.60 (0.48-0.71) |
| BART-MNLI zero-shot | own / tuned | 45% / 70% | 0.36 / 0.59 (0.50-0.67) |
| DistilBERT SST-2 (failure example) | own / tuned | 42% / 54% | 0.30 / 0.47 |

**Which model does what, and why**
- **No model is significantly better than the others** (the intervals overlap; a paired bootstrap of RoBERTa
  against VADER-tuned gives a difference that can be zero). All three are well above "always neutral".
- **RoBERTa gives the label and score** for practical reasons: a real neutral class (53% of our headlines are
  neutral), about 20x faster than BART-MNLI, and it explains its own decisions (occlusion).
- **Each label is explained by the model that gave it** (`sentiment.py explain`). RoBERTa: occlusion
  (remove one word, see how much RoBERTa's own score changes). VADER: its word list values.
  Showing VADER's words next to RoBERTa's label would mix two models and could contradict the label.
- **VADER** is the baseline the course asks for, and the most transparent model (fixed word values).
- **BART-MNLI** is a third, independent opinion: never trained on sentiment, only asked our codebook question.
  Its own labels almost never say "neutral" (1 of 230), so only its tuned score is useful.
- **DistilBERT** is kept as a documented failure: trained on movie reviews, it has no neutral class.

**Our labels: codebook review (12 Nov 2026).** A second pass labelled all 230 headlines against the codebook
without looking at the first labels (agreement 75%, Cohen's kappa 0.52). 30 clear cases were corrected, mostly
war, violence and threats that had been labelled neutral; the first label is kept in `label_before_review` and the
reason in `note`. Borderline cases kept the first label. RoBERTa's accuracy went down (73% -> 71%), so the
corrections did not simply favour the models.

**Known limits**
- Positive news is rare (11 of 150 random headlines) and under-detected by every model; a "positive" label is
  wrong more often than right. Typical failure: good news told with bad words
  ("Rescuers find two more people alive after flash flood disaster"), and positive words in neutral news
  ("Who's Winning the Race for Congress?").
- The 80 extra "positive candidate" headlines were picked by the models, not at random (16 were positive).
- RoBERTa's accuracy differs by outlet (Fox News 55%, NYT 84%, BBC 90%, Guardian 58%; 15-38 headlines each),
  so part of an outlet's "more negative" gap can be model error. Before the label review it seemed to call many
  neutral headlines negative; with the corrected labels its errors go both ways (19 vs 14 of 150).
- Driving words are found by masking one word at a time. Two approaches compared on the 230 labelled headlines
  (`python pipeline\sentiment.py compare-explain` -> `results/sentiment_explain_comparison.csv`): masking finds the
  more important words in 44% of headlines, deleting in 25% (tie 31%); its top word is "nonsense" in 3.5% vs 11%.
  Our own filler list now shows "not", "no", "never", "without" (about 5% of headlines show one of them).
- One annotator plus one codebook review; a second team member should label a sample blind.
- All failures with every model's label and driving words: `failures/sentiment_gold_vs_models.csv`.

## 3. Entities (`ner.py`)
People, organisations, places, groups ("Republicans", "Russian") and events in the headline + lead.
Details of the rules and commands: the top of `ner.py`. The transformer model needs a one-time download:
`python -m spacy download en_core_web_trf` (about 450 MB).

- **Two models compared:** `en_core_web_sm` (small, fast baseline) and `en_core_web_trf` (transformer).
  The small model relies on capital letters, so NYT Title Case headlines give entities such as "Trump Says" (ORG);
  the transformer reads the whole sentence. All 6,140 articles: sm under 1 minute, trf about 5 minutes on our laptops.
- **Name normalisation** (trf output -> `entities.csv`, column `how` says which rule decided):
  clean ("President Trump's" -> "Trump"), our alias list `pipeline/entity_aliases.csv` ("U.S." -> "United States"),
  one type per name, short name -> full name in the same article, surname -> the dominant full name in all articles.
  Result: "Donald Trump" is found in 957 articles; the exact name "Donald Trump" alone appears in only 48.
  Top 100 entities with their variants: `results/ner_top_entities.csv`.
- **Evaluation (needs our labels):** `python pipeline\ner.py sheets` makes two sheets in `data/sample/`:
  - `ner_to_label_types.xlsx`: every name either model found in 100 headlines (from the 150 random sentiment
    headlines). Choose the right type in `my_type` (`no` = not an entity or a wrong span such as "Trump Says",
    `other` = a real entity of a type we do not use). Write names both models missed in `missed`:
    `Keir Starmer (person); Gaza (place)`. Scores precision, recall and F1 for sm and trf.
  - `ner_to_label_same.xlsx`: 150 random mentions (60 people, 50 places, 40 organisations). `real_entity` is
    pre-filled with the name as found; change it to the full real-world name when needed ("Trump" -> "Donald Trump",
    "US" -> "United States"), so that the same thing always has the same text. Scores pairs of mentions
    (precision: mentions we joined are really the same; recall: same mentions we joined) before and after normalisation.
  Save each labelled sheet as CSV (`ner_gold_types.csv`, `ner_gold_same.csv`), then run `evaluate`.

**Known limits**
- "Trump" is always taken to be Donald Trump unless another Trump is named in the same article.
- Spelling variants that are not in the alias list stay separate (we added "Zelensky" = "Zelenskyy").
- Adjectives and countries stay separate on purpose: "Russian" (group) is not "Russia" (place).

## 5. Dashboard (`app/streamlit_app.py`)
`streamlit run app\streamlit_app.py` (opens in the browser; only this computer can see it).
Reads the outputs of steps 1-3; `app/data.py` loads and joins them. Layout from the mockup in our proposal (slide 4):
filters on the left, article cards in the middle (they scroll; click a card to explain it), and on the right "Why these labels?" (the 3 sentiment models,
RoBERTa's driving words highlighted in the headline, VADER's word values, each entity with the rule that named it),
trending entities (click one to open its profile) and sentiment over time. **People & places** (`app/entity_view.py`):
the coverage of one person, place or organisation (articles, tone vs all articles, coverage per week, which outlets
mention it and in which tone, names mentioned together with it, its most negative and positive headlines with a link
to their explanation). A further view compares the outlets (sentiment mix, gap from the
average, most mentioned names). A third tab, "How the labels work" (`app/methods.py`), explains every rule
behind the labels and why we chose it, with the numbers read from `results/`. Tala's first demo (`app.py` in the project folder) stays as it is.
Story cards and the per-story comparison come when step 4 exists.
