# Archive: earlier versions

Kept so the history of the project can be followed. **Nothing here is used by the pipeline or the dashboard**,
and the scripts are not maintained (their import paths still say `nlp.`, so they no longer run from here).

| What | Why it was replaced |
|---|---|
| `nlp-first-version/` | First NLP version (Oct–Nov 2026): 5 sentiment models (DistilBERT, SiEBERT, RoBERTa, TextBlob, VADER) and a spaCy/BERT NER test, without hand labels. Replaced by `pipeline/`, which is evaluated on hand labels. `sentiment/make_label_sheet.py` and `make_label_sheet_extra.py` picked the 230 hand-labelled headlines in `data/sample/sentiment_gold.csv`. |
| `results-first-version/` | Outputs of that version: label distributions and agreement between the 5 models (no accuracy, because there were no hand labels yet). SiEBERT and TextBlob were dropped: SiEBERT has no neutral class, TextBlob called 27% of the headlines positive. |
| `solution-idea-oct-2026.md` | The plan from 17–18 Oct 2026. What was built and what was dropped is described in `README.md` and `docs/findings.md`. |
