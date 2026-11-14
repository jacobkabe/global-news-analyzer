# Failure examples

Concrete cases where a model or a method goes wrong, as the course asks. Summary of all findings: `docs/findings.md`.

| File / case | What it shows |
|---|---|
| `sentiment_gold_vs_models.csv` | Every hand-labelled headline with the label, score and driving words of every model; the headlines most models get wrong first. |
| Good news with bad words | "Rescuers in Nepal find two more people alive after flash flood disaster": called negative by the models, the news is the rescue (positive). |
| Positive words, neutral news | "Who's Winning the Race for Congress?": VADER and others say positive, it is election news (neutral). Only 13 of RoBERTa's 32 positive labels are right. |
| BART-MNLI and zero | BART-MNLI puts almost every headline near −1; its own labels almost never say neutral (45% accuracy), so it needs our tuned cut-off. |
| DistilBERT SST-2 | Trained on movie reviews, it has no neutral class: 42% accuracy. |
| Accuracy by outlet | RoBERTa: BBC 90%, NYT 84%, Guardian 58%, Fox News 55% (`results/sentiment_by_outlet.csv`): an outlet can look more negative partly because of model errors. |
| Deleting a word breaks the sentence | "US workers facing deadly heat borrow from tomato pickers to win key protections": deleting "heat" gives "facing deadly borrow from…", so "heat" looked positive (+0.33) and "deadly" (−0.18) was not shown. Fixed by masking the word instead: *win* +0.36, *deadly* −0.20. |
| Stop words hide meaning | scikit-learn's stop-word list contains "not", "no", "never", "without", "fire": "No deaths after…" lost its key word. Fixed with our own filler list. |
| "Trump Says" | The small spaCy model takes NYT Title Case words as names: "Trump Says" (organisation). |
| `ner_errors.csv` (after the team labels) | Entity mistakes of both spaCy models and of the name rules. |
