# Screenshots

The dashboard (`app/streamlit_app.py`) on our data up to 15 Nov 2026 (10,259 articles, "Only news" on).

| File | View | What it shows |
|---|---|---|
| `1-articles.png` | Articles | Article cards; on the right "Why these labels?" for the selected headline (RoBERTa's probabilities, the 3 models, driving words, entities, limits), trending entities and the sentiment share per week |
| `2-people-and-places.png` | People & places | The profile of one name (here Donald Trump): a summary sentence built from the numbers, key numbers, coverage per week, names mentioned together, its most negative and most positive headlines |
| `3-outlets-compared.png` | Outlets compared | Sentiment mix per outlet with its dates; outlets with few articles set apart; each outlet compared with the Guardian on the same days |
| `4-how-the-labels-work.png` | How the labels work | The pipeline diagram and the sources |
| `5-evaluation.png` | How the labels work | The three sentiment models, their scores with the tuned cut-offs, and the evaluation on our hand labels |
