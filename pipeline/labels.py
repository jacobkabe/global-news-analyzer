"""
labels.py - hand labelling by the team: make one Excel file per person, then collect the labels.

Commands (from the project folder)
  python pipeline\\labels.py make                         -> data/raw/labelling/labelling_<Name>.xlsx
  python pipeline\\labels.py make --out "<folder>"        the Excel files go to another folder (shared OneDrive)
  python pipeline\\labels.py collect                      reads the filled files from data/raw/labelling/returned/
  python pipeline\\labels.py collect --folder "<folder>"  ... or from another folder (shared OneDrive)
The key (which headline is shared, random or a positive candidate) always stays in data/raw/labelling/key.csv,
so the team cannot see it.

Each file has a "READ ME" sheet (short guide; full guide: docs/labelling-guide.md) and three tasks:
  1 Sentiment     70 headlines: 30 that EVERYONE labels (to measure how much we agree)
                  + 25 new random headlines + 15 headlines a model called positive (more positive examples).
                  The three kinds are mixed and not marked, so nobody is influenced by where a headline comes from.
  2 Entity types  a quarter of data/sample/ner_to_label_types.xlsx (made by ner.py sheets)
  3 Same entity   a quarter of data/sample/ner_to_label_same.xlsx

collect writes
  data/sample/sentiment_team.csv     every sentiment label with its annotator and sample
  results/sentiment_agreement.csv    Cohen's kappa for every pair of annotators on the shared headlines
                                     (and against our reviewed labels in sentiment_gold.csv)
  data/sample/ner_gold_types.csv     input for: python pipeline\\ner.py evaluate
  data/sample/ner_gold_same.csv      input for: python pipeline\\ner.py evaluate

The Excel files contain headlines and summaries, so they stay in data/raw/ (ignored by Git).
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "data" / "raw" / "outputs"
SAMPLE = ROOT / "data" / "sample"
FOLDER = ROOT / "data" / "raw" / "labelling"
RETURNED = FOLDER / "returned"
GOLD = SAMPLE / "sentiment_gold.csv"
CUTOFFS = ROOT / "results" / "sentiment_cutoffs.csv"

TEAM = ["Adriana", "Klaudie", "Matylda", "Sara"]
N_SHARED = 30          # sentiment headlines everyone labels
N_RANDOM = 25          # new random headlines per person
N_CANDIDATES = 15      # new "a model says positive" headlines per person
SENTIMENT_CHOICES = ["negative", "neutral", "positive"]
TYPE_CHOICES = ["person", "organisation", "place", "group", "event", "other", "no"]

READ_ME = [
    "LABELLING GUIDE (short version; the full guide with examples: docs/labelling-guide.md)",
    "",
    "Work alone and do not discuss headlines with the others before everyone is done:",
    "we measure how much we agree, so we need independent answers. About 60-90 minutes in total.",
    "Fill only the yellow columns. Do not change other columns, do not sort or delete rows.",
    "",
    "SHEET 1 - SENTIMENT (column my_label: negative / neutral / positive)",
    "Ask one question: reading ONLY this headline, does it report something bad, something good, or neither?",
    "Judge the event, not the style. Do not open the article, do not think about which outlet wrote it.",
    "NEGATIVE: death, injury, illness, crime, violence, war, attacks, disaster, scandal, crisis, failure, loss;",
    "  also threats and warnings ('warns', 'risks', 'on alert'), and protests WITH violence (clashes, riots, looting).",
    "POSITIVE: a win, breakthrough, recovery, rescue, improvement, record achievement, or a light feel-good story.",
    "NEUTRAL: process news without clear harm or benefit: elections, debates, policy, reports, explainers,",
    "  announcements, business news; protests WITHOUT violence.",
    "STANCE TEST: if supporters and opponents of a side would label it differently, it is neutral",
    "  (an election win, a court decision for one side, a party's plan).",
    "Tricky cases: war news is negative even if one side is winning; a death is negative, also in an obituary;",
    "  sports wins are positive; positive WORDS do not make good news ('Who's Winning the Race for Congress?' = neutral).",
    "Mixed headline? Choose what the headline is mainly about, and write 'x' in 'unsure'.",
    "",
    "SHEET 2 - ENTITY TYPES (column my_type, one row per name a model found in the headline)",
    "person / organisation / place / group / event: the right type of this name (titles and 's are fine:",
    "  'Pope Leo's' = person). The rows under a headline (headline cell empty) belong to that headline.",
    "  place = country, city, region, building; group = nationality, religion or political group used as a",
    "  word ('Russian', 'Republicans', 'Muslim'); organisation = company, agency, party, team, newspaper.",
    "other = a real name of a type we do not use (a law, a product, a book, a ship...).",
    "no = not a name, or the wrong piece of text ('Trump Says', 'Saudi Aramco HQ' instead of 'Saudi Aramco').",
    "missed (first row of each headline): names in the headline that NO row lists, written like",
    "  Keir Starmer (person); Gaza (place)    - also the correct version of a 'no' row.",
    "",
    "SHEET 3 - SAME ENTITY (column real_entity, already filled with the name as found)",
    "Change it to the full real-world name when needed, so the same thing always has the same text:",
    "  'Trump' -> Donald Trump, 'US' -> United States, 'the EU' -> European Union, 'Starmer' -> Keir Starmer.",
    "Use the English Wikipedia name ('Labour Party (UK)'). Use the summary (lead) to see who is meant;",
    "  you may search the web for this sheet.",
    "If it is not a real name at all, write: no",
    "",
    "When done: save the file in the OneDrive folder (same name) and tell Jakub.",
]


# ---- make ---------------------------------------------------------------------------------------------
def cmd_make(a):
    FOLDER.mkdir(parents=True, exist_ok=True)
    target = Path(a.out) if a.out else FOLDER
    target.mkdir(parents=True, exist_ok=True)
    out = [target / f"labelling_{name}.xlsx" for name in TEAM]
    if any(p.exists() for p in out):
        sys.exit(f"{target} already has labelling files; delete or rename them first (they may contain labels)")

    articles = pd.read_csv(OUTPUTS / "articles_clean.csv", keep_default_na=False)
    gold = pd.read_csv(GOLD, keep_default_na=False)
    rng = np.random.default_rng(14112026)

    # shared headlines: from the random headlines we already labelled (so we can compare with them too)
    shared = gold[gold["sample"] == "random"].sample(N_SHARED, random_state=1)["id"].tolist()

    # new headlines: only news, never labelled before
    news = articles[(articles["content_type"] == "news") & ~articles["article_id"].isin(gold["id"])]
    new_random = news.sample(N_RANDOM * len(TEAM), random_state=2)["article_id"].tolist()
    candidates = positive_candidates(news[~news["article_id"].isin(new_random)])
    candidates = candidates.sample(N_CANDIDATES * len(TEAM), random_state=3)["article_id"].tolist()

    headline = articles.set_index("article_id")["headline"]
    ner_types = pd.read_excel(SAMPLE / "ner_to_label_types.xlsx", dtype=str).fillna("")
    ner_same = pd.read_excel(SAMPLE / "ner_to_label_same.xlsx", dtype=str).fillna("")
    type_groups = split_by_headline(ner_types, len(TEAM))
    same_groups = [ner_same.iloc[part] for part in np.array_split(np.arange(len(ner_same)), len(TEAM))]

    key = []
    for i, (name, path) in enumerate(zip(TEAM, out)):
        mine = ([(x, "shared") for x in shared]
                + [(x, "random_2") for x in new_random[i * N_RANDOM:(i + 1) * N_RANDOM]]
                + [(x, "positive_candidates_2") for x in candidates[i * N_CANDIDATES:(i + 1) * N_CANDIDATES]])
        order = rng.permutation(len(mine))
        mine = [mine[j] for j in order]
        sentiment = pd.DataFrame({"row": range(1, len(mine) + 1), "id": [x for x, _ in mine],
                                  "headline": [headline[x] for x, _ in mine],
                                  "my_label": "", "unsure": "", "note": ""})
        key += [{"annotator": name, "id": x, "sample": s} for x, s in mine]
        write_file(path, sentiment, type_groups[i], same_groups[i].reset_index(drop=True))
        print(f"{path.name}: {len(sentiment)} headlines, {type_groups[i]['article_id'].nunique()} entity-type "
              f"headlines, {len(same_groups[i])} same-entity mentions")
    pd.DataFrame(key).to_csv(FOLDER / "key.csv", index=False)
    print(f"\nsaved to {target} (key: {FOLDER / 'key.csv'}); give each person their file and docs/labelling-guide.md")
    print(f'when they are done: python pipeline\\labels.py collect --folder "{target}"')


def positive_candidates(news):
    """Headlines at least one model calls positive (RoBERTa own label, VADER or BART with our cut-offs)."""
    s = pd.read_csv(OUTPUTS / "sentiment.csv", keep_default_na=False)
    cut = pd.read_csv(CUTOFFS).set_index("model")
    positive = set(s.loc[(s["model"] == "roberta") & (s["label"] == "positive"), "article_id"])
    for m in ["vader", "bart_mnli"]:
        part = s[s["model"] == m]
        positive |= set(part.loc[part["score"].astype(float) >= cut.loc[m, "positive_if_score_at_least"], "article_id"])
    return news[news["article_id"].isin(positive)]


def split_by_headline(df, parts):
    """Split the entity-type sheet into parts without cutting a headline's rows apart."""
    df = df.copy()
    df["group"] = (df["headline"] != "").cumsum()          # the headline is written on its first row only
    groups = np.array_split(df["group"].unique(), parts)
    return [df[df["group"].isin(g)].drop(columns="group").reset_index(drop=True) for g in groups]


def write_file(path, sentiment, types, same):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    yellow = PatternFill("solid", fgColor="FFF2A8")
    sheets = {"1 Sentiment": (sentiment, {"row": 6, "id": 8, "headline": 90, "my_label": 14, "unsure": 9, "note": 40},
                              {"my_label": SENTIMENT_CHOICES}, ["my_label", "unsure", "note"]),
              "2 Entity types": (types, {"article_id": 10, "headline": 80, "mention": 28, "my_type": 15, "missed": 45},
                                 {"my_type": TYPE_CHOICES}, ["my_type", "missed"]),
              "3 Same entity": (same, {"article_id": 10, "headline": 60, "lead": 70, "start": 6, "mention": 24,
                                       "real_entity": 30}, {}, ["real_entity"])}
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame({"READ ME": READ_ME}).to_excel(xw, sheet_name="READ ME", index=False)
        xw.sheets["READ ME"].column_dimensions["A"].width = 120
        for name, (df, widths, dropdowns, to_fill) in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "A2"
            for j, col in enumerate(df.columns):
                letter = chr(ord("A") + j)
                ws.column_dimensions[letter].width = widths.get(col, 12)
                if col in to_fill:
                    for r in range(1, len(df) + 2):
                        ws[f"{letter}{r}"].fill = yellow
                if col in dropdowns:
                    dv = DataValidation(type="list", formula1='"' + ",".join(dropdowns[col]) + '"')
                    ws.add_data_validation(dv)
                    dv.add(f"{letter}2:{letter}{len(df) + 1}")
                if col in ("headline", "lead"):
                    for r in range(2, len(df) + 2):
                        ws[f"{letter}{r}"].alignment = Alignment(wrap_text=True, vertical="top")
            for cell in ws[1]:
                cell.font = Font(bold=True)


# ---- collect ------------------------------------------------------------------------------------------
def cmd_collect(a):
    from sklearn.metrics import cohen_kappa_score
    folder = Path(a.folder) if a.folder else RETURNED
    files = sorted(folder.glob("labelling_*.xlsx"))
    if not files:
        sys.exit(f"no labelling_<Name>.xlsx files in {folder}")
    key = pd.read_csv(FOLDER / "key.csv")
    sentiment, types, same, problems = [], [], [], []
    for f in files:
        name = f.stem.removeprefix("labelling_")
        s = pd.read_excel(f, sheet_name="1 Sentiment", dtype=str).fillna("")
        s["my_label"] = s["my_label"].astype(str).str.strip().str.lower()
        bad = s[~s["my_label"].isin(SENTIMENT_CHOICES)]
        if len(bad):
            problems.append(f"{name}: {len(bad)} sentiment rows without a valid label (rows {bad['row'].tolist()[:10]})")
        s["annotator"] = name
        s["id"] = s["id"].astype(int)
        sentiment.append(s[s["my_label"].isin(SENTIMENT_CHOICES)])

        t = pd.read_excel(f, sheet_name="2 Entity types", dtype=str).fillna("")
        t["my_type"] = t["my_type"].astype(str).str.strip().str.lower()
        bad = t[(t["mention"] != "") & ~t["my_type"].isin(TYPE_CHOICES)]
        if len(bad):
            problems.append(f"{name}: {len(bad)} entity-type rows without a valid type")
        types.append(t)

        m = pd.read_excel(f, sheet_name="3 Same entity", dtype=str).fillna("")
        m = m[m["real_entity"].astype(str).str.strip().str.lower() != "no"]   # not a real name: left out
        same.append(m)

    print("\n".join(problems) if problems else "all files complete")
    sentiment = pd.concat(sentiment).merge(key, left_on=["annotator", "id"], right_on=["annotator", "id"], how="left")
    sentiment = sentiment.rename(columns={"my_label": "label"})[
        ["id", "headline", "label", "annotator", "sample", "unsure", "note"]]
    sentiment.to_csv(SAMPLE / "sentiment_team.csv", index=False)

    # agreement on the shared headlines: every pair of annotators, and each one against our reviewed labels
    shared = sentiment[sentiment["sample"] == "shared"].pivot_table(index="id", columns="annotator",
                                                                   values="label", aggfunc="first")
    gold = pd.read_csv(GOLD, keep_default_na=False).set_index("id")["my_label"]
    shared["reviewed gold"] = shared.index.map(gold)
    rows = []
    for x, y in combinations(shared.columns, 2):
        both = shared[[x, y]].dropna()
        rows.append({"annotator_1": x, "annotator_2": y, "n": len(both),
                     "agreement": round((both[x] == both[y]).mean(), 3),
                     "cohen_kappa": round(cohen_kappa_score(both[x], both[y]), 3)})
    agreement = pd.DataFrame(rows)
    agreement.to_csv(ROOT / "results" / "sentiment_agreement.csv", index=False)
    people = agreement[(agreement["annotator_1"] != "reviewed gold") & (agreement["annotator_2"] != "reviewed gold")]
    print(f"\nsentiment: {len(sentiment)} labels; mean kappa between team members "
          f"{people['cohen_kappa'].mean():.2f} on {len(shared)} shared headlines")
    print(agreement.to_string(index=False))

    pd.concat(types).to_csv(SAMPLE / "ner_gold_types.csv", index=False)
    pd.concat(same).to_csv(SAMPLE / "ner_gold_same.csv", index=False)
    print(f"\nsaved: data/sample/sentiment_team.csv, results/sentiment_agreement.csv, "
          f"data/sample/ner_gold_types.csv, data/sample/ner_gold_same.csv")
    print("next: python pipeline\\ner.py evaluate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_make = sub.add_parser("make", help="make one labelling file per team member")
    p_make.add_argument("--out", help="folder for the Excel files (default: data/raw/labelling)")
    p_make.set_defaults(fn=cmd_make)
    p_collect = sub.add_parser("collect", help="collect the filled files")
    p_collect.add_argument("--folder", help="folder with the filled files (default: data/raw/labelling/returned)")
    p_collect.set_defaults(fn=cmd_collect)
    args = parser.parse_args()
    args.fn(args)
