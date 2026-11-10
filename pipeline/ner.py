"""
ner.py - step 3 of the pipeline: named entities in headline + lead, with name normalisation,
so that "Trump", "Donald Trump", "President Trump's" and "D.J. Trump" count as one entity.

Commands
  run         find entities with two spaCy models -> data/raw/outputs/entities_raw.csv
  normalise   merge name variants (trf model) -> data/raw/outputs/entities.csv, results/ner_top_entities.csv
  sheets      make two labelling sheets in data/sample/ (needs run and normalise first)
  evaluate    compare with our labels -> results/ner_metrics.csv, failures/ner_*.csv

Examples (from the project folder)
  python pipeline\ner.py run                  both models, all articles (sm under 1 minute, trf about 5 minutes)
  python pipeline\ner.py run --models sm --limit 100
  python pipeline\ner.py normalise
  python pipeline\ner.py sheets
  python pipeline\ner.py evaluate

Models (spaCy, compared on our labels)
  sm    en_core_web_sm: small and fast (baseline). Relies on capital letters, so NYT Title Case headlines
        ("Trump Says Economy Has ...") give entities such as "Trump Says" (ORG).
  trf   en_core_web_trf: a RoBERTa transformer, slower but reads the whole sentence.

Entity types we keep (spaCy label -> our type)
  PERSON -> person, ORG -> organisation, GPE/LOC/FAC -> place, NORP -> group ("Republicans", "Russian"),
  EVENT -> event. Dates, numbers, works of art etc. are not used by the dashboard.

How normalise works (the rule that changed each name is saved in the column "how")
  1. clean     remove "the", possessive 's, titles (President, Mr, Sen. ...), single initials of people;
               compare names without dots or capitals ("U.S." = "US")
  2. alias     our own list, pipeline/entity_aliases.csv ("US" -> "United States"); this rule wins over 3-4.
               An empty entity drops the name (e.g. "AI", which spaCy often calls a place).
  3. type      the same name can get different labels ("Trump" as PERSON and as ORG): the most common wins
  4. article   a short person name -> the longer one in the same article ("Trump" -> "Donald Trump"),
               only if exactly one longer name fits
  5. corpus    a surname alone -> the most common full name with that surname in all articles,
               only if that full name has at least 80% of them ("Trump" -> "Donald Trump", not "Melania Trump")
"""
import argparse
import re
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "outputs"
ARTICLES = OUT / "articles_clean.csv"
RAW = OUT / "entities_raw.csv"
ENTITIES = OUT / "entities.csv"
ALIASES = ROOT / "pipeline" / "entity_aliases.csv"
SAMPLE = ROOT / "data" / "sample"
SENTIMENT_GOLD = SAMPLE / "sentiment_gold.csv"
RESULTS = ROOT / "results"
FAILURES = ROOT / "failures"

MODELS = {"sm": "en_core_web_sm", "trf": "en_core_web_trf"}
MAIN_MODEL = "trf"
TYPES = {"PERSON": "person", "ORG": "organisation", "GPE": "place", "LOC": "place", "FAC": "place",
         "NORP": "group", "EVENT": "event"}
RAW_COLS = ["article_id", "model", "part", "start", "end", "text", "label"]
SAVE_EVERY = 512    # articles
MIN_SHARE = 0.8     # rule 5: the full name needs this share of all full names with the same surname
MIN_ARTICLES = 3    # rule 5: ... and must appear in at least this many articles

TITLES = re.compile(r"^((former|ex-|then-|us|u\.s\.|vice)\s*)?(president|vice-president|prime minister|pm|"
                    r"mr|mrs|ms|dr|sen|senator|rep|representative|gov|governor|king|queen|prince|princess|"
                    r"pope|sir|dame|lord|lady|judge|justice|secretary|chancellor|minister|mayor|general|gen)"
                    r"\.?\s+", re.I)

N_TYPE_SHEET = 100   # headlines in the entity-type sheet (taken from the 150 random sentiment headlines)
N_SAME_SHEET = {"person": 60, "place": 50, "organisation": 40}   # mentions in the same-entity sheet
GOLD_TYPES = SAMPLE / "ner_gold_types.csv"
GOLD_SAME = SAMPLE / "ner_gold_same.csv"
SHEET_TYPES = SAMPLE / "ner_to_label_types.xlsx"
SHEET_SAME = SAMPLE / "ner_to_label_same.xlsx"
TYPE_CHOICES = ["person", "organisation", "place", "group", "event", "other", "no"]


def load_articles():
    if not ARTICLES.exists():
        sys.exit(f"{ARTICLES} not found. Run: python pipeline\\clean.py")
    return pd.read_csv(ARTICLES, keep_default_na=False)


def model_text(row):
    """What the model reads: headline, then the lead. Entity positions refer to this string."""
    h, lead = row["headline"], row["lead"]
    if not lead:
        return h
    return f"{h} {lead}" if h[-1:] in ".!?:;" else f"{h}. {lead}"


# ---- run ------------------------------------------------------------------------------------------
def cmd_run(a):
    import spacy
    df = load_articles()
    old = pd.read_csv(RAW, keep_default_na=False) if RAW.exists() else pd.DataFrame(columns=RAW_COLS)
    rows = old.to_dict("records")
    for m in a.models:
        done = set(old.loc[old["model"] == m, "article_id"])
        todo = df[~df["article_id"].isin(done)]
        if a.limit:
            todo = todo.head(a.limit)
        print(f"{m}: {len(todo)} articles to process ({len(done)} already done)")
        if todo.empty:
            continue
        nlp = spacy.load(MODELS[m], disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
        t0 = time.time()
        texts = [model_text(r) for _, r in todo.iterrows()]
        for i, (aid, head, doc) in enumerate(zip(todo["article_id"], todo["headline"],
                                                 nlp.pipe(texts, batch_size=32)), 1):
            ents = [e for e in doc.ents if e.label_ in TYPES]
            for e in ents:
                rows.append(dict(article_id=aid, model=m, part="headline" if e.start_char < len(head) else "lead",
                                 start=e.start_char, end=e.end_char, text=e.text, label=e.label_))
            if not ents:   # marker row, so the article is not processed again
                rows.append(dict(article_id=aid, model=m, part="", start=-1, end=-1, text="", label=""))
            if i % SAVE_EVERY == 0:
                pd.DataFrame(rows, columns=RAW_COLS).to_csv(RAW, index=False)
                print(f"  {i}/{len(todo)}  ({time.time() - t0:.0f} s)", end="\r")
        pd.DataFrame(rows, columns=RAW_COLS).to_csv(RAW, index=False)
        print(f"  {len(todo)}/{len(todo)}  ({time.time() - t0:.0f} s)")
    print(f"saved to {RAW}")


def read_raw(model):
    if not RAW.exists():
        sys.exit(f"{RAW} not found. Run: python pipeline\\ner.py run")
    raw = pd.read_csv(RAW, keep_default_na=False)
    return raw[(raw["model"] == model) & (raw["start"] != -1)].copy()


# ---- normalise ------------------------------------------------------------------------------------
def clean_name(text, label):
    """'the U.S.' -> 'U.S.', 'President Trump's' -> 'Trump', 'D. J. Trump' -> 'Trump'"""
    s = re.sub(r"\s+", " ", text.replace("’", "'")).strip(" ,;:\"'-")
    s = re.sub(r"'s?$", "", s)
    s = re.sub(r"^the\s+", "", s, flags=re.I)
    if label == "PERSON":
        s = TITLES.sub("", s)
        words = [w for w in s.split() if len(w.strip(".")) > 1]   # drop initials such as "J."
        s = " ".join(words) if words else s
    return s.strip(" ,;:\"'-")


def key_of(name):
    """Names are compared without dots and capitals: 'U.S.' and 'US' -> 'us'."""
    return re.sub(r"\s+", " ", name.replace(".", "")).strip().casefold()


def read_aliases():
    al = pd.read_csv(ALIASES, keep_default_na=False, comment="#")
    return {key_of(r["alias"]): r["entity"] for _, r in al.iterrows()}


def normalise(raw):
    """raw entity rows -> same rows with name, entity, type and how (the rule that decided the entity)."""
    df = raw.copy()
    df["name"] = [clean_name(t, l) for t, l in zip(df["text"], df["label"])]
    df = df[df["name"].str.len() > 1].copy()
    df["key"] = df["name"].map(key_of)
    df["how"] = "same"

    # 2. our alias list
    aliases = read_aliases()
    is_alias = df["key"].isin(aliases)
    df.loc[is_alias, "how"] = "alias"
    df["entity_key"] = [key_of(aliases[k]) if a else k for k, a in zip(df["key"], is_alias)]
    df = df[df["entity_key"] != ""].copy()               # an empty alias drops the name

    # 3. one type per name: the most common label, counted by articles
    df["type"] = df["label"].map(TYPES)
    df["type"] = df["entity_key"].map(majority_type(df))

    # 4. within one article: short person name -> the one longer name that ends or starts with it
    people = df[(df["type"] == "person") & (df["how"] != "alias")]
    article_map = {}
    for aid, keys in people.groupby("article_id")["entity_key"]:
        keys = set(keys)
        for k in keys:
            longer = [x for x in keys if x != k and (x.endswith(" " + k) or x.startswith(k + " "))]
            if len(longer) == 1:
                article_map[(aid, k)] = longer[0]
    hit = [(aid, k) in article_map for aid, k in zip(df["article_id"], df["entity_key"])]
    df.loc[hit, "entity_key"] = [article_map[(aid, k)] for aid, k in zip(df.loc[hit, "article_id"], df.loc[hit, "entity_key"])]
    df.loc[hit, "how"] = "article"

    # 5. all articles: a surname alone -> the dominant full name with that surname
    corpus_map = surname_map(df[df["type"] == "person"])
    hit = (df["type"] == "person") & (df["how"] == "same") & df["entity_key"].isin(corpus_map)
    df.loc[hit, "entity_key"] = df.loc[hit, "entity_key"].map(corpus_map)
    df.loc[hit, "how"] = "corpus"

    df["entity"] = df["entity_key"].map(display_names(df, aliases))
    return df


def majority_type(df):
    counts = df.groupby(["entity_key", "type"])["article_id"].nunique().reset_index()
    counts = counts.sort_values(["entity_key", "article_id"], ascending=[True, False])
    return counts.drop_duplicates("entity_key").set_index("entity_key")["type"]


def surname_map(people):
    """{'trump': 'donald trump', ...} for one-word names whose full name is clear."""
    n_articles = people.groupby("entity_key")["article_id"].nunique()
    full = n_articles[n_articles.index.str.contains(" ")]
    by_surname = defaultdict(list)
    for k, n in full.items():
        by_surname[k.split()[-1]].append((n, k))
    out = {}
    for surname, names in by_surname.items():
        n, best = max(names)
        if surname in n_articles.index and n >= MIN_ARTICLES and n / sum(x for x, _ in names) >= MIN_SHARE:
            out[surname] = best
    return out


def display_names(df, aliases):
    """The name shown on the dashboard: the alias entity, or the most common spelling."""
    names = {key_of(v): v for v in aliases.values() if v}
    common = df.groupby("key")["name"].agg(lambda s: s.value_counts().index[0])
    return {k: names.get(k, common.get(k, k)) for k in df["entity_key"].unique()}


def cmd_normalise(a):
    df = normalise(read_raw(a.model))
    cols = ["article_id", "part", "start", "end", "text", "label", "entity", "type", "how"]
    df[cols].to_csv(ENTITIES, index=False)
    print(f"{len(df)} mentions, {df['entity'].nunique()} entities (before: {df['key'].nunique()} names)")
    print("rule that decided the entity:", df["how"].value_counts().to_dict())

    # top entities, with how many articles the exact name alone would find
    n_after = df.groupby(["entity", "type"])["article_id"].nunique()
    top = n_after.sort_values(ascending=False).head(100).reset_index(name="articles")
    exact = df[df["key"] == df["entity_key"]].groupby("entity")["article_id"].nunique()
    variants = df.groupby("entity")["name"].agg(lambda s: "; ".join(s.value_counts().index[:6]))
    top["articles_exact_name"] = top["entity"].map(exact).fillna(0).astype(int)
    top["variants"] = top["entity"].map(variants)
    RESULTS.mkdir(exist_ok=True)
    top.to_csv(RESULTS / "ner_top_entities.csv", index=False)
    print(top.head(15).to_string(index=False))
    print(f"saved to {ENTITIES} and {RESULTS / 'ner_top_entities.csv'}")


# ---- labelling sheets -----------------------------------------------------------------------------
def cmd_sheets(a):
    for p in (SHEET_TYPES, SHEET_SAME):
        if p.exists():
            sys.exit(f"{p} already exists; delete or rename it first (it may contain your labels)")
    df = load_articles().set_index("article_id")
    gold = pd.read_csv(SENTIMENT_GOLD)
    ids = gold.loc[gold["sample"] == "random", "id"].sample(N_TYPE_SHEET, random_state=42).tolist()

    # sheet 1: every name either model found in these headlines; you give the right type
    raw = pd.read_csv(RAW, keep_default_na=False)
    raw = raw[raw["article_id"].isin(ids) & (raw["part"] == "headline")]
    rows = []
    for aid in ids:
        found = sorted(set(raw.loc[raw["article_id"] == aid, "text"]))
        for i, t in enumerate(found or [""]):
            rows.append(dict(article_id=aid, headline=df.at[aid, "headline"] if i == 0 else "",
                             mention=t, my_type="", missed=""))
    write_sheet(pd.DataFrame(rows), SHEET_TYPES, "my_type")

    # sheet 2: random mentions from all articles; you write the real-world entity
    ent = pd.read_csv(ENTITIES, keep_default_na=False)
    parts = [ent[ent["type"] == t].sample(n, random_state=42) for t, n in N_SAME_SHEET.items()]
    s = pd.concat(parts).sample(frac=1, random_state=42)
    s = pd.DataFrame(dict(article_id=s["article_id"], headline=s["article_id"].map(df["headline"]),
                          lead=s["article_id"].map(df["lead"]), start=s["start"], mention=s["text"],
                          real_entity=s["text"]))
    write_sheet(s, SHEET_SAME, None)
    print(f"wrote {SHEET_TYPES.name} ({len(rows)} rows) and {SHEET_SAME.name} ({len(s)} rows) to {SAMPLE}")


def write_sheet(df, path, dropdown_col):
    from openpyxl.worksheet.datavalidation import DataValidation
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        df.to_excel(xw, index=False)
        ws = xw.sheets["Sheet1"]
        for col, width in zip("ABCDEFG", [10, 70, 30, 16, 40, 30, 30]):
            ws.column_dimensions[col].width = width
        if dropdown_col:
            letter = "ABCDEFG"[list(df.columns).index(dropdown_col)]
            dv = DataValidation(type="list", formula1='"' + ",".join(TYPE_CHOICES) + '"')
            ws.add_data_validation(dv)
            dv.add(f"{letter}2:{letter}{len(df) + 1}")


# ---- evaluate -------------------------------------------------------------------------------------
def read_gold(path):
    """The labelled sheet, saved as CSV (the .xlsx sheet itself is pre-filled, so it is not used)."""
    return pd.read_csv(path, keep_default_na=False) if path.exists() else None


def gold_entities(g):
    """Labelled sheet -> {(article_id, name key, type)}; 'missed' is written as 'Name (type); Name (type)'."""
    g = g.copy()
    g["article_id"] = g["article_id"].astype(int)
    out = set()
    for _, r in g.iterrows():
        t = str(r["my_type"]).strip().lower()
        if r["mention"] and t in TYPE_CHOICES[:5]:
            out.add((r["article_id"], key_of(str(r["mention"])), t))
        for name, t in re.findall(r"([^;()]+)\(([a-z]+)\)", str(r["missed"]).lower()):
            out.add((r["article_id"], key_of(name.strip()), t.strip()))
    return out


def prf(pred, gold):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return dict(precision=round(p, 3), recall=round(r, 3), f1=round(2 * p * r / (p + r), 3) if p + r else 0.0,
                n_pred=len(pred), n_gold=len(gold))


def pairs(ids, labels):
    """All pairs of mentions that a method puts in the same entity."""
    groups = defaultdict(list)
    for i, l in zip(ids, labels):
        groups[l].append(i)
    return {p for g in groups.values() for p in combinations(sorted(g), 2)}


def cmd_evaluate(a):
    metrics, failures = [], []
    g = read_gold(GOLD_TYPES)
    if g is None or not (g["my_type"].astype(str).str.strip() != "").any():
        print(f"no entity-type labels yet ({GOLD_TYPES.name}); skipping the model comparison")
    else:
        gold = gold_entities(g)
        ids = {a for a, _, _ in gold} | set(g["article_id"].astype(int))
        raw = pd.read_csv(RAW, keep_default_na=False)
        raw = raw[raw["article_id"].isin(ids) & (raw["part"] == "headline")]
        gold_spans = {(a, k) for a, k, _ in gold}
        for m in MODELS:
            r = raw[raw["model"] == m]
            pred = {(aid, key_of(t), TYPES[l]) for aid, t, l in zip(r["article_id"], r["text"], r["label"])}
            metrics.append(dict(task="entities in headlines", method=m, match="name and type", **prf(pred, gold)))
            metrics.append(dict(task="entities in headlines", method=m, match="name only",
                                **prf({(x, k) for x, k, _ in pred}, gold_spans)))
            for x in sorted(pred - gold):
                failures.append(dict(model=m, error="wrong (not an entity, wrong span or type)", article_id=x[0], name=x[1], type=x[2]))
            for x in sorted(gold - pred):
                failures.append(dict(model=m, error="missed", article_id=x[0], name=x[1], type=x[2]))

    g = read_gold(GOLD_SAME)
    if g is None:
        print(f"no same-entity labels yet ({GOLD_SAME.name}); skipping the normalisation check")
    else:
        g["article_id"] = g["article_id"].astype(int)
        norm = normalise(read_raw(MAIN_MODEL)).set_index(["article_id", "start"])
        g = g[[k in norm.index for k in zip(g["article_id"], g["start"])]].reset_index(drop=True)
        found = norm.loc[list(zip(g["article_id"], g["start"]))]
        g["before"] = [key_of(t) for t in g["mention"]]
        g["cleaned"] = found["key"].to_numpy()
        g["after"] = found["entity_key"].to_numpy()
        g["how"] = found["how"].to_numpy()
        g["gold"] = g["real_entity"].map(key_of)
        ids = list(range(len(g)))
        gold_pairs = pairs(ids, g["gold"])
        for method in ["before", "cleaned", "after"]:
            metrics.append(dict(task="same entity (pairs of mentions)", method=method, match="",
                                **prf(pairs(ids, g[method]), gold_pairs),
                                n_entities=g[method].nunique(), n_entities_gold=g["gold"].nunique()))
        # a mention is wrong if it is grouped with other mentions than in our labels
        for _, r in g.iterrows():
            if set(g.index[g["after"] == r["after"]]) == set(g.index[g["gold"] == r["gold"]]):
                continue
            failures.append(dict(model=MAIN_MODEL, error=f"normalised to '{r['after']}' (rule: {r['how']})",
                                 article_id=r["article_id"], name=r["mention"], type=f"real: {r['real_entity']}"))

    if not metrics:
        sys.exit("nothing to evaluate yet: label the sheets first (python pipeline\\ner.py sheets)")
    RESULTS.mkdir(exist_ok=True)
    FAILURES.mkdir(exist_ok=True)
    out = pd.DataFrame(metrics)
    out.to_csv(RESULTS / "ner_metrics.csv", index=False)
    pd.DataFrame(failures).to_csv(FAILURES / "ner_errors.csv", index=False)
    print(out.to_string(index=False))
    print(f"saved to {RESULTS / 'ner_metrics.csv'} and {FAILURES / 'ner_errors.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="find entities")
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--limit", type=int, help="only the first N articles (for tests)")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("normalise", help="merge name variants")
    p.add_argument("--model", choices=list(MODELS), default=MAIN_MODEL)
    p.set_defaults(fn=cmd_normalise)
    sub.add_parser("sheets", help="make the labelling sheets").set_defaults(fn=cmd_sheets)
    sub.add_parser("evaluate", help="compare with our labels").set_defaults(fn=cmd_evaluate)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
