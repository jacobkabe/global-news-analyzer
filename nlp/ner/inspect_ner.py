#!/usr/bin/env python3
"""
inspect_ner.py - look at the results of ner.py.

Put it next to ner.py. It reads <output-dir>/ner.csv and ner_runs.csv.

Commands
  summary      size, coverage, entities per article, label counts
  top          most frequent entities per label (by number of articles)
  suspicious   entities that look wrong (too short, lowercase names, digits in names, ...)
  conflicts    the same text tagged with different labels (e.g. "Washington" GPE vs PERSON)
  sample       random entities shown in context from the original article (also verifies offsets)
  compare      agreement between two models on the same articles

Examples
  python inspect_ner.py summary
  python inspect_ner.py top --label ORG --n 25
  python inspect_ner.py suspicious --input-type headline_lead
  python inspect_ner.py sample --label PERSON --n 15
  python inspect_ner.py compare en_core_web_sm en_core_web_trf
"""
import argparse
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]   # nlp/ner/inspect_ner.py -> repo root
DEFAULT_INPUT = os.environ.get("NEWS_CSV", str(ROOT / "data" / "news.csv"))
DEFAULT_OUTPUT_DIR = os.environ.get("NER_OUTPUT_DIR", str(ROOT / "data" / "raw" / "outputs"))
NAME_LABELS = {"PERSON", "ORG", "GPE", "NORP", "LOC", "FAC"}


def load(a, runs=False):
    name = "ner_runs.csv" if runs else "ner.csv"
    p = Path(a.output_dir) / name
    if not p.is_file():
        sys.exit(f"{p} not found. Run `python ner.py run` first (or pass --output-dir).")
    df = pd.read_csv(p, dtype={"text": str, "label": str}, keep_default_na=False)
    for col in ("model", "input_type", "source"):
        val = getattr(a, col, None)
        if val:
            df = df[df[col] == val]
    return df


# ---- summary ------------------------------------------------------------------------------------
def cmd_summary(a):
    ner, runs = load(a), load(a, runs=True)
    if runs.empty:
        sys.exit("No rows match these filters.")
    print(f"{len(ner)} entities, {runs['article_id'].nunique()} articles\n")

    g = runs.groupby(["input_type", "model", "source"]).agg(
        articles=("article_id", "count"),
        zero_ent=("n_entities", lambda s: (s == 0).mean() * 100),
        ents_per_article=("n_entities", "mean"),
        median_chars=("n_chars", "median"),
        sec_per_article=("seconds", "mean"),
    )
    g["zero_ent"] = g["zero_ent"].round(1)
    g = g.rename(columns={"zero_ent": "%_no_entities"})
    print("== coverage and speed ==")
    print(g.round(2).to_string())

    print("\n== labels ==")
    lab = ner["label"].value_counts()
    print(pd.DataFrame({"count": lab, "share_%": (100 * lab / lab.sum()).round(1)}).to_string())

    dup = ner.duplicated(["article_id", "model", "input_type", "start_char", "end_char"]).sum()
    print(f"\nduplicate spans (same article/model/input/offsets): {dup}  (should be 0)")


# ---- top ----------------------------------------------------------------------------------------
def cmd_top(a):
    d = load(a)
    labels = [a.label] if a.label else ["PERSON", "ORG", "GPE", "NORP", "LOC"]
    for lab in labels:
        top = (d[d["label"] == lab].groupby("text")["article_id"].nunique()
               .sort_values(ascending=False).head(a.n))
        print(f"\n== {lab} (articles) ==")
        for t, n in top.items():
            print(f"  {n:5}  {t}")


# ---- suspicious ---------------------------------------------------------------------------------
def reasons(text, label):
    r = []
    if len(text) <= 2:
        r.append("very short")
    if len(text) > 60:
        r.append("very long")
    if label in NAME_LABELS and text.islower():
        r.append("all lowercase")
    if label in {"PERSON", "GPE", "NORP", "LOC"} and re.search(r"\d", text):
        r.append("digit in name")
    if re.search(r"[|<>{}\[\]\\@#]", text):
        r.append("odd character")
    if re.match(r"^[^\w]|[^\w.)'’\"”]$", text):
        r.append("starts/ends with punctuation")
    if re.search(r"(’s|'s)$", text):
        r.append("possessive attached")
    if re.match(r"(?i)^(the|a|an)\s", text) and label == "PERSON":
        r.append("article in PERSON")
    if label == "PERSON" and " " not in text and text.isupper() and len(text) > 1:
        r.append("ALLCAPS PERSON (headline?)")
    return r


def cmd_suspicious(a):
    d = load(a)
    c = d.groupby(["text", "label"]).agg(n=("article_id", "count")).reset_index()
    c["why"] = [", ".join(reasons(t, l)) for t, l in zip(c["text"], c["label"])]
    bad = c[c["why"] != ""].sort_values("n", ascending=False)
    print(f"{len(bad)} distinct suspicious (text, label) pairs, "
          f"{int(bad['n'].sum())} of {len(d)} entity rows ({100 * bad['n'].sum() / max(len(d), 1):.1f}%)\n")
    for why, grp in bad.groupby("why"):
        print(f"-- {why} ({len(grp)})")
        for _, r in grp.head(a.n).iterrows():
            print(f"   {r['n']:4}  {r['label']:8} {r['text']!r}")


# ---- conflicts ----------------------------------------------------------------------------------
def cmd_conflicts(a):
    d = load(a).copy()
    d["key"] = d["text"].str.lower()
    per = d.groupby(["key", "label"]).size().unstack(fill_value=0)
    multi = per[(per > 0).sum(axis=1) > 1]
    multi = multi.assign(total=multi.sum(axis=1)).sort_values("total", ascending=False)
    print(f"{len(multi)} texts have more than one label\n")
    for key, row in multi.head(a.n).iterrows():
        parts = ", ".join(f"{l}={int(v)}" for l, v in row.drop("total").items() if v)
        print(f"  {key!r:35} {parts}")


# ---- sample (with context) ----------------------------------------------------------------------
def cmd_sample(a):
    try:
        import ner as nerlib  # reuse build_input so offsets match exactly
    except ImportError:
        sys.exit("Run this from the folder that contains ner.py.")
    d = load(a)
    if a.label:
        d = d[d["label"] == a.label]
    if d.empty:
        sys.exit("No entities match these filters.")
    arts = nerlib.load_articles(a.input).set_index("id")
    rows = d.sample(min(a.n, len(d)), random_state=a.seed)

    bad = 0
    for _, e in rows.iterrows():
        aid = int(e["article_id"])
        if aid not in arts.index:
            continue
        txt = nerlib.build_input(arts.loc[aid], e["input_type"]) or ""
        s, t = int(e["start_char"]), int(e["end_char"])
        ok = txt[s:t] == e["text"]
        bad += not ok
        left, right = txt[max(0, s - a.width):s], txt[t:t + a.width]
        flag = "" if ok else "   <-- OFFSET MISMATCH"
        print(f"[{e['label']}] {e['source']}/{aid}{flag}")
        print(f"   ...{left}[[{e['text']}]]{right}...\n")
    print(f"offset mismatches: {bad} of {len(rows)} (should be 0; otherwise news.csv changed since the run)")


# ---- compare ------------------------------------------------------------------------------------
def cmd_compare(a):
    d = load(a)
    m1, m2 = a.models
    d = d[d["model"].isin([m1, m2])]
    ids = set(d[d["model"] == m1]["article_id"]) & set(d[d["model"] == m2]["article_id"])
    d = d[d["article_id"].isin(ids)]
    if not ids:
        sys.exit("No articles were processed by both models (with the same input type).")

    def spans(m):
        x = d[d["model"] == m]
        return set(zip(x["article_id"], x["start_char"], x["end_char"], x["label"])), \
               dict(((r.article_id, r.start_char, r.end_char, r.label), r.text) for r in x.itertuples())

    s1, t1 = spans(m1)
    s2, t2 = spans(m2)
    both = s1 & s2
    print(f"{len(ids)} articles in common")
    print(f"{m1}: {len(s1)} entities | {m2}: {len(s2)} entities | exact match: {len(both)}")
    print(f"agreement (F1-style): {2 * len(both) / max(len(s1) + len(s2), 1):.3f}\n")
    rng = random.Random(a.seed)
    for name, only, tx in ((m1, s1 - s2, t1), (m2, s2 - s1, t2)):
        only = sorted(only)
        print(f"-- only in {name} ({len(only)}), sample")
        for k in rng.sample(only, min(a.n, len(only))):
            print(f"   article {k[0]:<7} {k[3]:8} {tx[k]!r}")
        print()


# ---- cli ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    common.add_argument("--input", default=DEFAULT_INPUT, help="news CSV (only needed for `sample`)")
    common.add_argument("--model")
    common.add_argument("--input-type")
    common.add_argument("--source")
    common.add_argument("--seed", type=int, default=1)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("summary", parents=[common]).set_defaults(fn=cmd_summary)

    p = sub.add_parser("top", parents=[common])
    p.add_argument("--label")
    p.add_argument("--n", type=int, default=15)
    p.set_defaults(fn=cmd_top)

    p = sub.add_parser("suspicious", parents=[common])
    p.add_argument("--n", type=int, default=10, help="examples per reason")
    p.set_defaults(fn=cmd_suspicious)

    p = sub.add_parser("conflicts", parents=[common])
    p.add_argument("--n", type=int, default=25)
    p.set_defaults(fn=cmd_conflicts)

    p = sub.add_parser("sample", parents=[common])
    p.add_argument("--label")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--width", type=int, default=70, help="characters of context on each side")
    p.set_defaults(fn=cmd_sample)

    p = sub.add_parser("compare", parents=[common])
    p.add_argument("models", nargs=2)
    p.add_argument("--n", type=int, default=15)
    p.set_defaults(fn=cmd_compare)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()