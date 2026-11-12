#!/usr/bin/env python3
"""
ner.py - spaCy named-entity recognition for the news analyzer.

Reads a news CSV (columns: id, source, url, title, summary, text, ...), runs a spaCy model and
writes one row per entity to CSV. Resumable: re-running skips articles that are already done.

Commands
  run             extract entities
  stats           coverage, speed and most frequent entities
  export-sample   stratified JSONL for hand-annotating a gold set (no predictions included)

Examples
  python ner.py run --inputs headline_lead --limit 100            # quick test, all sources
  python ner.py run --inputs fulltext                              # whole articles (rows that have text)
  python ner.py run --inputs headline_lead fulltext --model en_core_web_trf
  python ner.py stats --input-type fulltext
  python ner.py export-sample --n 100 --input-type headline_lead

Input types (entity offsets refer to the string build_input() returns for that type)
  headline_lead            "<title>. <summary>"      all sources
  headline_lead_sentcase   same, Title Case headlines converted to sentence case (casing experiment)
  fulltext                 "<title>. <text>"         only rows with text (Guardian today)

Outputs (in --output-dir, default data/raw/outputs)
  ner.csv       article_id, source, model, input_type, start_char, end_char, text, label
                label is spaCy's own label (PERSON, ORG, GPE, NORP, LOC, FAC, DATE, ...)
  ner_runs.csv  article_id, source, model, input_type, n_chars, n_entities, seconds
                one row per processed article, including articles with zero entities
"""
import argparse
import html
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = "data/raw/news.csv"   # made by src/collect/export_csv.py
DEFAULT_OUTPUT_DIR = "data/raw/outputs"
DEFAULT_MODEL = "en_core_web_sm"
INPUT_TYPES = ("headline_lead", "headline_lead_sentcase", "fulltext")
CHUNK_CHARS = 1500           # long articles are split at sentence boundaries into chunks of this size
MIN_FULLTEXT_CHARS = 400     # rows with less text are treated as having no full text
BATCH_ARTICLES = 16
SAVE_EVERY_BATCHES = 20

NER_COLS = ["article_id", "source", "model", "input_type", "start_char", "end_char", "text", "label"]
RUN_COLS = ["article_id", "source", "model", "input_type", "n_chars", "n_entities", "seconds"]


# ---- loading ------------------------------------------------------------------------------------
def load_articles(path):
    """Read the CSV (empty cells stay ''), and drop rows whose normalized URL was already seen."""
    p = Path(path)
    if not p.is_file():
        sys.exit(f"Input file not found: {p.resolve()}  (use --input /path/to/news.csv)")
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    missing = {"id", "source", "title", "summary", "text"} - set(df.columns)
    if missing:
        sys.exit(f"{p} is missing required columns: {sorted(missing)}")
    df["id"] = df["id"].astype(int)
    if "url" in df.columns:
        key = df["url"].str.lower().str.replace(r"[?#].*$", "", regex=True).str.rstrip("/")
        dup = key.duplicated(keep="first") & (key != "")
        print(f"{len(df)} rows, {int(dup.sum())} duplicate URLs dropped", file=sys.stderr)
        df = df[~dup]
    return df.reset_index(drop=True)


# ---- text preparation ---------------------------------------------------------------------------
_INVISIBLE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_AUTHOR_SUFFIX = re.compile(r"\s+\|\s+[^|]+$")                 # Guardian: "Title | Author"
_SENT_SPLIT = re.compile(r"(?<=[.!?…”\"’])\s+(?=[A-Z“\"‘'])")


def clean(s):
    """Unescape HTML entities (e.g. &#039;), drop invisible characters, collapse whitespace."""
    if not s:
        return ""
    s = html.unescape(str(s))
    s = _INVISIBLE.sub("", s).replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def clean_title(title):
    # without this, the author name after '|' is extracted as a PERSON in every opinion piece
    return _AUTHOR_SUFFIX.sub("", clean(title))


def end_sentence(s):
    return s if (not s or s[-1] in ".!?…:;\"”’") else s + "."


def is_title_case(s):
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'’-]*", s) if len(w) >= 4]
    return len(words) >= 3 and sum(w[0].isupper() for w in words) / len(words) >= 0.8


def sentence_case(s):
    """Only touches headlines that look Title Case. Keeps ALLCAPS acronyms and the first word.
    Note: this removes capitalization cues, so it can also hurt (e.g. 'Nasa', 'Pfas')."""
    if not is_title_case(s):
        return s
    return " ".join(w if (i == 0 or w.isupper() or w == "I") else w.lower()
                    for i, w in enumerate(s.split(" ")))


def build_input(row, input_type):
    """The exact string the model sees. Entity offsets refer to this string."""
    title = clean_title(row["title"])
    if input_type == "headline_lead_sentcase":
        title = sentence_case(title)
    if input_type in ("headline_lead", "headline_lead_sentcase"):
        summary = clean(row["summary"])
        return f"{end_sentence(title)} {summary}".strip() if (title or summary) else None
    if input_type == "fulltext":
        text = clean(row["text"])
        if len(text) < MIN_FULLTEXT_CHARS:
            return None
        return f"{end_sentence(title)} {text}".strip()
    raise ValueError(f"unknown input type: {input_type}")


def chunk(text, max_chars=CHUNK_CHARS):
    """Split at sentence boundaries into contiguous slices -> [(offset, chunk_text)].
    Entities do not span sentences, so no overlap is needed and offsets stay exact."""
    spans, pos = [], 0
    for m in _SENT_SPLIT.finditer(text):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(text)))

    out, cs, ce = [], None, None
    for a, b in spans:
        while b - a > max_chars:                         # hard-split an overlong "sentence"
            cut = text.rfind(" ", a, a + max_chars)
            if cut <= a:
                cut = a + max_chars
            if cs is not None:
                out.append((cs, text[cs:ce]))
                cs = None
            out.append((a, text[a:cut]))
            a = cut + 1
        if cs is None:
            cs, ce = a, b
        elif b - cs > max_chars:
            out.append((cs, text[cs:ce]))
            cs, ce = a, b
        else:
            ce = b
    if cs is not None:
        out.append((cs, text[cs:ce]))
    return [(o, t) for o, t in out if t.strip()]


# ---- spaCy --------------------------------------------------------------------------------------
def load_nlp(model):
    try:
        import spacy
    except ImportError:
        sys.exit("spaCy is not installed. Run: pip install spacy")
    try:
        spacy.prefer_gpu()
    except Exception:
        pass
    t0 = time.time()
    try:
        nlp = spacy.load(model)
    except OSError:
        sys.exit(f"spaCy model '{model}' is not installed. Run: python -m spacy download {model}")
    # NER does not need these components; disabling them makes processing faster
    drop = [p for p in ("lemmatizer", "tagger", "parser", "attribute_ruler", "senter") if p in nlp.pipe_names]
    nlp.select_pipes(disable=drop)
    print(f"loaded {model} in {time.time() - t0:.1f}s", file=sys.stderr)
    return nlp


# ---- run ----------------------------------------------------------------------------------------
def _read_or_empty(path, cols):
    return pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame(columns=cols)


def _save(out_dir, ner_rows, run_rows):
    pd.DataFrame(ner_rows, columns=NER_COLS).to_csv(out_dir / "ner.csv", index=False)
    pd.DataFrame(run_rows, columns=RUN_COLS).to_csv(out_dir / "ner_runs.csv", index=False)


def cmd_run(a):
    df = load_articles(a.input)
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ner_rows = _read_or_empty(out_dir / "ner.csv", NER_COLS).to_dict("records")
    run_rows = _read_or_empty(out_dir / "ner_runs.csv", RUN_COLS).to_dict("records")
    done = {(r["article_id"], r["model"], r["input_type"]) for r in run_rows}
    print(f"resuming: {len(done)} article/model/input combinations already done", file=sys.stderr)

    nlp = None
    for it in a.inputs:
        sub = df if not a.sources else df[df["source"].isin(a.sources)]
        if it == "fulltext":
            sub = sub[sub["text"].str.len() >= MIN_FULLTEXT_CHARS]
        if not a.redo:
            sub = sub[[(int(i), a.model, it) not in done for i in sub["id"]]]
        if a.limit:
            sub = sub.head(a.limit)
        work = [(int(r["id"]), r["source"], build_input(r, it)) for _, r in sub.iterrows()]
        work = [w for w in work if w[2]]
        print(f"[{a.model} / {it}] {len(work)} articles to process", file=sys.stderr)
        if not work:
            continue
        if nlp is None:
            nlp = load_nlp(a.model)

        t_start = time.time()
        for bi, i in enumerate(range(0, len(work), BATCH_ARTICLES)):
            group = work[i:i + BATCH_ARTICLES]
            flat, owner = [], []
            for aid, _, txt in group:
                for off, ch in chunk(txt, a.chunk_chars):
                    flat.append(ch)
                    owner.append((aid, off))
            t0 = time.time()
            docs = list(nlp.pipe(flat, batch_size=32))
            dt = time.time() - t0

            found = defaultdict(list)
            for (aid, off), doc in zip(owner, docs):
                for e in doc.ents:
                    found[aid].append((e.start_char + off, e.end_char + off, e.label_))

            total_chars = sum(len(t) for _, _, t in group) or 1
            for aid, src, txt in group:
                if a.redo:
                    ner_rows[:] = [r for r in ner_rows if not (r["article_id"] == aid and r["model"] == a.model and r["input_type"] == it)]
                    run_rows[:] = [r for r in run_rows if not (r["article_id"] == aid and r["model"] == a.model and r["input_type"] == it)]
                n = 0
                for s, e, label in sorted(found[aid]):
                    span = txt[s:e]
                    s += len(span) - len(span.lstrip())
                    span = span.strip()
                    if not span:
                        continue
                    ner_rows.append(dict(article_id=aid, source=src, model=a.model, input_type=it,
                                         start_char=s, end_char=s + len(span), text=span, label=label))
                    n += 1
                run_rows.append(dict(article_id=aid, source=src, model=a.model, input_type=it,
                                     n_chars=len(txt), n_entities=n, seconds=dt * len(txt) / total_chars))
                done.add((aid, a.model, it))
            if bi % SAVE_EVERY_BATCHES == SAVE_EVERY_BATCHES - 1:
                _save(out_dir, ner_rows, run_rows)
            rate = (i + len(group)) / (time.time() - t_start)
            print(f"  {i + len(group)}/{len(work)} articles ({rate:.1f}/s)", end="\r", file=sys.stderr)
        _save(out_dir, ner_rows, run_rows)
        print(file=sys.stderr)
    print(f"saved to {out_dir}/ner.csv and ner_runs.csv", file=sys.stderr)


# ---- stats --------------------------------------------------------------------------------------
def cmd_stats(a):
    out_dir = Path(a.output_dir)
    if not (out_dir / "ner.csv").exists():
        sys.exit(f"No results in {out_dir}. Run `python ner.py run` first.")
    ner = pd.read_csv(out_dir / "ner.csv", keep_default_na=False)
    runs = pd.read_csv(out_dir / "ner_runs.csv", keep_default_na=False)

    s = (runs.groupby(["input_type", "model", "source"])
         .agg(articles=("article_id", "count"), ents_per_article=("n_entities", "mean"),
              chars=("n_chars", "sum"), ents=("n_entities", "sum"), sec_per_article=("seconds", "mean")))
    s["ents_per_1k_chars"] = 1000 * s["ents"] / s["chars"]
    print("== coverage and speed ==")
    print(s.drop(columns=["chars", "ents"]).round(2).to_string())

    d = ner[ner["input_type"] == a.input_type]
    print(f"\n== entity labels ({a.input_type}) ==")
    print(d["label"].value_counts().to_string())

    print(f"\n== most frequent entities, by number of articles ({a.input_type}; raw, not normalized) ==")
    for label in a.labels:
        top = d[d["label"] == label].groupby("text")["article_id"].nunique().sort_values(ascending=False).head(a.top)
        if len(top):
            print(f"{label:8}", ", ".join(f"{t} ({n})" for t, n in top.items()))


# ---- gold sample --------------------------------------------------------------------------------
def cmd_export_sample(a):
    df = load_articles(a.input)
    groups = defaultdict(list)
    for _, r in df.iterrows():
        txt = build_input(r, a.input_type)
        if txt:
            key = r["source"] if r["source"] in ("guardian", "nyt") else "rss_other"
            groups[key].append({"article_id": int(r["id"]), "source": r["source"],
                                "input_type": a.input_type, "text": txt, "annotations": []})
    if a.input_type == "fulltext":
        quotas = {"guardian": a.n}
    else:
        quotas = {"guardian": round(a.n * 0.4), "nyt": round(a.n * 0.4)}
        quotas["rss_other"] = a.n - quotas["guardian"] - quotas["nyt"]
    rng, sample = random.Random(a.seed), []
    for k, q in quotas.items():
        pool = groups.get(k, [])
        sample += rng.sample(pool, min(q, len(pool)))
        if len(pool) < q:
            print(f"warning: only {len(pool)} items for {k}, wanted {q}", file=sys.stderr)
    rng.shuffle(sample)
    out = Path(a.out or Path(a.output_dir) / f"gold_ner_{a.input_type}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in sample) + "\n", encoding="utf-8")
    print(f"wrote {len(sample)} items to {out}", file=sys.stderr)


# ---- cli ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input", default=DEFAULT_INPUT, help=f"news CSV (default: {DEFAULT_INPUT})")
    common.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"(default: {DEFAULT_OUTPUT_DIR})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", parents=[common], help="extract entities")
    p.add_argument("--model", default=DEFAULT_MODEL, help="installed spaCy model, e.g. en_core_web_sm, en_core_web_trf")
    p.add_argument("--inputs", nargs="+", choices=INPUT_TYPES, default=["headline_lead"])
    p.add_argument("--sources", nargs="+", help="e.g. guardian nyt (default: all)")
    p.add_argument("--limit", type=int, help="max articles per input type (for tests)")
    p.add_argument("--redo", action="store_true", help="reprocess articles already in the output")
    p.add_argument("--chunk-chars", type=int, default=CHUNK_CHARS)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("stats", parents=[common], help="coverage, speed and top entities")
    p.add_argument("--input-type", choices=INPUT_TYPES, default="headline_lead")
    p.add_argument("--labels", nargs="+", default=["PERSON", "ORG", "GPE", "NORP", "LOC"])
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("export-sample", parents=[common], help="stratified sample for gold annotation")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--input-type", choices=INPUT_TYPES, default="headline_lead")
    p.add_argument("--out", help="output JSONL (default: <output-dir>/gold_ner_<input-type>.jsonl)")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=cmd_export_sample)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()