# Builds the modern comparison corpus from the NCHLT clean file using sentence filters, duplicate removal and a per-document cap.
import json, os, random, re
from collections import Counter, defaultdict
import pandas as pd
from clean_corpus import sentence_split, reject_reason

CLEAN_CORPUS = "data/modern_corpus/xh/Corpora/CORP.NCHLT.xh.CLEAN.2.0.txt"
OUT_FILE = "data/modern_corpus/modern_sentences.jsonl"
HEADER_LINES = 30
HEADER_WORDS = re.compile(r"nchlt|corpus|corpora|creative ?commons|licen[cs]e|ctext|nwu\.ac\.za|attribute|copyright|north-west university|url:", re.I)

# The file markers preserve source-document boundaries.
DOC_TAG = re.compile(r"^<fn>(.*)</fn>$")
MAX_PER_DOC = 200
random.seed(42)

def read_lines(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().splitlines()
    except UnicodeDecodeError:
        with open(path, encoding="latin-1") as f:
            return f.read().splitlines()

lines = read_lines(CLEAN_CORPUS)
print(f"{len(lines)} lines. First 8 (check the layout):")
for ln in lines[:8]:
    print("  |", ln[:120])

docs, reasons, seen = defaultdict(list), Counter(), set()
doc = "(before the first document)"
for i, line in enumerate(lines):
    line = line.strip()
    m = DOC_TAG.match(line)
    if m:
        doc = m.group(1)
        continue
    if not re.search(r"[A-Za-z]", line):
        continue
    if i < HEADER_LINES and HEADER_WORDS.search(line):
        reasons["licence header"] += 1
        continue
    for sent in sentence_split(line):
        why = reject_reason(sent)
        if why:
            reasons[why] += 1
            continue

        # Ignore case and repeated whitespace when checking duplicates.
        key = re.sub(r"\s+", " ", sent).strip().lower()
        if key in seen:
            reasons["duplicate"] += 1
            continue
        seen.add(key)
        docs[doc].append(sent)

sizes = sorted((len(v) for v in docs.values()), reverse=True)
total_before_cap = sum(sizes)
print(f"\n{len(docs)} documents; the 5 largest hold {sum(sizes[:5])} of {total_before_cap} sentences")

kept = []
for d, sents in docs.items():
    if MAX_PER_DOC and len(sents) > MAX_PER_DOC:
        reasons["over per-document cap"] += len(sents) - MAX_PER_DOC
        sents = random.sample(sents, MAX_PER_DOC)
    kept += [{"text": x, "doc": d} for x in sents]

os.makedirs("data/modern_corpus", exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    for x in kept:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

df = pd.DataFrame({"words": [len(x["text"].split()) for x in kept], "chars": [len(x["text"]) for x in kept]})
os.makedirs("results/filtering", exist_ok=True)
df.describe().to_csv("results/filtering/modern_length_stats.csv")
print(f"{len(kept)} modern sentences kept; dropped: {dict(reasons)}")
print(df.describe(percentiles=[.05, .25, .5, .75, .95]).round(1))
