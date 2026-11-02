# Counts selected visible text anomalies by corpus and period and compares them with anomalies produced by synthetic noise.
import json, re
import pandas as pd

PATTERNS = {
    "digit inside a word": re.compile(r"[A-Za-z]\d+[A-Za-z]"),
    "symbol inside a word": re.compile(r"[A-Za-z][|~^*_\\][A-Za-z]"),
    "full stop inside a word": re.compile(r"[a-z]\.[A-Za-z]"),
}

def per_1000_tokens(sentences):
    n_tokens = sum(len(x.split()) for x in sentences)
    row = {name: 1000 * sum(len(p.findall(x)) for x in sentences) / n_tokens for name, p in PATTERNS.items()}
    row["total"] = sum(row.values())
    row["sentences"] = len(sentences)
    return row

def period(source, year):
    if source == "ISIG":
        return "ISIG 1870-79" if year < 1880 else "ISIG 1880-88"
    return "IMVO 1884-89" if year < 1890 else "IMVO 1890-94"

groups = {}
for line in open("data/historical_clean/historical_clean.jsonl", encoding="utf-8"):
    r = json.loads(line)
    key = period(r["source"], int(r["date"][:4]))
    groups.setdefault(key, []).extend(r["sentences"])
groups["historical (all)"] = [x for k, v in groups.items() for x in v]
groups["modern (NCHLT)"] = [json.loads(l)["text"] for l in open("data/modern_corpus/modern_sentences.jsonl", encoding="utf-8")]

table = pd.DataFrame({name: per_1000_tokens(sents) for name, sents in groups.items()}).T
table.to_csv("results/filtering/visible_junk_rate.csv")
print(table.round(2).to_string())


import random
CONFUSIONS = {"l": "1", "I": "l", "O": "0", "o": "0", "e": "c", "c": "e",
              "m": "rn", "n": "u", "u": "n", "h": "b", "b": "h", "i": "j"}

def corrupt(sentence, rate, rng):
    out = []
    for ch in sentence:
        if ch.isalpha() and rng.random() < rate:
            r = rng.random()
            if ch in CONFUSIONS and r < 0.6:
                out.append(CONFUSIONS[ch])
            elif r < 0.8:
                continue
            else:
                out.append(ch + rng.choice("|~^*_.,"))
        else:
            out.append(ch)
    return "".join(out)


# Exclude full stops because they can be valid punctuation within tokens.
def visible(sentences):
    row = per_1000_tokens(sentences)
    return row["digit inside a word"] + row["symbol inside a word"]

modern_pool = groups["modern (NCHLT)"][:3000]
modern_base = visible(modern_pool)
noisy = [corrupt(x, 0.01, random.Random(f"0.01-{i}")) for i, x in enumerate(modern_pool)]
slope = (visible(noisy) - modern_base) / 0.01
print(f"\nSynthetic corruption at 1% adds {visible(noisy) - modern_base:.1f} visibly damaged tokens per 1,000.")
print("Excess visible damage over the modern text, and the synthetic corruption rate that would produce it:")
for name, sents in groups.items():
    if name.startswith("modern"):
        continue
    excess = max(visible(sents) - modern_base, 0.0)
    print(f"  {name:18s} {excess:5.2f} per 1,000 tokens  ->  about {100 * excess / slope:.3f}% of letters")
