# Draws the original historical and modern samples with equal counts across shared character-length bins.
import json, os, random
import numpy as np

N_SAMPLE = 300
N_BINS = 5
random.seed(42)

hist = []
for line in open("data/historical_clean/historical_clean.jsonl", encoding="utf-8"):
    r = json.loads(line)
    hist += [{"text": s, "file": r["file"]} for s in r["sentences"]]
modern = [json.loads(l) for l in open("data/modern_corpus/modern_sentences.jsonl", encoding="utf-8")]


# Use the historical length distribution to define shared bins.
edges = np.quantile([len(x["text"]) for x in hist], np.linspace(0, 1, N_BINS + 1))
edges[0], edges[-1] = 0, float("inf")

def bin_of(x):
    return int(np.searchsorted(edges, len(x["text"]), side="right") - 1)

def draw(pool):
    bins = [[] for _ in range(N_BINS)]
    for x in pool:
        bins[bin_of(x)].append(x)
    per_bin = min(N_SAMPLE // N_BINS, *(len(b) for b in bins))
    return bins, per_bin

hist_bins, n_h = draw(hist)
modern_bins, n_m = draw(modern)
per_bin = min(n_h, n_m)

os.makedirs("data/samples", exist_ok=True)
for name, bins in [("hist_sample", hist_bins), ("modern_sample", modern_bins)]:
    out = [x for b in bins for x in random.sample(b, per_bin)]
    random.shuffle(out)
    with open(f"data/samples/{name}.jsonl", "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{name}: {len(out)} sentences, mean {np.mean([len(x['text']) for x in out]):.0f} characters")
