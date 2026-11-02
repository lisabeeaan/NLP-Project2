# Adds exploratory normalised scores, confidence intervals, effect sizes, newspaper comparisons and synthetic-noise comparisons.
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from transformers import AutoTokenizer
from load_models import MODELS

rng = np.random.default_rng(42)
scores = pd.read_json("results/perplexity/scores.jsonl", lines=True)
hist_meta = pd.read_json("data/samples/hist_sample.jsonl", lines=True).drop_duplicates("text")

def add_metrics(df, model_key):
    """Add negative log scores per token and per character."""
    tok = AutoTokenizer.from_pretrained(MODELS[model_key]["name"])
    is_mlm = MODELS[model_key]["type"] == "mlm"
    nll_tok, nll_chr = [], []
    for sentence, score in zip(df["sentence"], df["score"]):
        n_ids = len(tok(sentence)["input_ids"])
        if is_mlm:
            n = max(n_ids - 2, 1)
            per_token = -score
        else:
            n = max(n_ids - 1, 1)
            per_token = float(np.log(score))
        nll_tok.append(per_token)
        nll_chr.append(per_token * n / len(sentence))
    df = df.copy()
    df["nll_token"], df["nll_char"] = nll_tok, nll_chr
    return df

def holm(p):
    p = np.asarray(p, dtype=float)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj

frames = [add_metrics(scores[scores["model"] == k], k) for k in MODELS if (scores["model"] == k).any()]
all_scores = pd.concat(frames)

rows = []
for metric in ["nll_token", "nll_char"]:
    block = []
    for model_key, d in all_scores.groupby("model", sort=False):
        h = d.loc[d["corpus"] == "historical", metric].to_numpy()
        m = d.loc[d["corpus"] == "modern", metric].to_numpy()
        u, p = mannwhitneyu(h, m, alternative="two-sided")
        cliffs = 2 * u / (len(h) * len(m)) - 1
        boots = [np.median(rng.choice(h, len(h))) - np.median(rng.choice(m, len(m))) for _ in range(2000)]
        block.append({"metric": metric, "model": model_key, "has_xh": MODELS[model_key]["has_xh"],
                      "n_hist": len(h), "n_modern": len(m),
                      "median_hist": np.median(h), "median_modern": np.median(m),
                      "gap": np.median(h) - np.median(m),
                      "gap_ci_low": np.percentile(boots, 2.5), "gap_ci_high": np.percentile(boots, 97.5),
                      "cliffs_delta": cliffs, "p": p})
    for row, adj in zip(block, holm([r["p"] for r in block])):
        row["p_holm"] = adj
    rows += block
summary = pd.DataFrame(rows)
summary.to_csv("results/perplexity/summary_table.csv", index=False)
print("Gap = median(historical) - median(modern) of the negative log-probability; positive means historical text is harder.")
print(summary.round(3).to_string(index=False))


# Attach source metadata for the newspaper comparisons.
hist = all_scores[all_scores["corpus"] == "historical"].merge(
    hist_meta[["text", "file"]], left_on="sentence", right_on="text", how="left")
hist[["source", "year"]] = hist["file"].str.extract(r"^([A-Za-z]+)_(\d{4})")
hist["year"] = hist["year"].astype(float)
def period(r):
    if r["source"] == "ISIG":
        return "ISIG 1870-79" if r["year"] < 1880 else "ISIG 1880-88"
    return "IMVO 1884-89" if r["year"] < 1890 else "IMVO 1890-94"
hist["period"] = hist.apply(period, axis=1)
modern_med = all_scores[all_scores["corpus"] == "modern"].groupby("model")["nll_char"].median()
by_period = hist.groupby(["model", "period"])["nll_char"].agg(n="count", median="median").reset_index()
by_period["gap_vs_modern"] = by_period["median"] - by_period["model"].map(modern_med)
by_period.to_csv("results/perplexity/by_period.csv", index=False)
print("\nPer-character gap by newspaper and period (positive = harder than modern):")
print(by_period.round(3).to_string(index=False))


contrast = []
for model_key, d in hist.groupby("model", sort=False):
    a = d.loc[d["source"] == "IMVO", "nll_char"].to_numpy()
    b = d.loc[d["source"] == "ISIG", "nll_char"].to_numpy()
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    boots = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b))) for _ in range(2000)]
    contrast.append({"model": model_key, "n_imvo": len(a), "n_isig": len(b),
                     "imvo_minus_isig": np.median(a) - np.median(b),
                     "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5), "p": p})
for row, adj in zip(contrast, holm([r["p"] for r in contrast])):
    row["p_holm"] = adj
contrast = pd.DataFrame(contrast)
contrast.to_csv("results/perplexity/imvo_vs_isig.csv", index=False)
print("\nImvo minus Isigidimi, median per-character negative log-probability (positive = Imvo harder):")
print(contrast.round(3).to_string(index=False))


noise = pd.read_json("results/noise/noise_scores.jsonl", lines=True)
is_mlm = noise["model"].map(lambda k: MODELS[k]["type"] == "mlm")
noise["bad"] = np.where(is_mlm, -noise["score"], np.log(noise["score"].where(~is_mlm, 1.0)))
base = noise[noise["rate"] == 0].groupby("model")["bad"].mean()
out = []
for model_key, d in all_scores.groupby("model", sort=False):
    gap_mean = d.loc[d["corpus"] == "historical", "nll_token"].mean() - d.loc[d["corpus"] == "modern", "nll_token"].mean()
    row = {"model": model_key, "mean_gap": gap_mean}
    for rate in [0.01, 0.02, 0.05]:
        shift = noise[(noise["model"] == model_key) & (noise["rate"] == rate)]["bad"].mean() - base[model_key]
        row[f"noise_{int(rate * 100)}%_share_of_gap"] = shift / gap_mean if gap_mean > 0.25 else np.nan
    out.append(row)
print("\nShare of the historical-modern gap that noise at each rate would reproduce (NaN = no meaningful gap to explain):")
print(pd.DataFrame(out).round(2).to_string(index=False))
