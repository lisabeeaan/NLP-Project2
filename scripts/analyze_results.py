# Compares exploratory score distributions with Mann-Whitney tests and saves the original per-model boxplots.
import json, pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame([json.loads(l) for l in open("results/perplexity/scores.jsonl", encoding="utf-8")])

for model in df["model"].unique():
    sub = df[df["model"] == model]
    hist_scores = sub[sub["corpus"] == "historical"]["score"]
    mod_scores = sub[sub["corpus"] == "modern"]["score"]
    stat, p = mannwhitneyu(hist_scores, mod_scores, alternative="two-sided")
    print(f"{model}: historical median={hist_scores.median():.3f}, "
          f"modern median={mod_scores.median():.3f}, p={p:.4g}")


# Each model has its own scale; higher PLL and lower perplexity are better.
models = list(df["model"].unique())
fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 4.5), sharey=False)
for ax, model in zip(axes, models):
    sub = df[df["model"] == model]
    sns.boxplot(data=sub, x="corpus", y="score", ax=ax)
    tag = "" if sub["has_xh"].iloc[0] else " (baseline, no isiXhosa)"
    ax.set_title(model + tag, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("mean PLL per token" if model != "inkubalm" else "perplexity")
plt.tight_layout()
plt.savefig("results/perplexity/score_comparison.png", dpi=200)
