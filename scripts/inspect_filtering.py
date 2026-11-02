# Shows how proposed page filters would affect the corpus and saves threshold histograms and examples for inspection.
import os, random
import pandas as pd

XH_DIR = "data/XhTxt"
TRACKER_CSV = "data/LanguageSplitTracker70-94.csv"


# These thresholds are diagnostic; this script does not filter the corpus.
MIN_XH_PURITY, MIN_XH_WORDCOUNT = 90, 100

t = pd.read_csv(TRACKER_CSV)
t = t[t["Edition"].isin(set(os.listdir(XH_DIR)))].copy()
t["xh_word_share"] = 100 * t["Xho_Wordcount"] / (t["Xho_Wordcount"] + t["Eng_Wordcount"]).clip(lower=1)
t["source"] = t["Edition"].str.extract(r"^([A-Za-z]+)_")
t["year"] = t["Edition"].str.extract(r"_(\d{4})-")
t["page"] = t["Edition"].str.extract(r"_p(\d+)")

low_purity = t["xh_word_share"] < MIN_XH_PURITY
thin = t["Xho_Wordcount"] < MIN_XH_WORDCOUNT
t["status"] = "would keep"
t.loc[low_purity & ~thin, "status"] = "would drop: low purity only"
t.loc[~low_purity & thin, "status"] = "would drop: thin only"
t.loc[low_purity & thin, "status"] = "would drop: both"

print(f"load_txt.py actually keeps {int((t['Xho_Wordcount'] >= MIN_XH_WORDCOUNT).sum())} of {len(t)} pages (purity filter off)\n")
print("--- What a 90% purity filter would do ---")
print(t["status"].value_counts(), "\n")
print("--- By newspaper ---")
print(t.groupby(["source", "status"]).size().unstack(fill_value=0), "\n")
print("--- By year ---")
print(t.groupby(["year", "status"]).size().unstack(fill_value=0), "\n")
print("--- By page number within the edition ---")
print(t.groupby(["source", "page", "status"]).size().unstack(fill_value=0), "\n")

print("--- Pages kept under other thresholds (rows: min isiXhosa word share %, columns: min words) ---")
grid = {w: [int(((t["xh_word_share"] >= p) & (t["Xho_Wordcount"] >= w)).sum()) for p in [70, 80, 90, 95]]
        for w in [0, 50, 100, 200, 300]}
print(pd.DataFrame(grid, index=[70, 80, 90, 95]), "\n")


random.seed(42)
rows = []
for status in ["would drop: low purity only", "would drop: thin only", "would drop: both"]:
    grp = t[t["status"] == status]
    rows.append(grp.sample(min(10, len(grp)), random_state=42).assign(why=status))
borderline = t[(t["status"] != "would keep") &
               (((t["xh_word_share"] >= MIN_XH_PURITY - 10) & (t["xh_word_share"] < MIN_XH_PURITY)) |
                ((t["Xho_Wordcount"] >= MIN_XH_WORDCOUNT // 2) & (t["Xho_Wordcount"] < MIN_XH_WORDCOUNT)))]
rows.append(borderline.sample(min(10, len(borderline)), random_state=42).assign(why="borderline"))

os.makedirs("results/filtering", exist_ok=True)
out = pd.concat(rows)[["Edition", "why", "Xh%", "xh_word_share", "Xho_Wordcount", "Eng_Wordcount", "Tot_Wordcount"]]
out.to_csv("results/filtering/files_to_inspect.csv", index=False)
print(f"Wrote {len(out)} files to inspect to results/filtering/files_to_inspect.csv")


import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
t["xh_word_share"].hist(bins=40, ax=ax[0]); ax[0].axvline(MIN_XH_PURITY, color="red"); ax[0].set_title("isiXhosa share of words per page (%)")
t["Xho_Wordcount"].clip(upper=3000).hist(bins=40, ax=ax[1]); ax[1].axvline(MIN_XH_WORDCOUNT, color="red")
ax[1].set_title("isiXhosa words per page (capped at 3000)")
plt.tight_layout(); plt.savefig("results/filtering/threshold_histograms.png", dpi=200)
