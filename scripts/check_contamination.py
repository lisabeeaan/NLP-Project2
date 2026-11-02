# Uses common English words to compare possible English contamination across page-level isiXhosa share bands.
import os
import pandas as pd

XH_DIR = "data/XhTxt"
TRACKER_CSV = "data/LanguageSplitTracker70-94.csv"
ENGLISH_WORDS = {"the", "of", "and", "for", "with", "from", "this", "that", "which", "are",
                 "was", "will", "your", "you", "per", "all", "have", "has",
                 "been", "not", "but"}

def english_rate(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    words = [w.lower().strip(".,;:!?()\"'“”‘’") for w in text.split()]
    return sum(w in ENGLISH_WORDS for w in words) / max(len(words), 1)

t = pd.read_csv(TRACKER_CSV)
t["xh_word_share"] = 100 * t["Xho_Wordcount"] / (t["Xho_Wordcount"] + t["Eng_Wordcount"]).clip(lower=1)
t = t[t["Edition"].isin(set(os.listdir(XH_DIR))) & (t["Xho_Wordcount"] >= 100)].copy()
t["english_rate"] = [english_rate(os.path.join(XH_DIR, f)) for f in t["Edition"]]
t["share_band"] = pd.cut(t["xh_word_share"], [-1, 25, 50, 70, 90, 100])
print(t.groupby("share_band", observed=True)["english_rate"]
        .agg(pages="count", median="median", p90=lambda x: x.quantile(0.9)))
