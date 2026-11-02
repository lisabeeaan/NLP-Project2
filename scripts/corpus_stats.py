# Reports the cleaned historical corpus by newspaper, year and sentence length and saves a sample of retained sentences.
import json, os
import pandas as pd

rows = []
for line in open("data/historical_clean/historical_clean.jsonl", encoding="utf-8"):
    r = json.loads(line)
    for sent in r["sentences"]:
        rows.append({"source": r["source"], "year": (r["date"] or "")[:4], "file": r["file"],
                     "sentence": sent, "words": len(sent.split()), "chars": len(sent)})
df = pd.DataFrame(rows)

print(f"{len(df)} sentences from {df['file'].nunique()} pages")
print("\nSentences by newspaper:\n", df.groupby("source").size())
print("\nSentences by year:\n", df.groupby("year").size().to_string())
print("\nWords per sentence:\n", df["words"].describe(percentiles=[.05, .25, .5, .75, .95]).round(1))

os.makedirs("results/filtering", exist_ok=True)
with open("results/filtering/kept_sentences_sample.txt", "w", encoding="utf-8") as f:
    for _, row in df.sample(min(50, len(df)), random_state=42).iterrows():
        f.write(f"[{row['file']}] {row['sentence']}\n")
df[["words", "chars"]].describe().to_csv("results/filtering/historical_length_stats.csv")
