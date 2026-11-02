# Loads historical isiXhosa page files, applies the configured tracker thresholds and saves the text with page metadata.
import os, re, json
import pandas as pd

XH_DIR = "data/XhTxt"
TRACKER_CSV = "data/LanguageSplitTracker70-94.csv"
OUT_FILE = "data/historical_clean/historical_raw.jsonl"


# Page purity filtering is disabled.
MIN_XH_PURITY = 0
MIN_XH_WORDCOUNT = 100

def read_text(path):

    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()

tracker = pd.read_csv(TRACKER_CSV)
tracker = tracker.set_index("Edition")

records = []
skipped_thin, skipped_missing = 0, 0
for fname in sorted(os.listdir(XH_DIR)):
    if not fname.endswith(".txt"):
        continue

    if fname not in tracker.index:
        skipped_missing += 1
        continue

    row = tracker.loc[fname]

    xh_word_share = 100 * row["Xho_Wordcount"] / max(row["Xho_Wordcount"] + row["Eng_Wordcount"], 1)
    if xh_word_share < MIN_XH_PURITY or row["Xho_Wordcount"] < MIN_XH_WORDCOUNT:
        skipped_thin += 1
        continue

    path = os.path.join(XH_DIR, fname)
    text = read_text(path)

    match = re.match(r"([A-Za-z]+)_(\d{4}-\d{2}-\d{2})_p(\d+)", fname)
    source, date, page = (match.groups() if match else (None, None, None))


    # Convert pandas scalars before writing JSON.
    def to_native(x):
        return x.item() if hasattr(x, "item") else x

    records.append({"file": fname, "source": source, "date": date, "page": page,
                     "text": text, "xh_percent": to_native(row["Xh%"]),
                     "xh_word_share": float(xh_word_share),
                     "xh_wordcount": to_native(row["Xho_Wordcount"])})

os.makedirs("data/historical_clean", exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    for r in records:

        f.write(json.dumps(r, ensure_ascii=False,
                            default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")

print(f"Loaded {len(records)} pages "
      f"(skipped {skipped_thin} thin/low-purity, {skipped_missing} missing from tracker)")
