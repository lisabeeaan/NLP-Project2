# Normalises historical page text, joins likely broken passages, splits sentences and records which sentences fail the filters.
import json, os, random, re
from collections import Counter


MIN_WORDS = 3
MIN_CHARS = 25
MAX_WORDS = 80
MAX_DIGIT_SHARE = 0.3
MAX_UPPER_SHARE = 0.8
MAX_ENGLISH_SHARE = 0.15


# Keep title words such as sir and dear out of this filter.
ENGLISH_WORDS = {"the", "of", "and", "for", "with", "from", "this", "that", "which", "are",
                 "was", "will", "your", "you", "per", "all", "have", "has",
                 "been", "not", "but"}

def clean_block(block):
    block = block.replace("\u00ad", "")
    block = re.sub(r"\[MISSING PICTURE\]|\?{2,}", " ", block)
    block = re.sub(r"(\w)-[ \t]*\n[ \t]*(\w)", r"\1\2", block)
    block = re.sub(r"\s*\n\s*", " ", block)
    block = re.sub(r"(?<=\w)- (?=[a-z])", "", block)
    block = re.sub(r"[ \t]+", " ", block)
    block = re.sub(r"[|_~^]{2,}", "", block)
    return block.strip()

TITLES = ("mr", "mrs", "dr", "rev", "capt", "col", "genl", "st", "messrs")

def should_join(prev, nxt):
    """Check whether the next block appears to continue this one."""
    p = prev.rstrip()
    if re.search(r"\b(?:" + "|".join(TITLES) + r")\.$", p, re.I):
        return True
    open_end = not re.search(r"[.!?:;\"”’)\]]$", p)
    return open_end and nxt[:1].islower()

def join_broken_blocks(blocks):
    out = []
    for b in blocks:
        if out and should_join(out[-1], b):
            if out[-1].endswith("-"):
                out[-1] = out[-1][:-1] + b
            else:
                out[-1] = out[-1] + " " + b
        else:
            out.append(b)
    return out

def clean_page(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [clean_block(b) for b in re.split(r"\n\s*\n", text)]
    blocks = [b for b in blocks if b and not re.fullmatch(r"\d{1,4}", b)]
    return join_broken_blocks(blocks)

def ends_with_abbreviation(sent):
    """Check for a trailing initial or title abbreviation."""
    if re.search(r"(?:^|\s)[A-Za-z]\.$", sent):
        return True
    return bool(re.search(r"\b(?:" + "|".join(TITLES) + r")\.$", sent, re.I))

def sentence_split(block):


    parts = [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z“\"‘'])", block) if x.strip()]
    out = []
    for part in parts:
        if out and ends_with_abbreviation(out[-1]):
            out[-1] += " " + part
        else:
            out.append(part)
    return out

def reject_reason(sentence):
    words = sentence.split()
    if len(words) < MIN_WORDS or len(sentence) < MIN_CHARS:
        return "too short"
    if len(words) > MAX_WORDS:
        return "too long"
    if sum(any(c.isdigit() for c in w) for w in words) / len(words) > MAX_DIGIT_SHARE:
        return "mostly numbers"
    if sentence[:1].islower():
        return "starts mid-sentence"
    letters = [c for c in sentence if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > MAX_UPPER_SHARE:
        return "all caps"
    english = sum(w.lower().strip(".,;:!?()\"'“”‘’") in ENGLISH_WORDS for w in words)
    if english / len(words) > MAX_ENGLISH_SHARE:
        return "english words"
    return None

if __name__ == "__main__":
    records = [json.loads(l) for l in open("data/historical_clean/historical_raw.jsonl", encoding="utf-8")]
    cleaned, dropped = [], []
    reasons = Counter()
    for r in records:
        blocks = clean_page(r["text"])
        kept = []
        for b in blocks:
            for sent in sentence_split(b):
                why = reject_reason(sent)
                if why:
                    reasons[why] += 1
                    dropped.append((why, sent))
                else:
                    kept.append(sent)
        r["text_clean"] = "\n\n".join(blocks)
        r["sentences"] = kept
        cleaned.append(r)

    with open("data/historical_clean/historical_clean.jsonl", "w", encoding="utf-8") as f:
        for r in cleaned:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


    os.makedirs("results/filtering", exist_ok=True)
    random.seed(42)
    by_reason = {}
    for why, sent in dropped:
        by_reason.setdefault(why, []).append(sent)

    # Sample each rejection reason separately.
    with open("results/filtering/dropped_sentences_sample.txt", "w", encoding="utf-8") as f:
        for why, sents in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            f.write(f"=== {why} ({len(sents)} sentences dropped) ===\n")
            for sent in random.sample(sents, min(25, len(sents))):
                f.write(sent + "\n")
            f.write("\n")

    n_sents = sum(len(r["sentences"]) for r in cleaned)
    print(f"{len(cleaned)} pages, {n_sents} sentences kept; dropped: {dict(reasons)}")
