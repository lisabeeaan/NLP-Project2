# Calculates subword tokens per whitespace word for each model on the original historical and modern samples.
import json
from transformers import AutoTokenizer
from load_models import MODELS


hist_sample = [json.loads(l)["text"] for l in open("data/samples/hist_sample.jsonl", encoding="utf-8")]
modern_sample = [json.loads(l)["text"] for l in open("data/samples/modern_sample.jsonl", encoding="utf-8")]

def fragmentation_rate(sentences, tokenizer):
    rates = []
    for s in sentences:
        words = s.split()
        if not words:
            continue
        n_subwords = len(tokenizer.tokenize(s))
        rates.append(n_subwords / len(words))
    return rates

results = []
for model_key in MODELS:
    tok = AutoTokenizer.from_pretrained(MODELS[model_key]["name"])
    for label, sents in [("historical", hist_sample), ("modern", modern_sample)]:
        rates = fragmentation_rate(sents, tok)
        for r in rates:
            results.append({"model": model_key, "corpus": label, "fragmentation_rate": r})

with open("results/tokenization/fragmentation.jsonl", "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
