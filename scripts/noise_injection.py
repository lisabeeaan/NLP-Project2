# Adds reproducible synthetic character noise to modern sample sentences and measures the resulting model score changes.
import json, os, random
import pandas as pd
from load_models import load, MODELS
from scoring import score_sentence

N_NOISE = 100
RATES = [0.0, 0.01, 0.02, 0.05]


# These substitutions approximate OCR noise; they are not measured archive errors.
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

modern = [json.loads(l)["text"] for l in open("data/samples/modern_sample.jsonl", encoding="utf-8")][:N_NOISE]

results = []
for model_key in MODELS:
    tok, model, mtype = load(model_key)
    for rate in RATES:
        for i, s in enumerate(modern):
            rng = random.Random(f"{rate}-{i}")
            results.append({"model": model_key, "rate": rate,
                            "score": score_sentence(corrupt(s, rate, rng), tok, model, mtype)})
    del model

os.makedirs("results/noise", exist_ok=True)
df = pd.DataFrame(results)
df.to_json("results/noise/noise_scores.jsonl", orient="records", lines=True, force_ascii=False)
summary = df.groupby(["model", "rate"])["score"].mean().unstack()
summary["change_at_5%"] = summary[0.05] - summary[0.0]
print(summary)
