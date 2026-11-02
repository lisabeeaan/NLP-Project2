# Scores the original historical and modern samples with every available model and saves the exploratory results.
import json, os, time
import torch
from load_models import load, MODELS
from scoring import score_sentence

N_RUN = None


hist_sample = [json.loads(l)["text"] for l in open("data/samples/hist_sample.jsonl", encoding="utf-8")][:N_RUN]
modern_sample = [json.loads(l)["text"] for l in open("data/samples/modern_sample.jsonl", encoding="utf-8")][:N_RUN]
os.makedirs("results/perplexity", exist_ok=True)

results = []
for model_key in MODELS:
    try:
        tok, model, mtype = load(model_key)
    except Exception as e:
        print(f"Skipping {model_key}: {e}")
        continue
    t0 = time.time()
    for label, sents in [("historical", hist_sample), ("modern", modern_sample)]:
        for s in sents:
            try:
                score = score_sentence(s, tok, model, mtype)
                results.append({"model": model_key, "has_xh": MODELS[model_key]["has_xh"],
                                "corpus": label, "sentence": s, "score": score})
            except Exception as e:
                print(f"Skipped a sentence ({model_key}, {label}): {e}")
    del model
    torch.cuda.empty_cache()
    print(f"{model_key} finished in {(time.time() - t0) / 60:.1f} min", flush=True)


    # Save completed results after each model.
    with open("results/perplexity/scores.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
