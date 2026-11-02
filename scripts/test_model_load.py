# Loads each checkpoint and prints its device, model type, vocabulary size and tokenisation of a short example.
import torch
from load_models import load, MODELS


test_sentence = "Molo, unjani? Ndifuna ukufunda ngembali yakwaXhosa."

for model_key in MODELS:
    print(f"\n--- {model_key} (isiXhosa in pretraining: {MODELS[model_key]['has_xh']}) ---")
    try:
        tok, model, mtype = load(model_key)
        tokens = tok.tokenize(test_sentence)
        print(f"Device: {next(model.parameters()).device}")
        print(f"Model type: {mtype}")
        print(f"Vocab size: {tok.vocab_size}")
        print(f"Tokens ({len(tokens)}): {tokens}")
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"FAILED to load {model_key}: {e}")
