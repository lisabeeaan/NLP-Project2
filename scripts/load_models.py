# Defines the checkpoint configuration and provides a loader that places each model on the available device in evaluation mode.
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# has_xh records documented isiXhosa training coverage.
MODELS = {
    "mbert":     {"name": "bert-base-multilingual-cased", "type": "mlm", "has_xh": False},
    "afriberta": {"name": "castorini/afriberta_large",     "type": "mlm", "has_xh": False},
    "xlmr":      {"name": "FacebookAI/xlm-roberta-large",  "type": "mlm", "has_xh": True},
    "afroxlmr":  {"name": "Davlan/afro-xlmr-large",        "type": "mlm", "has_xh": True},
    "inkubalm":  {"name": "lelapa/InkubaLM-0.4B",          "type": "clm", "has_xh": True},
}

def load(model_key):
    cfg = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(cfg["name"])
    if cfg["type"] == "mlm":
        model = AutoModelForMaskedLM.from_pretrained(cfg["name"])
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg["name"])
    model.to(DEVICE)
    model.eval()
    return tok, model, cfg["type"]
