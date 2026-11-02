# Provides the original masked-model mean pseudo-log-likelihood and causal-model perplexity scoring functions.
import torch, math

@torch.no_grad()
def pseudo_log_likelihood(sentence, tokenizer, model):
    """Return mean log probability from masking one subword at a time."""
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"][0].to(model.device)
    mask_id = tokenizer.mask_token_id
    total_logprob = 0.0
    n_tokens = 0
    for i in range(1, len(input_ids) - 1):
        masked = input_ids.clone()
        true_token = masked[i].item()
        masked[i] = mask_id
        out = model(masked.unsqueeze(0))
        logits = out.logits[0, i]
        logprobs = torch.log_softmax(logits, dim=-1)
        total_logprob += logprobs[true_token].item()
        n_tokens += 1
    return total_logprob / max(n_tokens, 1)

@torch.no_grad()
def causal_perplexity(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    out = model(input_ids, labels=input_ids)
    return math.exp(out.loss.item())

def score_sentence(sentence, tokenizer, model, model_type):
    if model_type == "mlm":
        return pseudo_log_likelihood(sentence, tokenizer, model)
    else:
        return causal_perplexity(sentence, tokenizer, model)
