# Defines an unused helper for masked-subword top-1 and top-k accuracy without running an evaluation.
import torch

@torch.no_grad()
def top_k_accuracy(sentence, tokenizer, model, k=5):
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"][0].to(model.device)
    mask_id = tokenizer.mask_token_id
    correct_top1, correct_topk, total = 0, 0, 0
    for i in range(1, len(input_ids) - 1):
        masked = input_ids.clone()
        true_token = masked[i].item()
        masked[i] = mask_id
        out = model(masked.unsqueeze(0))
        topk = torch.topk(out.logits[0, i], k).indices.tolist()
        if true_token == topk[0]:
            correct_top1 += 1
        if true_token in topk:
            correct_topk += 1
        total += 1
    return correct_top1 / total, correct_topk / total
