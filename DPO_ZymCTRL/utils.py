import torch
import torch.nn.functional as F

def perplexity_from_logits(logits: torch.Tensor,
                           labels: torch.Tensor,
                           attention_mask: torch.Tensor) -> torch.Tensor:
    """
    logits          : (B, L, V) – raw decoder outputs
    labels          : (B, L)     – token ids (usually same as input_ids i.e. no
                                              need to shift)
    attention_mask  : (B, L)     – 1 for real tokens, 0 for padding

    Returns a scalar perplexity for the whole batch.
    """

    if attention_mask is None:
        attention_mask = torch.ones_like(labels)

    # check for shape normality
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    # GPT-style models predict token t given everything < t,
    # so predictions at position i are compared to label at i+1.
    shift_logits   = logits[:, :-1, :].contiguous()      # (B, L-1, V)
    shift_labels   = labels[:, 1:].contiguous()          # (B, L-1)
    shift_mask     = attention_mask[:, 1:]               # (B, L-1)

    # Step 1: log-probs
    log_probs = F.log_softmax(shift_logits, dim=-1)      # (B, L-1, V)

    # Step 2: gather log-prob of the true token
    #          -> (B, L-1)
    nll = -log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Step 3: mask padding, average, exponentiate
    nll = nll * shift_mask                               # zero-out pads
    mean_nll = nll.sum() / shift_mask.sum()              # average over real tokens
    ppl = torch.exp(mean_nll)

    return ppl