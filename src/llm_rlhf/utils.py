"""Small tensor utilities shared by PPO, DPO and GRPO.

Most RLHF objectives are computed over *response* tokens only, not the full
prompt+response sequence. That makes masked reductions ubiquitous: every loss
and metric in this codebase boils down to "average this quantity over the
response tokens, ignoring padding and the prompt".
"""
import torch
import torch.nn.functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Mean of `values` over positions where `mask == 1`.

    Equivalent to `(values * mask).sum() / mask.sum()` but along the given dim.
    """
    if dim is None:
        return (values * mask).sum() / mask.sum().clamp(min=1)
    return (values * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1)


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Zero-mean, unit-variance normalization over masked positions.

    Used by GAE to stabilise advantages: the policy loss is sensitive to the
    *scale* of the advantage, so we whiten across the batch.
    """
    mean = masked_mean(values, mask)
    var = masked_mean((values - mean) ** 2, mask)
    return (values - mean) * torch.rsqrt(var + eps)


def sequence_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Sum of per-token log-probabilities for each sequence in a batch.

    Causal LMs predict token `t+1` from positions `<= t`, so we shift logits
    left and labels right before gathering. The returned tensor has shape
    `[batch_size]` — one scalar log-prob per sequence (used by DPO/GRPO).
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous().float()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * shift_mask).sum(dim=-1)


def per_token_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Per-token log-probabilities aligned to predicted tokens.

    Returns shape `[batch_size, seq_len - 1]` — needed by PPO, which scores
    individual tokens rather than whole sequences.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    return log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
