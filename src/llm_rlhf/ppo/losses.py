"""PPO loss functions, written as pure tensor ops.

Each function corresponds to a piece of the PPO objective. We keep them
free of any model or optimizer state so they can be read and unit-tested
in isolation. The trainer in `trainer.py` glues them together.

Shapes used throughout:
    B = batch size
    T = response length (number of tokens *generated* by the policy)

For a `prompt + response` sequence of total length L, the response mask
has 1s on the response tokens and 0s on the prompt (and padding).
"""
import torch
import torch.nn.functional as F

from llm_rlhf.utils import masked_mean, masked_whiten


# ---------------------------------------------------------------------------
# KL-penalised per-token reward
# ---------------------------------------------------------------------------
#
# At the end of each rollout we have a scalar reward from the reward model
# (only well-defined on the *last* response token) and per-token KL between
# the current policy and the frozen reference policy. PPO operates on a
# token-level reward signal:
#
#     r_t = -β * KL(π_θ || π_ref)_t,                  t < T-1
#     r_{T-1} = -β * KL(...)_{T-1} + scalar_reward
#
# The KL is estimated, not computed exactly — see Schulman's k_3 estimator.


def kl_estimate(
    log_pi: torch.Tensor,
    log_ref: torch.Tensor,
    method: str = "k_3",
) -> torch.Tensor:
    """Per-token KL estimate between policy and reference distributions.

    `log_pi` and `log_ref` are log-probabilities of the *same* tokens. We can't
    compute KL exactly because we don't store the full per-token distribution,
    only the log-prob of the chosen token. The estimators below are all
    unbiased or consistent.

    * `k_3` — `(r - 1) - log r` where r = exp(log_pi - log_ref). Schulman's
      preferred estimator: always positive, low variance.
    * `abs` — `|log r|`. Cheap and stable, but not unbiased.
    * `mse` — `1/2 (log r)^2`. Common when KL is used as a soft penalty.
    """
    log_ratio = log_pi - log_ref
    if method == "k_3":
        return torch.exp(log_ratio) - 1.0 - log_ratio
    if method == "abs":
        return log_ratio.abs()
    if method == "mse":
        return 0.5 * log_ratio.pow(2)
    raise ValueError(f"Unknown KL estimator: {method!r}")


def compute_rewards_with_kl_penalty(
    scalar_rewards: torch.Tensor,        # [B] — from the reward model
    policy_log_probs: torch.Tensor,      # [B, T]
    reference_log_probs: torch.Tensor,   # [B, T]
    response_mask: torch.Tensor,         # [B, T]
    kl_coef: float = 0.1,
    method: str = "k_3",
    reward_clip: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the per-token reward used by GAE.

    Returns:
        rewards:      [B, T] — KL penalty everywhere + scalar reward on the
                      final response token of each row.
        kl_per_token: [B, T] — KL estimate, for logging.
    """
    kl = kl_estimate(policy_log_probs, reference_log_probs, method) * response_mask
    rewards = -kl_coef * kl

    if reward_clip is not None:
        scalar_rewards = scalar_rewards.clamp(-reward_clip, reward_clip)

    # Place the scalar reward on the last unmasked token of each row.
    last_idx = response_mask.sum(dim=1).long() - 1
    last_idx = last_idx.clamp(min=0)
    row = torch.arange(rewards.size(0), device=rewards.device)
    rewards[row, last_idx] = rewards[row, last_idx] + scalar_rewards

    return rewards, kl


# ---------------------------------------------------------------------------
# Generalised Advantage Estimation (GAE)
# ---------------------------------------------------------------------------
#
#     δ_t = r_t + γ V(s_{t+1}) - V(s_t)
#     A_t = Σ_{l=0..T-1-t} (γ λ)^l δ_{t+l}
#     R_t = A_t + V(s_t)        (used as the regression target for V)
#
# We iterate backwards so each A_t reuses the next step's running sum.


def compute_gae(
    rewards: torch.Tensor,        # [B, T]
    values: torch.Tensor,         # [B, T]
    response_mask: torch.Tensor,  # [B, T]
    gamma: float = 1.0,
    lam: float = 0.95,
    whiten: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised Advantage Estimation. Returns `(advantages, returns)`."""
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)

    with torch.no_grad():
        for t in reversed(range(T)):
            next_value = values[:, t + 1] if t + 1 < T else torch.zeros_like(values[:, 0])
            delta = rewards[:, t] + gamma * next_value - values[:, t]
            last_gae = (delta + gamma * lam * last_gae) * response_mask[:, t]
            advantages[:, t] = last_gae

        returns = advantages + values
        if whiten:
            advantages = masked_whiten(advantages, response_mask)

    return advantages, returns


# ---------------------------------------------------------------------------
# Clipped surrogate policy loss
# ---------------------------------------------------------------------------
#
#     ratio_t = exp(log π_θ(a_t|s_t) - log π_old(a_t|s_t))
#     L = -E[ min( ratio * A, clip(ratio, 1-ε, 1+ε) * A ) ]
#
# We write the negation explicitly (`-A * ratio`) and take a *max*, which is
# equivalent and lets the autograd graph mirror the formula above.


def compute_policy_loss(
    old_log_probs: torch.Tensor,  # [B, T]
    log_probs: torch.Tensor,      # [B, T]
    advantages: torch.Tensor,     # [B, T]
    response_mask: torch.Tensor,  # [B, T]
    clip_epsilon: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    ratio = torch.exp(log_probs - old_log_probs)
    unclipped = -advantages * ratio
    clipped = -advantages * ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    loss = masked_mean(torch.max(unclipped, clipped), response_mask)

    with torch.no_grad():
        clip_frac = masked_mean(
            ((ratio - 1.0).abs() > clip_epsilon).float(), response_mask
        )
        approx_kl = masked_mean(0.5 * (log_probs - old_log_probs).pow(2), response_mask)

    return loss, {"clip_frac": clip_frac.item(), "approx_kl": approx_kl.item()}


# ---------------------------------------------------------------------------
# Clipped value-function loss
# ---------------------------------------------------------------------------
#
# Same clipping idea applied to the value head: prevent each update from
# moving V too far from the value used to compute the advantages. This
# stabilises training in practice but is omitted from the original PPO paper.


def compute_value_loss(
    value_preds: torch.Tensor,    # [B, T]
    returns: torch.Tensor,        # [B, T]
    old_values: torch.Tensor,     # [B, T]
    response_mask: torch.Tensor,  # [B, T]
    clip_epsilon: float = 0.2,
) -> torch.Tensor:
    clipped_preds = old_values + (value_preds - old_values).clamp(-clip_epsilon, clip_epsilon)
    loss_unclipped = (value_preds - returns).pow(2)
    loss_clipped = (clipped_preds - returns).pow(2)
    return 0.5 * masked_mean(torch.max(loss_unclipped, loss_clipped), response_mask)


# ---------------------------------------------------------------------------
# Entropy bonus
# ---------------------------------------------------------------------------
#
# An entropy term added to the loss encourages exploration: a sharper
# (lower-entropy) policy gets penalised, a more uniform one doesn't.


def entropy_from_logits(logits: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    per_token = -(probs * log_probs).sum(dim=-1)  # [B, T]
    return masked_mean(per_token, response_mask)
