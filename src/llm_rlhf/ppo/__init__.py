"""PPO for language models.

Read in this order:

    losses.py   — pure functions: KL penalty, GAE, clipped policy/value, entropy
    model.py    — actor-critic wrapper (policy + value head)
    trainer.py  — the outer loop that rolls out responses and applies the losses

PPO is the most fiddly piece of the pipeline. Splitting the math (`losses.py`)
from the orchestration (`trainer.py`) makes each part testable in isolation.
"""
from llm_rlhf.ppo.losses import (
    compute_gae,
    compute_policy_loss,
    compute_rewards_with_kl_penalty,
    compute_value_loss,
    entropy_from_logits,
)
from llm_rlhf.ppo.model import PPOModel
from llm_rlhf.ppo.trainer import PPOConfig, PPOTrainer

__all__ = [
    "PPOConfig",
    "PPOModel",
    "PPOTrainer",
    "compute_gae",
    "compute_policy_loss",
    "compute_rewards_with_kl_penalty",
    "compute_value_loss",
    "entropy_from_logits",
]
