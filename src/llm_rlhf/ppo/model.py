"""Actor-critic wrapper used by the PPO trainer.

Two practical choices for the value (critic) function in PPO-for-LLMs:

1. *Shared backbone, two heads* — keep one transformer and add a small value
   head alongside the LM head. Cheap and the default in TRL.
2. *Two models* — separate actor and critic networks. More expressive but
   doubles GPU memory. The reward model from `reward.py` already gives us
   exactly this kind of scalar-output model, so we reuse it.

We go with option 2 for clarity: each component does one job, and the
RewardModel class is reused so students see how the reward head and the
value head are structurally identical.
"""
import torch
import torch.nn as nn


class PPOModel(nn.Module):
    """Bundles a (LoRA-finetuned) actor and a critic with a scalar value head."""

    def __init__(self, actor: nn.Module, critic: nn.Module):
        super().__init__()
        self.actor = actor      # AutoModelForCausalLM (the SFT policy)
        self.critic = critic    # RewardModel-style scalar predictor — but per-token

    def actor_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.actor(input_ids=input_ids, attention_mask=attention_mask).logits

    def critic_values(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Per-token value predictions.

        We need V(s_t) at every response position, not just one scalar per
        sequence. So we re-implement the forward of `RewardModel` here, but
        keep all the hidden states rather than just the last token's.
        """
        outputs = self.critic.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]          # [B, L, H]
        # Backbone runs in fp16 on GPU but the reward head is fp32 — cast
        # the hidden states to match the head's dtype (same fix as in reward.py).
        last_hidden = last_hidden.to(self.critic.reward_head.weight.dtype)
        return self.critic.reward_head(last_hidden).squeeze(-1)  # [B, L]
