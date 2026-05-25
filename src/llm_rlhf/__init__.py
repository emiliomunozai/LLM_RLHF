"""llm_rlhf — an educational, modular reimplementation of an RLHF pipeline.

The package mirrors the conceptual stages of RLHF so each file maps to one idea:

    model.py    — load a pretrained causal LM (the policy we will align)
    sft.py      — supervised fine-tuning on instruction data (LoRA + Dolly-15k)
    reward.py   — reward model trained on pairwise preferences (Bradley–Terry)
    ppo/        — PPO trainer: KL-penalty rewards, GAE, clipped policy + value
    dpo.py      — Direct Preference Optimization (no reward model, no RL loop)
    grpo.py     — Group Relative Policy Optimization (group-relative advantages)
    utils.py    — masked tensor ops and small helpers shared across modules
    config.py   — TOML config loading

Read the modules in the order above; each one builds on the previous.
"""

from llm_rlhf.model import PretrainedLLM

__all__ = ["PretrainedLLM"]
