"""Canonical evaluation prompt(s) threaded through every notebook.

Why this exists
---------------
RLHF only makes sense when you can *see* the model change. Each notebook
in `notebooks/` shows one stage of the pipeline; to make the progression
visible we evaluate every stage on the **same canonical prompt** and the
same small set of comparison prompts.

That way a student can flip between notebooks and read, in order:

    01 (base):  word-salad continuation  →  evidence of "no instruction following"
    02 (SFT):   coherent answer in QA format
    03 (RM):    scalar score that prefers SFT output over base output
    04-06:     further alignment polish

The prompt is intentionally simple and child-targeted — it surfaces the
*format* failure of the base model (continuing with more questions, going
off-topic, etc.) without needing a tricky evaluation rubric.
"""

CANONICAL_PROMPT: str = "Explain quantum computing to a 10-year-old."

# A handful of secondary prompts of different "shapes". Use this list when
# you want a slightly broader picture (instruction / story / factual).
EVAL_PROMPTS: list[str] = [
    CANONICAL_PROMPT,                                     # instruction
    "Why is the sky blue?",                               # factual question
    "Write a haiku about machine learning.",              # creative instruction
]


# ---------------------------------------------------------------------------
# Default checkpoint paths used by the chained pipeline.
#
# The notebooks save adapters to these paths; downstream notebooks check
# whether the path exists and, if so, load the previous stage's checkpoint
# automatically. This keeps each notebook runnable in isolation (falls back
# to the base model) *and* rewards running the full pipeline in order.
# ---------------------------------------------------------------------------

SFT_ADAPTER_PATH: str = "checkpoints/sft/adapter"
REWARD_CHECKPOINT_PATH: str = "checkpoints/reward"
DPO_ADAPTER_PATH: str = "checkpoints/dpo/policy"
PPO_ADAPTER_PATH: str = "checkpoints/ppo/actor"
GRPO_ADAPTER_PATH: str = "checkpoints/grpo/policy"
