# LLM_RLHF — Modular RLHF From Scratch

An educational re-implementation of the full RLHF pipeline on small language models. The companion `tutorial.ipynb` provides the written explanations and history; this repo turns the same ideas into a clean, modular codebase you can read, run, and extend.

## Pipeline

```
PretrainedLLM  ─►  SFT  ─►  Reward Model  ─►  PPO  ──► aligned policy
                                              │
                                              ├─►  DPO  (no reward model, no RL)
                                              │
                                              └─►  GRPO (group-relative advantages)
```

Each stage lives in one file (PPO is split into three) so you can read any
one of them top-to-bottom without jumping around.

## Layout

```
configs/                TOML configs — one per stage, plus default.toml
src/llm_rlhf/
    model.py            PretrainedLLM        — HF causal LM wrapper
    sft.py              SupervisedFineTuner  — LoRA on Dolly-15k
    reward.py           RewardModel + Trainer — Bradley–Terry pairwise loss
    ppo/
        losses.py       KL penalty, GAE, clipped policy / value, entropy
        model.py        PPOModel — actor (policy) + critic (value head)
        trainer.py      PPOTrainer — rollout / GAE / clipped update loop
    dpo.py              DPOTrainer  — sigmoid loss on preference pairs
    grpo.py             GRPOTrainer — group-relative advantages
    utils.py            masked_mean, masked_whiten, log-prob helpers
    config.py           tomllib-based config loader
notebooks/
    00_walkthrough.ipynb  guided tour that maps tutorial → modules
tutorial.ipynb          original tutorial with theory + worked examples
```

## Reading order

If you're new to RLHF, read the modules in this order — each builds on the
previous, and the docstring at the top of each file motivates the math:

1. `model.py` — the base policy we will align
2. `sft.py` — turn an autocomplete model into an instruction-follower
3. `reward.py` — learn a scalar "humans prefer this" function
4. `ppo/losses.py` then `ppo/trainer.py` — RL with a learned reward
5. `dpo.py` — derive the same goal without RL or a reward model
6. `grpo.py` — replace PPO's value function with a group baseline

## Setup

We use [`uv`](https://docs.astral.sh/uv/) exclusively (no `pip`).

```bash
uv sync                          # create venv + install deps
uv run python -c "from llm_rlhf import PretrainedLLM; PretrainedLLM()"
```

GPU is recommended but not required. The default model
(`facebook/opt-350m`, 350M params) fits in 4 GB of VRAM and runs on CPU
for experimentation, just slowly.

## Running a stage

Every trainer takes a dataclass config; the TOML files in `configs/` mirror
those fields one-to-one:

```python
from llm_rlhf import PretrainedLLM
from llm_rlhf.sft import SupervisedFineTuner, SFTConfig
from llm_rlhf.config import load_toml

llm = PretrainedLLM()
sft = SupervisedFineTuner(llm, SFTConfig(**load_toml("configs/sft.toml")))
sft.prepare_data()
sft.setup_peft()
sft.train()
```

See `notebooks/00_walkthrough.ipynb` for the full pipeline end-to-end.

## What this codebase is *not*

* A production RLHF library. For that, use [TRL](https://github.com/huggingface/trl) or [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF).
* A benchmark of which algorithm is "best". The point is to make each algorithm legible.

The single source-of-truth for *why* any line of code looks the way it does
is the docstring at the top of its module, followed by the explanations in
`tutorial.ipynb`.
