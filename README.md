# How Is Reasoning Built in a Language Model?

One notebook trains a small model (Qwen2.5-0.5B) in three stages and measures,
after each one, how many grade-school math problems (GSM8K) it solves:

1. **Pretraining**: predicting text (already done by the model's authors).
2. **Imitation**: copying solutions written by people.
3. **Reinforcement learning (GRPO)**: learning from its own attempts, rewarded when correct.

## Run

```bash
uv sync --group gpu
```

Open `notebooks/reasoning_from_scratch.ipynb` and run it top to bottom (GPU required). The trained models are included in `notebooks/outputs/`, so it runs in a few minutes; set `RETRAIN = True` to train everything yourself (~40 min).

`notebooks/optional_policy_gradient_intuition.ipynb` is an optional side
lesson on policy gradients in a 2D toy problem.
