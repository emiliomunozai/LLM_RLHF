"""End-to-end RLHF pipeline runner.

Trains each stage on the project's actual configs and captures generations
on EVAL_PROMPTS at every stage. Output is saved as `pipeline_outputs.json`
in the project root, and consumed by the notebooks' 'Progresión esperada'
tables.

Usage (from project root):
    uv run --no-group cpu --group gpu python scripts/run_pipeline.py [STAGE]

STAGE is one of: base, sft, pairs, rm, dpo, ppo, grpo, all (default).
Each stage reads checkpoints written by previous stages, so run them in
order the first time.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset

from llm_rlhf import CANONICAL_PROMPT, EVAL_PROMPTS, PretrainedLLM
from llm_rlhf.config import load_toml
from llm_rlhf.eval import (
    DPO_ADAPTER_PATH,
    GRPO_ADAPTER_PATH,
    PPO_ADAPTER_PATH,
    REWARD_CHECKPOINT_PATH,
    SFT_ADAPTER_PATH,
)
from llm_rlhf.sft import SFTConfig, SupervisedFineTuner

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "pipeline_outputs.json"
PAIRS_PATH = ROOT / "checkpoints" / "preference_pairs.json"


# ---------------------------------------------------------------------------
# Bookkeeping helpers
# ---------------------------------------------------------------------------


def load_results() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {}


def save_results(results: dict) -> None:
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def stage_header(name: str) -> None:
    print()
    print("=" * 70)
    print(f"  STAGE: {name}")
    print("=" * 70, flush=True)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def gen_base_format(llm: PretrainedLLM, prompt: str, max_new_tokens: int = 120) -> str:
    """Generate from the *base* prompt directly (no User/Assistant template).

    Used for the base-model stage so we capture what an unaligned LM does
    with a raw instruction.
    """
    from llm_rlhf.model import GenerationConfig
    return llm.generate(prompt, GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9))


def gen_chat_format(llm: PretrainedLLM, prompt: str, max_new_tokens: int = 120) -> str:
    """Generate using the SFT/chat template: User: ... Assistant: ...

    Used for every post-SFT stage.
    """
    import torch
    formatted = f"User: {prompt}\n\nAssistant: "
    enc = llm.tokenizer(formatted, return_tensors="pt").to(llm.device)
    with torch.no_grad():
        out = llm.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=llm.tokenizer.pad_token_id,
        )
    return llm.tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)


def capture(stage: str, gen_fn, llm: PretrainedLLM) -> dict[str, str]:
    """Run gen_fn on every EVAL_PROMPT and store/print results."""
    print(f"\n--- generations [{stage}] ---")
    out: dict[str, str] = {}
    for p in EVAL_PROMPTS:
        text = gen_fn(llm, p)
        out[p] = text
        print(f"PROMPT:     {p}")
        print(f"COMPLETION: {text}\n", flush=True)
    return out


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def run_base() -> None:
    stage_header("BASE")
    results = load_results()
    llm = PretrainedLLM()
    results["base"] = capture("base", gen_base_format, llm)
    save_results(results)
    del llm
    free_gpu()


def run_sft() -> None:
    stage_header("SFT")
    results = load_results()
    llm = PretrainedLLM()
    cfg = SFTConfig(**load_toml(ROOT / "configs" / "sft.toml"))
    cfg.output_dir = str(ROOT / "checkpoints" / "sft")
    sft = SupervisedFineTuner(llm, cfg)
    sft.prepare_data()
    sft.setup_peft()
    t0 = time.time()
    adapter_path = sft.train()
    print(f"SFT training took {time.time() - t0:.1f}s")
    print(f"Adapter saved to: {adapter_path}")
    # The peft model already wraps the actor; use sft.generate for chat format.
    results["sft"] = {p: sft.generate(p) for p in EVAL_PROMPTS}
    print("\n--- generations [sft] ---")
    for p, t in results["sft"].items():
        print(f"PROMPT:     {p}\nCOMPLETION: {t}\n", flush=True)
    save_results(results)
    del sft, llm
    free_gpu()


def run_pairs(num_prompts: int = 30) -> None:
    """Build preference pairs by sampling 4 responses per prompt with the SFT model."""
    stage_header("PREFERENCE PAIRS")
    from llm_rlhf.reward import RewardConfig, RewardModelTrainer, simple_quality_score

    llm = PretrainedLLM()
    sft_path = ROOT / SFT_ADAPTER_PATH
    if sft_path.exists():
        llm.load_adapter(str(sft_path))
        print(f"Loaded SFT adapter from {sft_path}")
    else:
        print(f"WARNING: no SFT adapter at {sft_path} — sampling from base model.")

    def gen(prompt: str) -> str:
        return gen_chat_format(llm, prompt, max_new_tokens=80)

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    prompts = [r["instruction"] for r in ds.select(range(num_prompts))]
    print(f"Building preference pairs from {len(prompts)} prompts...")

    # Use the trainer just for its preference-pair construction logic.
    rm_cfg = RewardConfig(**load_toml(ROOT / "configs" / "reward.toml"))
    rm_cfg.num_responses_per_prompt = 4
    # Build pairs by hand (avoids instantiating a RewardModel just to call the helper)
    pairs = []
    from tqdm import tqdm
    for p in tqdm(prompts, desc="sampling"):
        responses = [gen(p) for _ in range(rm_cfg.num_responses_per_prompt)]
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                si, sj = simple_quality_score(responses[i]), simple_quality_score(responses[j])
                if si > sj:
                    pairs.append([p, responses[i], responses[j]])
                elif sj > si:
                    pairs.append([p, responses[j], responses[i]])
    print(f"Built {len(pairs)} preference pairs.")

    PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAIRS_PATH.write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote pairs to {PAIRS_PATH}")
    del llm
    free_gpu()


def run_rm() -> None:
    stage_header("REWARD MODEL")
    from llm_rlhf.reward import RewardConfig, RewardModel, RewardModelTrainer

    pairs = [tuple(t) for t in json.loads(PAIRS_PATH.read_text(encoding="utf-8"))]
    print(f"Loaded {len(pairs)} pairs.")

    llm = PretrainedLLM()
    sft_path = ROOT / SFT_ADAPTER_PATH
    if sft_path.exists():
        llm.load_adapter(str(sft_path))

    rm = RewardModel(llm)
    cfg = RewardConfig(**load_toml(ROOT / "configs" / "reward.toml"))
    cfg.output_dir = str(ROOT / REWARD_CHECKPOINT_PATH)
    trainer = RewardModelTrainer(rm, cfg)
    t0 = time.time()
    trainer.train(pairs)
    print(f"RM training took {time.time() - t0:.1f}s")
    trainer.save()

    # Sanity check: does the RM score the SFT output above the base output on canonical?
    results = load_results()
    base_out = results.get("base", {}).get(CANONICAL_PROMPT, "")
    sft_out = results.get("sft", {}).get(CANONICAL_PROMPT, "")
    if base_out and sft_out:
        scores = trainer.score([CANONICAL_PROMPT, CANONICAL_PROMPT], [base_out, sft_out])
        print(f"RM score (base output)  : {scores[0].item():+.3f}")
        print(f"RM score (SFT  output)  : {scores[1].item():+.3f}")
        results.setdefault("rm_scores", {})["canonical_base"] = scores[0].item()
        results["rm_scores"]["canonical_sft"] = scores[1].item()
        save_results(results)

    del trainer, rm, llm
    free_gpu()


def run_dpo() -> None:
    stage_header("DPO")
    from llm_rlhf.dpo import DPOConfig, DPOTrainer

    pairs = [tuple(t) for t in json.loads(PAIRS_PATH.read_text(encoding="utf-8"))]
    print(f"DPO on {len(pairs)} preference pairs.")

    llm = PretrainedLLM()
    sft_path = ROOT / SFT_ADAPTER_PATH
    if sft_path.exists():
        llm.load_adapter(str(sft_path), trainable=True)

    sft_wrapper = SupervisedFineTuner(llm)  # gives us .model + .tokenizer
    sft_wrapper.model = llm.model

    cfg = DPOConfig(**load_toml(ROOT / "configs" / "dpo.toml"))
    cfg.output_dir = str(ROOT / "checkpoints" / "dpo")
    trainer = DPOTrainer(sft_wrapper, cfg)
    t0 = time.time()
    trainer.train(pairs)
    print(f"DPO training took {time.time() - t0:.1f}s")
    trainer.save()

    results = load_results()
    # `trainer.policy` is the trained model. Patch llm.model so gen_chat_format works.
    llm.model = trainer.policy
    results["dpo"] = capture("dpo", gen_chat_format, llm)
    save_results(results)

    del trainer, llm
    free_gpu()


def run_ppo() -> None:
    stage_header("PPO")
    from llm_rlhf.ppo import PPOConfig, PPOTrainer
    from llm_rlhf.reward import RewardConfig, RewardModel, RewardModelTrainer

    # ---- load policy (SFT) ----
    llm = PretrainedLLM()
    sft_path = ROOT / SFT_ADAPTER_PATH
    if sft_path.exists():
        llm.load_adapter(str(sft_path), trainable=True)
    sft_wrapper = SupervisedFineTuner(llm)
    sft_wrapper.model = llm.model

    # ---- load reward model ----
    # The RM was trained on top of an SFT-adapted backbone, so we must
    # reconstruct that same structure before loading the state dict.
    rm_llm = PretrainedLLM()
    if sft_path.exists():
        rm_llm.load_adapter(str(sft_path), trainable=False)
    rm = RewardModel(rm_llm)
    rm_cfg = RewardConfig(**load_toml(ROOT / "configs" / "reward.toml"))
    rm_trainer = RewardModelTrainer(rm, rm_cfg)
    rm_trainer.load(str(ROOT / REWARD_CHECKPOINT_PATH))

    # ---- PPO ----
    cfg = PPOConfig(**load_toml(ROOT / "configs" / "ppo.toml"))
    cfg.output_dir = str(ROOT / "checkpoints" / "ppo")
    ppo = PPOTrainer(sft_wrapper, rm, rm_trainer, cfg)

    # Use a small set of prompts (Dolly slice) for rollouts.
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    prompts = [r["instruction"] for r in ds.select(range(20))]

    t0 = time.time()
    ppo.train(prompts, num_iterations=10, batch_size=4)
    print(f"PPO training took {time.time() - t0:.1f}s")
    ppo.save()

    # Generate. PPO's actor was modified in place inside sft_wrapper.model.
    results = load_results()
    llm.model = ppo.model.actor
    results["ppo"] = capture("ppo", gen_chat_format, llm)
    save_results(results)

    del ppo, rm_trainer, rm, rm_llm, llm
    free_gpu()


def run_grpo() -> None:
    stage_header("GRPO")
    from llm_rlhf.grpo import GRPOConfig, GRPOTrainer
    from llm_rlhf.reward import RewardConfig, RewardModel, RewardModelTrainer

    llm = PretrainedLLM()
    sft_path = ROOT / SFT_ADAPTER_PATH
    if sft_path.exists():
        llm.load_adapter(str(sft_path), trainable=True)
    sft_wrapper = SupervisedFineTuner(llm)
    sft_wrapper.model = llm.model

    rm_llm = PretrainedLLM()
    if sft_path.exists():
        rm_llm.load_adapter(str(sft_path), trainable=False)
    rm = RewardModel(rm_llm)
    rm_cfg = RewardConfig(**load_toml(ROOT / "configs" / "reward.toml"))
    rm_trainer = RewardModelTrainer(rm, rm_cfg)
    rm_trainer.load(str(ROOT / REWARD_CHECKPOINT_PATH))

    cfg = GRPOConfig(**load_toml(ROOT / "configs" / "grpo.toml"))
    cfg.output_dir = str(ROOT / "checkpoints" / "grpo")
    # GRPO's `_step` concatenates tensors across prompts in a batch; that
    # only works when their padded sequence lengths match. The educational
    # version of `grpo.py` doesn't pad across prompts, so we run one prompt
    # per iteration (compensated by bumping num_iterations).
    cfg.num_prompts_per_iter = 1
    cfg.num_iterations = max(cfg.num_iterations, 20)
    grpo = GRPOTrainer(sft_wrapper, rm_trainer, cfg)

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    prompts = [r["instruction"] for r in ds.select(range(20))]

    t0 = time.time()
    grpo.train(prompts)
    print(f"GRPO training took {time.time() - t0:.1f}s")
    grpo.save()

    results = load_results()
    llm.model = grpo.policy
    results["grpo"] = capture("grpo", gen_chat_format, llm)
    save_results(results)

    del grpo, rm_trainer, rm, rm_llm, llm
    free_gpu()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STAGES = {
    "base": run_base,
    "sft":  run_sft,
    "pairs": run_pairs,
    "rm":   run_rm,
    "dpo":  run_dpo,
    "ppo":  run_ppo,
    "grpo": run_grpo,
}


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "all":
        order = ["base", "sft", "pairs", "rm", "dpo", "ppo", "grpo"]
    else:
        order = [arg]
    for stage in order:
        if stage not in STAGES:
            raise SystemExit(f"unknown stage: {stage!r} (choose from {list(STAGES)})")
        STAGES[stage]()
    print("\n=== pipeline complete ===")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
