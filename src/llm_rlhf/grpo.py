"""Group Relative Policy Optimization (GRPO).

GRPO (Shao et al., 2024) is the algorithm behind DeepSeek-R1. It keeps PPO's
clipped surrogate objective but throws away the value function. Instead of
estimating advantages with `A = Q - V`, it estimates them *relative to the
group*:

    For each prompt x, sample G responses {y_1, ..., y_G}.
    Score them with the reward model: r_1, ..., r_G.
    Advantage of response i = (r_i - mean(r)) / std(r).

The advantage is shared across every token of response `y_i`. We then
optimise the same PPO clipped objective using this group-relative advantage,
plus a KL penalty against a frozen reference policy.

Why does this work? In language tasks the *value function* is hard to learn
well (sparse, delayed rewards over long sequences). Replacing it with a
group baseline eliminates that source of error, at the cost of needing to
sample G ≥ 2 responses per prompt every step.
"""
import copy
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

from llm_rlhf.utils import masked_mean, per_token_logprobs


@dataclass
class GRPOConfig:
    group_size: int = 4
    beta: float = 0.04            # KL coefficient
    clip_epsilon: float = 0.2
    learning_rate: float = 5e-7
    num_iterations: int = 10
    num_prompts_per_iter: int = 4
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    max_grad_norm: float = 1.0
    output_dir: str = "checkpoints/grpo"


class GRPOTrainer:
    def __init__(self, sft_model, reward_scorer, config: GRPOConfig | None = None):
        self.cfg = config or GRPOConfig()
        self.tokenizer = sft_model.tokenizer
        self.device = sft_model.base.device

        self.policy = sft_model.model
        self.reward_scorer = reward_scorer

        self.reference = copy.deepcopy(self.policy).eval()
        for p in self.reference.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.cfg.learning_rate)

    # ---- rollout -----------------------------------------------------------

    @torch.no_grad()
    def _sample_group(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        formatted = f"User: {prompt}\n\nAssistant: "
        enc = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        prompt_len = enc["input_ids"].size(1)

        # Generate G responses by repeating the prompt.
        repeated = {k: v.expand(self.cfg.group_size, -1) for k, v in enc.items()}
        gen = self.policy.generate(
            **repeated,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        full_mask = (gen != self.tokenizer.pad_token_id).long()
        response_mask = full_mask.clone()
        response_mask[:, :prompt_len] = 0

        responses = [
            self.tokenizer.decode(gen[i, prompt_len:], skip_special_tokens=True)
            for i in range(gen.size(0))
        ]
        return gen, full_mask, response_mask.float(), responses

    # ---- group-relative advantages ----------------------------------------

    @staticmethod
    def group_relative_advantage(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Normalise rewards within the group: (r - mean) / std."""
        return (rewards - rewards.mean()) / (rewards.std() + eps)

    # ---- one update step --------------------------------------------------

    def _step(self, prompts: list[str]) -> dict:
        """Sample one group per prompt, then do a single gradient step."""
        # Accumulate everything as flat tensors of shape [B, L]/[B, T] where
        # B = num_prompts * group_size.
        all_ids, all_mask, all_resp_mask = [], [], []
        all_old_logprobs, all_ref_logprobs = [], []
        all_advantages = []

        for prompt in prompts:
            full_ids, full_mask, response_mask, responses = self._sample_group(prompt)

            rewards = self.reward_scorer.score([prompt] * self.cfg.group_size, responses).to(self.device)
            advantages = self.group_relative_advantage(rewards)  # [G]

            with torch.no_grad():
                old_logits = self.policy(input_ids=full_ids, attention_mask=full_mask).logits
                ref_logits = self.reference(input_ids=full_ids, attention_mask=full_mask).logits
                old_lp = per_token_logprobs(old_logits, full_ids)
                ref_lp = per_token_logprobs(ref_logits, full_ids)

            all_ids.append(full_ids)
            all_mask.append(full_mask)
            all_resp_mask.append(response_mask[:, 1:])  # shifted to match logprobs
            all_old_logprobs.append(old_lp)
            all_ref_logprobs.append(ref_lp)
            # broadcast scalar advantage over response tokens
            all_advantages.append(advantages[:, None].expand_as(old_lp))

        # Pad/concatenate naively — we assume equal max_new_tokens so shapes match.
        full_ids = torch.cat(all_ids, dim=0)
        full_mask = torch.cat(all_mask, dim=0)
        resp_mask = torch.cat(all_resp_mask, dim=0)
        old_logprobs = torch.cat(all_old_logprobs, dim=0)
        ref_logprobs = torch.cat(all_ref_logprobs, dim=0)
        advantages = torch.cat(all_advantages, dim=0)

        # Recompute log-probs with grad enabled.
        logits = self.policy(input_ids=full_ids, attention_mask=full_mask).logits
        new_logprobs = per_token_logprobs(logits, full_ids)

        ratio = torch.exp(new_logprobs - old_logprobs)
        unclipped = ratio * advantages
        clipped = ratio.clamp(1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * advantages
        policy_loss = -masked_mean(torch.min(unclipped, clipped), resp_mask)

        # KL penalty between policy and reference (k_3 estimator).
        log_ratio = new_logprobs - ref_logprobs
        kl = torch.exp(log_ratio) - 1.0 - log_ratio
        kl_loss = masked_mean(kl, resp_mask)

        loss = policy_loss + self.cfg.beta * kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "kl": kl_loss.item(),
            "advantage_abs_mean": advantages.abs().mean().item(),
        }

    # ---- outer loop --------------------------------------------------------

    def train(self, prompts: list[str]) -> None:
        n = self.cfg.num_prompts_per_iter
        for it in tqdm(range(self.cfg.num_iterations), desc="GRPO iterations"):
            offset = (it * n) % max(len(prompts), 1)
            batch = prompts[offset : offset + n]
            if len(batch) < n:
                batch = (batch + prompts)[:n]
            stats = self._step(batch)
            print(f"[iter {it}] " + ", ".join(f"{k}={v:.4f}" for k, v in stats.items()))

    def save(self, path: str | None = None) -> str:
        path = path or self.cfg.output_dir
        os.makedirs(path, exist_ok=True)
        self.policy.save_pretrained(os.path.join(path, "policy"))
        self.tokenizer.save_pretrained(path)
        print(f"Saved GRPO policy to {path}")
        return path
