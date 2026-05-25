"""PPO trainer — the outer rollout / update loop.

The training loop is conceptually three stages:

    1. ROLLOUT    Sample responses from the current policy on a batch of prompts.
                  Record old log-probs, old values, the reward-model score, and
                  the response mask (which tokens are part of the response).
    2. COMPUTE    Turn the scalar reward into a per-token reward (KL penalty
                  applied to the policy ↔ reference divergence). Use GAE to get
                  advantages and returns.
    3. UPDATE     For `ppo_epochs` mini-epochs, recompute log-probs/values with
                  the current policy and apply clipped policy + value losses.

Why mini-epochs? PPO is "off-policy within a small window": we sample once,
then squeeze multiple gradient updates out of the same rollout while
clipping prevents us from drifting too far from the old policy.

This trainer is written for clarity, not throughput. A production PPO loop
adds: rollout batching, distributed actor / critic, response-length
penalties, KL controller, and reward whitening — all worthwhile extensions
once the educational version is understood.
"""
import copy
import os
from dataclasses import dataclass

import torch
from tqdm import tqdm

from llm_rlhf.ppo.losses import (
    compute_gae,
    compute_policy_loss,
    compute_rewards_with_kl_penalty,
    compute_value_loss,
    entropy_from_logits,
)
from llm_rlhf.ppo.model import PPOModel
from llm_rlhf.utils import per_token_logprobs


@dataclass
class PPOConfig:
    # Rollout
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    # Reward shaping
    kl_coef: float = 0.1
    kl_method: str = "k_3"
    reward_clip: float | None = 5.0
    # GAE
    gamma: float = 1.0
    lam: float = 0.95
    # Optimisation
    learning_rate: float = 1e-6
    ppo_epochs: int = 4
    mini_batch_size: int = 2
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    output_dir: str = "checkpoints/ppo"


class PPOTrainer:
    def __init__(
        self,
        sft_model,                # SupervisedFineTuner (gives us actor + tokenizer + device)
        reward_model,             # RewardModel (used as the value/critic backbone too)
        reward_scorer,            # RewardModelTrainer (has .score(prompts, responses))
        config: PPOConfig | None = None,
    ):
        self.cfg = config or PPOConfig()
        self.tokenizer = sft_model.tokenizer
        self.device = sft_model.base.device

        # Actor = SFT policy. Critic = a *copy* of the reward model used as a
        # value head. The reward model itself stays frozen and is queried via
        # reward_scorer.
        critic = copy.deepcopy(reward_model)
        self.model = PPOModel(actor=sft_model.model, critic=critic).to(self.device)
        self.reward_scorer = reward_scorer

        # Frozen reference policy for KL penalty. Sharing the actor's
        # initial weights is the standard choice.
        self.reference = copy.deepcopy(sft_model.model).eval()
        for p in self.reference.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.cfg.learning_rate,
        )

    # ------------------------------------------------------------------ rollout

    @torch.no_grad()
    def rollout(self, prompts: list[str]) -> dict:
        """Sample one response per prompt and pre-compute everything we need.

        Returns a dict of tensors aligned to `[B, L]` (prompt+response) and
        the response mask aligned to `[B, T]` (response only).
        """
        self.model.eval()
        formatted = [f"User: {p}\n\nAssistant: " for p in prompts]
        enc = self.tokenizer(formatted, return_tensors="pt", padding=True).to(self.device)
        prompt_lens = enc["attention_mask"].sum(dim=1)

        gen = self.model.actor.generate(
            **enc,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # full = prompt + response, padded to the longest in the batch
        full_ids = gen
        full_mask = (full_ids != self.tokenizer.pad_token_id).long()

        # response_mask = 1 only on generated tokens
        response_mask = full_mask.clone()
        for i, plen in enumerate(prompt_lens):
            response_mask[i, :plen] = 0

        # log-probs and values under the (current) actor and reference
        old_logits = self.model.actor_logits(full_ids, full_mask)
        ref_logits = self.reference(input_ids=full_ids, attention_mask=full_mask).logits
        old_values = self.model.critic_values(full_ids, full_mask)

        old_logprobs = per_token_logprobs(old_logits, full_ids)
        ref_logprobs = per_token_logprobs(ref_logits, full_ids)

        # We work on the shifted (predicted-token) axis from here on. Slice
        # the response mask to match.
        shifted_response_mask = response_mask[:, 1:].float()
        shifted_old_values = old_values[:, :-1]

        # Scalar reward from the reward model
        responses_text = []
        for i in range(full_ids.size(0)):
            plen = int(prompt_lens[i])
            resp_ids = full_ids[i, plen:]
            responses_text.append(self.tokenizer.decode(resp_ids, skip_special_tokens=True))
        scalar_rewards = self.reward_scorer.score(prompts, responses_text).to(self.device)

        return {
            "full_ids": full_ids,
            "full_mask": full_mask,
            "response_mask": shifted_response_mask,
            "old_logprobs": old_logprobs,
            "ref_logprobs": ref_logprobs,
            "old_values": shifted_old_values,
            "scalar_rewards": scalar_rewards,
            "responses_text": responses_text,
        }

    # ------------------------------------------------------------------ update

    def update(self, rollout: dict) -> dict:
        """Run `ppo_epochs` mini-epochs over the rollout."""
        rewards_per_token, kl_per_token = compute_rewards_with_kl_penalty(
            scalar_rewards=rollout["scalar_rewards"],
            policy_log_probs=rollout["old_logprobs"],
            reference_log_probs=rollout["ref_logprobs"],
            response_mask=rollout["response_mask"],
            kl_coef=self.cfg.kl_coef,
            method=self.cfg.kl_method,
            reward_clip=self.cfg.reward_clip,
        )

        advantages, returns = compute_gae(
            rewards=rewards_per_token,
            values=rollout["old_values"],
            response_mask=rollout["response_mask"],
            gamma=self.cfg.gamma,
            lam=self.cfg.lam,
        )

        B = rollout["full_ids"].size(0)
        idxs = torch.randperm(B, device=self.device)
        bs = self.cfg.mini_batch_size

        running = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                   "clip_frac": 0.0, "approx_kl": 0.0}
        n_steps = 0

        self.model.train()
        for _ in range(self.cfg.ppo_epochs):
            for start in range(0, B, bs):
                mb = idxs[start : start + bs]
                full_ids = rollout["full_ids"][mb]
                full_mask = rollout["full_mask"][mb]
                resp_mask = rollout["response_mask"][mb]

                logits = self.model.actor_logits(full_ids, full_mask)
                values = self.model.critic_values(full_ids, full_mask)
                new_logprobs = per_token_logprobs(logits, full_ids)
                new_values = values[:, :-1]

                policy_loss, policy_stats = compute_policy_loss(
                    old_log_probs=rollout["old_logprobs"][mb],
                    log_probs=new_logprobs,
                    advantages=advantages[mb],
                    response_mask=resp_mask,
                    clip_epsilon=self.cfg.clip_epsilon,
                )
                value_loss = compute_value_loss(
                    value_preds=new_values,
                    returns=returns[mb],
                    old_values=rollout["old_values"][mb],
                    response_mask=resp_mask,
                    clip_epsilon=self.cfg.clip_epsilon,
                )
                entropy = entropy_from_logits(logits[:, :-1, :], resp_mask)

                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                running["policy_loss"] += policy_loss.item()
                running["value_loss"] += value_loss.item()
                running["entropy"] += entropy.item()
                running["clip_frac"] += policy_stats["clip_frac"]
                running["approx_kl"] += policy_stats["approx_kl"]
                n_steps += 1

        return {k: v / max(n_steps, 1) for k, v in running.items()} | {
            "mean_reward": rollout["scalar_rewards"].mean().item(),
            "mean_kl_per_token": (kl_per_token.sum() / rollout["response_mask"].sum().clamp(min=1)).item(),
        }

    # ------------------------------------------------------------------ outer

    def train(self, prompts: list[str], num_iterations: int = 10, batch_size: int = 4) -> None:
        for it in tqdm(range(num_iterations), desc="PPO iterations"):
            batch = prompts[(it * batch_size) % len(prompts) : (it * batch_size) % len(prompts) + batch_size]
            if len(batch) < batch_size:
                batch = (batch + prompts)[:batch_size]
            rollout = self.rollout(batch)
            stats = self.update(rollout)
            print(f"[iter {it}] " + ", ".join(f"{k}={v:.4f}" for k, v in stats.items()))

    def save(self, path: str | None = None) -> str:
        path = path or self.cfg.output_dir
        os.makedirs(path, exist_ok=True)
        self.model.actor.save_pretrained(os.path.join(path, "actor"))
        torch.save(self.model.critic.state_dict(), os.path.join(path, "critic.pt"))
        self.tokenizer.save_pretrained(path)
        print(f"Saved PPO model to {path}")
        return path
