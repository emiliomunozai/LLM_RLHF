"""Direct Preference Optimization (DPO).

DPO collapses the two-stage RLHF pipeline (reward model + PPO) into a single
supervised loss on preference pairs. The mathematical insight is that for
the KL-regularised RLHF objective

    max_π  E[ r(x, y) ]  -  β · KL(π || π_ref)

the optimal policy satisfies

    π*(y | x) ∝ π_ref(y | x) · exp( r(x, y) / β ).

Rearranging gives `r(x, y) = β · log( π(y|x) / π_ref(y|x) ) + Z(x)`. Plug
that into the Bradley–Terry preference model and the per-prompt constant
`Z(x)` *cancels*, leaving a loss in terms of the policy alone:

    L_DPO = -E[ log σ( β · ( log π/π_ref [y_w] - log π/π_ref [y_l] ) ) ].

No reward model, no rollouts, no advantage estimation — just gradient
descent on a sigmoid loss. The frozen reference policy `π_ref` is the SFT
checkpoint we started from.
"""
import copy
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

from llm_rlhf.utils import sequence_logprobs


@dataclass
class DPOConfig:
    beta: float = 0.1
    learning_rate: float = 5e-7
    num_epochs: int = 3
    batch_size: int = 4
    max_seq_length: int = 512
    max_grad_norm: float = 1.0
    output_dir: str = "checkpoints/dpo"


class DPOTrainer:
    def __init__(self, sft_model, config: DPOConfig | None = None):
        self.cfg = config or DPOConfig()
        self.tokenizer = sft_model.tokenizer
        self.device = sft_model.base.device

        # Policy = the SFT model (trainable).
        self.policy = sft_model.model

        # Reference = a frozen copy of the policy at training start.
        # The reference must never change during DPO training, otherwise the
        # implicit rewards become inconsistent across steps.
        self.reference = copy.deepcopy(self.policy).eval()
        for p in self.reference.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.cfg.learning_rate)

    # ---- batching ----------------------------------------------------------

    def _tokenize(self, prompt: str, response: str) -> dict:
        return self.tokenizer(
            f"User: {prompt}\n\nAssistant: {response}",
            truncation=True,
            max_length=self.cfg.max_seq_length,
        )

    def _collate(self, pairs: list[tuple[str, str, str]]):
        preferred = [f"User: {p}\n\nAssistant: {w}" for (p, w, _) in pairs]
        dispreferred = [f"User: {p}\n\nAssistant: {l}" for (p, _, l) in pairs]
        return (
            self.tokenizer(preferred, padding=True, truncation=True,
                           max_length=self.cfg.max_seq_length, return_tensors="pt").to(self.device),
            self.tokenizer(dispreferred, padding=True, truncation=True,
                           max_length=self.cfg.max_seq_length, return_tensors="pt").to(self.device),
        )

    # ---- loss --------------------------------------------------------------

    def compute_loss(self, pref_inputs, disp_inputs) -> tuple[torch.Tensor, dict]:
        # Policy log-probs (with gradient).
        pol_pref = sequence_logprobs(
            self.policy(**pref_inputs).logits, pref_inputs["input_ids"], pref_inputs["attention_mask"]
        )
        pol_disp = sequence_logprobs(
            self.policy(**disp_inputs).logits, disp_inputs["input_ids"], disp_inputs["attention_mask"]
        )

        # Reference log-probs (no gradient).
        with torch.no_grad():
            ref_pref = sequence_logprobs(
                self.reference(**pref_inputs).logits, pref_inputs["input_ids"], pref_inputs["attention_mask"]
            )
            ref_disp = sequence_logprobs(
                self.reference(**disp_inputs).logits, disp_inputs["input_ids"], disp_inputs["attention_mask"]
            )

        # Implicit reward = β · ( log π - log π_ref ).
        pref_ratio = pol_pref - ref_pref
        disp_ratio = pol_disp - ref_disp
        logits = self.cfg.beta * (pref_ratio - disp_ratio)
        loss = -F.logsigmoid(logits).mean()

        with torch.no_grad():
            metrics = {
                "loss": loss.item(),
                "accuracy": (pref_ratio > disp_ratio).float().mean().item(),
                "reward_margin": (pref_ratio - disp_ratio).mean().item(),
                "pref_kl": pref_ratio.mean().item(),
                "disp_kl": disp_ratio.mean().item(),
            }
        return loss, metrics

    # ---- training ----------------------------------------------------------

    def train(self, pairs: list[tuple[str, str, str]]) -> None:
        print(f"DPO: {len(pairs)} pairs, {self.cfg.num_epochs} epochs, β={self.cfg.beta}")
        bs = self.cfg.batch_size
        for epoch in range(self.cfg.num_epochs):
            batches = [pairs[i : i + bs] for i in range(0, len(pairs), bs)]
            running = {"loss": 0.0, "accuracy": 0.0, "reward_margin": 0.0}
            pbar = tqdm(batches, desc=f"DPO epoch {epoch+1}/{self.cfg.num_epochs}")
            for batch in pbar:
                pref_inputs, disp_inputs = self._collate(batch)
                self.optimizer.zero_grad()
                loss, metrics = self.compute_loss(pref_inputs, disp_inputs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                for k in running:
                    running[k] += metrics[k]
                pbar.set_postfix({k: f"{metrics[k]:.3f}" for k in running})
            n = max(len(batches), 1)
            print(f"  epoch {epoch+1}: " + ", ".join(f"{k}={v/n:.3f}" for k, v in running.items()))

    # ---- persistence -------------------------------------------------------

    def save(self, path: str | None = None) -> str:
        path = path or self.cfg.output_dir
        os.makedirs(path, exist_ok=True)
        self.policy.save_pretrained(os.path.join(path, "policy"))
        self.tokenizer.save_pretrained(path)
        print(f"Saved DPO policy to {path}")
        return path
