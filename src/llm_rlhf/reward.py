"""Reward model — a learned proxy for human preferences.

Training a reward model is the second stage of classical RLHF (the first
being SFT). The idea:

1. Take the SFT model and sample multiple responses per prompt.
2. Have humans (here, a simple heuristic — see `simple_quality_score`) rank
   pairs of responses: `y_w` (preferred) vs `y_l` (dispreferred).
3. Train a small head on top of the LM backbone that outputs a *scalar*
   score r(prompt, response). The objective is the Bradley–Terry log-loss:

        L = -E[ log σ( r(y_w) - r(y_l) ) ]

   This drives the model to assign higher scores to preferred completions.

The trained reward model is then used by PPO and GRPO as the reward signal.
DPO sidesteps this stage entirely by re-parameterising the reward in terms
of the policy itself.
"""
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from llm_rlhf.model import PretrainedLLM


@dataclass
class RewardConfig:
    learning_rate: float = 1e-5
    num_epochs: int = 2
    batch_size: int = 8
    max_seq_length: int = 512
    num_responses_per_prompt: int = 4
    output_dir: str = "checkpoints/reward"


def simple_quality_score(response: str) -> float:
    """Stand-in for a human annotator.

    Real RLHF datasets cost six figures to label; for an educational pipeline
    we approximate "people prefer engaging, moderate-length, conversational
    responses" with a handful of cheap heuristics. The exact rules matter
    less than the *signal* they create — the reward model only needs *some*
    consistent preference structure to learn from.
    """
    score = 0.0
    if "?" in response:
        score += 2
    for word in ("imagine", "think", "consider", "hey", "wow", "amazing"):
        if word in response.lower():
            score += 1
    for word in ("like", "similar", "imagine", "think of"):
        if word in response.lower():
            score += 1
    length = len(response.split())
    if 20 <= length <= 100:
        score += 1
    elif length < 10:
        score -= 2
    return score


class RewardModel(nn.Module):
    """Causal LM backbone with the LM head replaced by a scalar reward head.

    We take the hidden state of the *last non-padding token* of the
    `prompt+response` sequence and project it to a single scalar. Using the
    last token is the convention in RLHF — it gives the reward head a
    representation that has attended over the entire sequence.
    """

    def __init__(self, base: PretrainedLLM):
        super().__init__()
        self.device = base.device
        self.tokenizer = base.tokenizer
        self.backbone = base.model
        # Drop the language-modelling head; we no longer predict tokens.
        if hasattr(self.backbone, "lm_head"):
            self.backbone.lm_head = nn.Identity()

        # OPT-350m projects the final hidden state from `hidden_size=1024` down
        # to `word_embed_proj_dim=512` before the LM head, so `config.hidden_size`
        # is *not* the actual output dim of `hidden_states[-1]`. Determine the
        # real dim with a dummy forward pass.
        hidden_size = self._infer_hidden_dim()
        self.reward_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.reward_head.weight, std=0.01)
        nn.init.zeros_(self.reward_head.bias)

        self.to(self.device)
        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"Reward model: {n_params:.1f}M parameters")

    @torch.no_grad()
    def _infer_hidden_dim(self) -> int:
        dummy_ids = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        dummy_mask = torch.ones(1, 1, dtype=torch.long, device=self.device)
        out = self.backbone(
            input_ids=dummy_ids, attention_mask=dummy_mask, output_hidden_states=True
        )
        return out.hidden_states[-1].size(-1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]
        # Pick the hidden state of the final non-pad token in each row.
        last_idx = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        pooled = last_hidden[batch_idx, last_idx]
        # Backbone runs in fp16 on GPU but the reward head is fp32 — cast
        # the pooled hidden state to match the head's dtype.
        pooled = pooled.to(self.reward_head.weight.dtype)
        return self.reward_head(pooled).squeeze(-1)


class RewardModelTrainer:
    def __init__(self, model: RewardModel, config: RewardConfig | None = None):
        self.model = model
        self.cfg = config or RewardConfig()
        self.tokenizer = model.tokenizer
        self.device = model.device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.learning_rate)

    # ---- preference data ---------------------------------------------------

    def create_preference_dataset(
        self,
        sft_generate,
        prompts: list[str],
        scorer=simple_quality_score,
    ) -> list[tuple[str, str, str]]:
        """Sample many responses per prompt, then label pairs by `scorer`.

        `sft_generate` is any callable `prompt -> str` (e.g.
        `SupervisedFineTuner.generate`). Decoupling the generator from the
        trainer lets us reuse this method with any policy.
        """
        pairs = []
        for prompt in tqdm(prompts, desc="Sampling preferences"):
            responses = [sft_generate(prompt) for _ in range(self.cfg.num_responses_per_prompt)]
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    si, sj = scorer(responses[i]), scorer(responses[j])
                    if si > sj:
                        pairs.append((prompt, responses[i], responses[j]))
                    elif sj > si:
                        pairs.append((prompt, responses[j], responses[i]))
        print(f"Built {len(pairs)} preference pairs")
        return pairs

    # ---- training ----------------------------------------------------------

    def _tokenize_pair_batch(self, batch):
        def tokenize(texts):
            return self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_seq_length,
                return_tensors="pt",
            ).to(self.device)

        preferred = [f"User: {p}\n\nAssistant: {w}" for (p, w, _) in batch]
        dispreferred = [f"User: {p}\n\nAssistant: {l}" for (p, _, l) in batch]
        return tokenize(preferred), tokenize(dispreferred)

    @staticmethod
    def bradley_terry_loss(r_pref: torch.Tensor, r_disp: torch.Tensor) -> torch.Tensor:
        return -F.logsigmoid(r_pref - r_disp).mean()

    def train_step(self, batch):
        self.optimizer.zero_grad()
        pref_inputs, disp_inputs = self._tokenize_pair_batch(batch)
        r_pref = self.model(pref_inputs["input_ids"], pref_inputs["attention_mask"])
        r_disp = self.model(disp_inputs["input_ids"], disp_inputs["attention_mask"])
        loss = self.bradley_terry_loss(r_pref, r_disp)
        loss.backward()
        self.optimizer.step()
        return {
            "loss": loss.item(),
            "accuracy": (r_pref > r_disp).float().mean().item(),
            "reward_gap": (r_pref - r_disp).mean().item(),
        }

    def train(self, pairs: list[tuple[str, str, str]]):
        print(f"Training reward model for {self.cfg.num_epochs} epochs on {len(pairs)} pairs")
        bs = self.cfg.batch_size
        for epoch in range(self.cfg.num_epochs):
            running = {"loss": 0.0, "accuracy": 0.0, "reward_gap": 0.0}
            batches = [pairs[i : i + bs] for i in range(0, len(pairs), bs)]
            pbar = tqdm(batches, desc=f"Epoch {epoch+1}/{self.cfg.num_epochs}")
            for batch in pbar:
                metrics = self.train_step(batch)
                for k, v in metrics.items():
                    running[k] += v
                pbar.set_postfix({k: f"{v:.3f}" for k, v in metrics.items()})
            n = max(len(batches), 1)
            print(f"Epoch {epoch+1}: " + ", ".join(f"{k}={v/n:.3f}" for k, v in running.items()))

    # ---- inference ---------------------------------------------------------

    @torch.no_grad()
    def score(self, prompts: list[str], responses: list[str]) -> torch.Tensor:
        self.model.eval()
        rewards = []
        for prompt, response in zip(prompts, responses):
            text = f"User: {prompt}\n\nAssistant: {response}"
            enc = self.tokenizer(
                text, truncation=True, max_length=self.cfg.max_seq_length, return_tensors="pt"
            ).to(self.device)
            rewards.append(self.model(enc["input_ids"], enc["attention_mask"]).item())
        return torch.tensor(rewards)

    # ---- persistence -------------------------------------------------------

    def save(self, path: str | None = None) -> str:
        path = path or self.cfg.output_dir
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "reward_model.pt"))
        self.tokenizer.save_pretrained(path)
        print(f"Saved reward model to {path}")
        return path

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(path, "reward_model.pt")))
        print(f"Loaded reward model from {path}")
