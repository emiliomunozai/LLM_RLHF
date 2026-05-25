"""PretrainedLLM — a thin wrapper around a Hugging Face causal LM.

This is the *starting point* of the whole pipeline: an unaligned model that
predicts text well but does not know how to follow instructions. Everything
downstream (SFT, reward modeling, PPO/DPO/GRPO) takes a `PretrainedLLM` as
its base.

We keep the wrapper deliberately tiny — just `generate`, `save_checkpoint`
and `load_adapter` — because the educational point is that an LLM is just a
function `prompt → continuation`. The interesting work happens in the
training-loop modules.

Default model is `facebook/opt-350m` because it is small enough to fit on
laptop GPUs and large enough to show qualitative alignment differences.
"""
from dataclasses import dataclass

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationConfig:
    max_new_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


class PretrainedLLM:
    def __init__(self, model_name: str = "facebook/opt-350m", device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"Loaded {n_params:.1f}M parameters.")

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        cfg = config or GenerationConfig()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=cfg.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_len = inputs.input_ids.shape[1]
        return self.tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True)

    def next_token_distribution(
        self,
        prompt: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 10,
    ) -> tuple[list[str], "torch.Tensor"]:
        """Return the top-`top_k` candidate next tokens and their probabilities
        after temperature scaling and (optional) nucleus (top-p) truncation.

        This is the *same* math `generate()` uses internally to sample, exposed
        so notebooks can visualise how `temperature` and `top_p` reshape the
        distribution:

        - `temperature` divides the logits before softmax. `T<1` sharpens
          (the model becomes more confident in the top token); `T>1` flattens
          (more diversity). `T=1` leaves the raw distribution untouched.
        - `top_p` keeps the smallest set of tokens whose cumulative probability
          is at least `top_p`, then renormalises. The remaining mass is set to
          zero — those tokens can never be sampled. `top_p=1.0` is a no-op.

        Returns `(tokens, probs)` where `probs` is on CPU as a 1-D tensor of
        length `top_k`, aligned with `tokens` (most-probable first).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1].float()

        logits = logits / max(temperature, 1e-8)
        probs = torch.softmax(logits, dim=-1)

        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            # Mask tokens *past* the nucleus; shift right so the token that
            # crosses the threshold is itself kept (otherwise top_p≈0 returns
            # an empty set).
            cutoff = cumulative > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum()
            probs = torch.zeros_like(probs).scatter(0, sorted_idx, sorted_probs)

        top_probs, top_ids = probs.topk(top_k)
        tokens = [self.tokenizer.decode([i]) for i in top_ids.tolist()]
        return tokens, top_probs.cpu()

    def save_checkpoint(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Saved model + tokenizer to {path}")

    def load_adapter(self, adapter_path: str, trainable: bool = False) -> None:
        """Attach a LoRA adapter trained by SFT/DPO/etc.

        `trainable=False` (default) loads the adapter in inference mode —
        every parameter, including the LoRA weights, has `requires_grad=False`.
        This is what you want when *using* the model.

        `trainable=True` keeps the LoRA weights trainable, which is what
        downstream stages (DPO/PPO/GRPO) need when they want to *continue*
        training from the SFT checkpoint.
        """
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            is_trainable=trainable,
        )
        print(f"Loaded adapter from {adapter_path} (trainable={trainable})")
