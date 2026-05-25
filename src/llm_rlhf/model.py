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

    def save_checkpoint(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Saved model + tokenizer to {path}")

    def load_adapter(self, adapter_path: str) -> None:
        """Attach a LoRA adapter trained by SFT/DPO/etc."""
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        print(f"Loaded adapter from {adapter_path}")
