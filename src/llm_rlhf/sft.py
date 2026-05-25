"""Supervised Fine-Tuning (SFT).

SFT is the first alignment step: we teach the pretrained LM to *follow the
instruction-response format* by showing it thousands of high-quality
demonstrations. After SFT the model still doesn't know what "good" answers
look like — only what the *shape* of an instruction-following exchange looks
like. RLHF (next steps) fixes the "good vs bad" part.

The training data is `databricks/databricks-dolly-15k`: 15k instruction /
response pairs hand-written by Databricks employees, covering open QA,
classification, brainstorming and summarization.

LoRA (Low-Rank Adaptation) is used so we only train ~0.5% of the parameters
— two low-rank matrices inserted into the attention projections. That keeps
fine-tuning fast and the saved adapter tiny, which is exactly the property
that makes it practical to share fine-tunes.
"""
import os
from dataclasses import dataclass

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from llm_rlhf.model import PretrainedLLM


@dataclass
class SFTConfig:
    dataset_name: str = "databricks/databricks-dolly-15k"
    max_seq_length: int = 512
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    # Optimization
    output_dir: str = "checkpoints/sft"
    num_epochs: int = 1
    batch_size: int = 8
    grad_accum_steps: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 200


def format_instruction(example: dict) -> dict:
    """Format a row from Dolly (or any instruction dataset) into a single string.

    We collapse different column names ('instruction'/'response',
    'prompt'/'completion', 'input'/'output') into one canonical template so the
    rest of the pipeline (reward model, PPO, DPO) can rely on the format.
    """
    if "instruction" in example and "response" in example:
        prompt, response = example["instruction"], example["response"]
    elif "prompt" in example and "completion" in example:
        prompt, response = example["prompt"], example["completion"]
    else:
        prompt = str(example.get("input", ""))
        response = str(example.get("output", ""))
    return {"formatted_text": f"User: {prompt.strip()}\n\nAssistant: {response.strip()}"}


class SupervisedFineTuner:
    def __init__(self, base: PretrainedLLM, config: SFTConfig | None = None):
        self.base = base
        self.cfg = config or SFTConfig()
        self.tokenizer = base.tokenizer
        self.model = base.model
        self.tokenized_dataset = None

    def prepare_data(self):
        print(f"Loading {self.cfg.dataset_name}...")
        raw = load_dataset(self.cfg.dataset_name)
        processed = raw.map(format_instruction)

        def tokenize(batch):
            enc = self.tokenizer(
                batch["formatted_text"],
                padding="max_length",
                truncation=True,
                max_length=self.cfg.max_seq_length,
            )
            enc["labels"] = [
                [(-100 if t == self.tokenizer.pad_token_id else t) for t in ids]
                for ids in enc["input_ids"]
            ]
            return enc

        self.tokenized_dataset = processed.map(
            tokenize, batched=True, remove_columns=processed["train"].column_names
        )
        return self.tokenized_dataset

    def setup_peft(self):
        """Insert LoRA adapters into the attention projections."""
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=list(self.cfg.lora_target_modules),
            bias="none",
            inference_mode=False,
        )
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, peft_config)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable LoRA params: {trainable:,} ({trainable/total:.2%} of {total:,})")
        # Keep base.model in sync so downstream stages see the adapted model
        self.base.model = self.model
        return self.model

    def train(self) -> str:
        if self.tokenized_dataset is None:
            raise RuntimeError("Call prepare_data() before train().")

        args = TrainingArguments(
            output_dir=self.cfg.output_dir,
            num_train_epochs=self.cfg.num_epochs,
            per_device_train_batch_size=self.cfg.batch_size,
            gradient_accumulation_steps=self.cfg.grad_accum_steps,
            warmup_ratio=self.cfg.warmup_ratio,
            weight_decay=self.cfg.weight_decay,
            learning_rate=self.cfg.learning_rate,
            logging_steps=self.cfg.logging_steps,
            save_steps=self.cfg.save_steps,
            save_total_limit=3,
            fp16=(self.base.device == "cuda"),
            report_to="none",
        )
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.tokenized_dataset["train"],
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        trainer.train()

        adapter_path = os.path.join(self.cfg.output_dir, "adapter")
        self.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)
        print(f"Saved LoRA adapter to {adapter_path}")
        return adapter_path

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate with the SFT format, so the model sees the same template as in training."""
        formatted = f"User: {prompt}\n\nAssistant: "
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.base.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
