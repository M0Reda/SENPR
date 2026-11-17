"""
train_lora.py -- Stage 7 (Optional): LoRA Fine-Tune
=====================================================
Fine-tunes the base LLM on the pre-made SFT JSONL files to improve
tone and domain vocabulary. Retrieval remains the source of truth.

Enabled only when config.yaml: lora.enabled: true.

Requirements (install manually with GPU):
    pip install unsloth[colab-new] trl peft bitsandbytes transformers datasets accelerate

Run:
    cd edubot
    python src/train_lora.py

The adapter is saved to ./lora_adapter/ and loaded automatically by bot.py
if that directory exists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
cfg: dict[str, Any] = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA_DIR = (ROOT / cfg["data_dir"]).resolve()

lora_cfg = cfg.get("lora", {})
ENABLED  = lora_cfg.get("enabled", False)


def main() -> None:
    if not ENABLED:
        print("LoRA fine-tuning is disabled in config.yaml (lora.enabled: false).")
        print("Set lora.enabled: true and install the GPU dependencies to enable it.")
        return

    # ── Import heavy GPU deps ─────────────────────────────────────────────────
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, DataCollatorForSeq2Seq
        from transformers import TrainingArguments
        from datasets import Dataset
    except ImportError as e:
        print(f"LoRA dependencies not installed: {e}")
        print("Run: pip install unsloth trl peft bitsandbytes transformers datasets accelerate")
        return

    base_model   = lora_cfg["base_model"]
    output_dir   = ROOT / lora_cfg["output_dir"]
    max_steps    = lora_cfg["max_steps"]
    lora_r       = lora_cfg["lora_r"]
    lora_alpha   = lora_cfg["lora_alpha"]
    lora_dropout = lora_cfg.get("lora_dropout", 0.05)
    batch_size   = lora_cfg.get("per_device_train_batch_size", 2)
    grad_accum   = lora_cfg.get("gradient_accumulation_steps", 4)

    print(f"\nLoading base model: {base_model} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=4096,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ── Load train data ───────────────────────────────────────────────────────
    train_path = DATA_DIR / "EduBot_Training" / "sft_train.jsonl"
    val_path   = DATA_DIR / "EduBot_Training" / "sft_validation.jsonl"

    def load_jsonl(path: Path) -> list[dict]:
        data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    print(f"Loading training data from {train_path} ...")
    train_data = load_jsonl(train_path)
    val_data   = load_jsonl(val_path)
    print(f"  Train: {len(train_data):,} conversations")
    print(f"  Val:   {len(val_data):,} conversations")

    # ── Format as chat template ───────────────────────────────────────────────
    def format_conversation(conv: dict) -> dict:
        msgs = conv.get("messages", conv.get("conversations", []))
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    train_dataset = Dataset.from_list([format_conversation(c) for c in train_data])
    val_dataset   = Dataset.from_list([format_conversation(c) for c in val_data])

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=4096,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            max_steps=max_steps,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_steps=100,
            output_dir=str(output_dir),
            report_to="none",
            seed=42,
        ),
    )

    print(f"\nFine-tuning for {max_steps} steps ...")
    trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nAdapter saved to {output_dir}")
    print("Set lora.enabled: true in config.yaml; bot.py will load it automatically.")


if __name__ == "__main__":
    main()
