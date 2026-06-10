"""QLoRA fine-tuning of Qwen3-4B on datasheet Q/A.

Plain transformers + peft + TRL + bitsandbytes (not Unsloth, whose version
pins conflict with this machine's torch 2.12 / CUDA 13 stack; the standard
stack is more portable and equally capable here). 4-bit NF4 base with a LoRA
adapter on the attention/MLP projections.

Runs in the isolated .venv-train. Outputs the adapter to models/qlora-adapter/.

    .venv-train/bin/python scripts/train_qlora.py [--epochs 3] [--device cuda:1]
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    train_file: Path = typer.Option(Path("data/train/train.jsonl"), "--train"),
    val_file: Path = typer.Option(Path("data/train/val.jsonl"), "--val"),
    base_model: str = typer.Option("Qwen/Qwen3-4B", "--base"),
    out_dir: Path = typer.Option(Path("models/qlora-adapter"), "--out"),
    epochs: int = typer.Option(3, "--epochs"),
    lr: float = typer.Option(2e-4, "--lr"),
    batch: int = typer.Option(2, "--batch"),
    grad_accum: int = typer.Option(8, "--grad-accum"),
    max_seq: int = typer.Option(1024, "--max-seq"),
    device: str = typer.Option("cuda:1", "--device"),
    seed: int = typer.Option(0, "--seed"),
    max_steps: int = typer.Option(-1, "--max-steps", help="Cap steps for a smoke test (-1 = full)"),
) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    dev_index = int(device.split(":")[1]) if ":" in device else 0
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_cfg, device_map={"": dev_index}
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def to_text(example):
        return {"text": tok.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False)}

    ds_train = load_dataset("json", data_files=str(train_file), split="train").map(to_text)
    ds_val = load_dataset("json", data_files=str(val_file), split="train").map(to_text)

    args = SFTConfig(
        output_dir=str(out_dir / "_checkpoints"),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=True,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        seed=seed,
        report_to=[],
        dataset_text_field="text",
        max_length=max_seq,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
        processing_class=tok,
    )
    result = trainer.train()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "train_metrics.json").write_text(
        json.dumps(result.metrics, indent=2), encoding="utf-8"
    )
    typer.echo(f"adapter saved to {out_dir}; final loss {result.metrics.get('train_loss')}")


if __name__ == "__main__":
    app()
