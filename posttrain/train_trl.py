"""TRL completion-only SFT. Supports torchrun and optional Trainer scaling configs."""

import argparse
import json
import os
from pathlib import Path
from typing import Any
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from .data import cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", type=Path, default=Path("configs/cpu.json"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--strategy", choices=["ddp", "fsdp", "zero3"], default="ddp")
    args = p.parse_args()
    if args.steps < 1:
        p.error("steps must be positive")
    cfg = json.loads(args.profile.read_text())
    source = os.getenv("LOCAL_MODEL", cfg["model"])
    cuda = cfg["device"] == "cuda"
    if cuda and not torch.cuda.is_available():
        p.error("CUDA profile needs a CUDA build and device")
    if args.strategy != "ddp" and not cuda:
        p.error("FSDP and ZeRO profiles require CUDA")
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)
    tok = AutoTokenizer.from_pretrained(source, revision=cfg["revision"])
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        source, revision=cfg["revision"], use_safetensors=True, torch_dtype=torch.float32
    )
    options: dict[str, Any] = {}
    if args.strategy == "zero3":
        options["deepspeed"] = "configs/zero3.json"
    if args.strategy == "fsdp":
        options.update(
            fsdp="full_shard auto_wrap",
            fsdp_config={
                "transformer_layer_cls_to_wrap": model.model.layers[0].__class__.__name__,
                "use_orig_params": True,
            },
        )
    training = SFTConfig(
        output_dir=str(args.out),
        max_steps=args.steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=0.0003,
        lr_scheduler_type="linear",
        seed=cfg["seed"],
        bf16=cuda,
        fp16=False,
        use_cpu=not cuda,
        gradient_checkpointing=cuda,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        report_to=[],
        save_strategy="steps",
        save_steps=args.steps,
        save_total_limit=2,
        max_length=128,
        ddp_find_unused_parameters=False,
        completion_only_loss=True,
        dataloader_pin_memory=cuda,
        **options,
    )
    rows = [
        dict(prompt=r["prompt"], completion=" " + r["target"]) for r in cases("train", cfg["train_groups"])
    ]
    trainer = SFTTrainer(
        model=model,
        args=training,
        processing_class=tok,
        train_dataset=Dataset.from_list(rows),
        peft_config=LoraConfig(
            task_type="CAUSAL_LM",
            r=cfg["rank"],
            lora_alpha=2 * cfg["rank"],
            target_modules=["q_proj", "v_proj"],
        ),
    )
    result = trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        (args.out / "metrics.json").write_text(
            json.dumps(
                {
                    **result.metrics,
                    "world_size": training.world_size,
                    "strategy": args.strategy,
                    "optimizer_steps": trainer.state.global_step,
                    "device": str(training.device),
                    "base_revision": cfg["revision"],
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
