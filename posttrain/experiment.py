"""Fit LoRA to a pinned causal LM, then optimise against a frozen SFT policy."""

import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .data import LETTERS, cases, digest


def choice_tokens(tokenizer):
    tokens = [tokenizer.encode(" " + c, add_special_tokens=False) for c in LETTERS]
    if any(len(t) != 1 for t in tokens) or len({t[0] for t in tokens}) != 4:
        raise ValueError("Profile requires four distinct single-token completions")
    return [t[0] for t in tokens]


def logits_for(model, tokenizer, prompts):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    logits = model(**inputs, use_cache=False).logits
    last = inputs.attention_mask.sum(1) - 1
    return logits[torch.arange(len(prompts), device=model.device), last].float()


@torch.inference_mode()
def predictions(model, tokenizer, rows, ids, size):
    model.eval()
    output = []
    for start in range(0, len(rows), size):
        batch = rows[start : start + size]
        logits = logits_for(model, tokenizer, [r["prompt"] for r in batch])
        selected = logits[:, ids].argmax(-1).tolist()
        free = logits.argmax(-1).tolist()
        for row, idx, tok in zip(batch, selected, free):
            output.append({**row, "prediction": LETTERS[idx], "unconstrained_token": tokenizer.decode([tok])})
    return output


def pair_margin(scores, batch, ids):
    ar = torch.arange(len(batch), device=scores.device)
    pos = [ids[LETTERS.index(r["target"])] for r in batch]
    neg = [ids[LETTERS.index(r["rejected"])] for r in batch]
    return scores[ar, pos] - scores[ar, neg]


def run(config, out: Path, local_model=None):
    start = time.monotonic()
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["threads"])
    rng = random.Random(config["seed"])
    device = config["device"]
    src = local_model or config["model"]
    opts = dict(revision=config["revision"], trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(src, **opts)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    load_opts: dict[str, Any] = {}
    if config["qlora"]:
        if device != "cuda" or not torch.cuda.is_available():
            raise ValueError("4-bit profile requires CUDA")
        load_opts.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
            device_map={"": 0},
        )
    model = AutoModelForCausalLM.from_pretrained(
        src, use_safetensors=True, torch_dtype=getattr(torch, config["precision"]), **opts, **load_opts
    )
    if config["qlora"]:
        model = prepare_model_for_kbit_training(model)
    else:
        model.to(device)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=config["rank"],
            lora_alpha=2 * config["rank"],
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
        ),
    )
    ids = choice_tokens(tokenizer)
    train = cases("train", config["train_groups"], config["seed"])
    test = cases("test", config["test_groups"], config["seed"])
    out.mkdir(parents=True, exist_ok=False)
    stages = {}
    size = config["batch_size"]

    def record(stage):
        preds = predictions(model, tokenizer, test, ids, size)
        stages[stage] = sum(r["target"] == r["prediction"] for r in preds) / len(preds)
        (out / f"{stage}.json").write_text(json.dumps(preds, indent=2) + "\n")

    record("base")
    losses: dict[str, list[float]] = {}
    for stage in ("sft", "dpo"):
        opt = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=config["learning_rate"] * (1 if stage == "sft" else 0.25),
        )
        steps = config[stage + "_steps"]
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        reference: dict[str, float] = {}
        if stage == "dpo":
            model.eval()
            with torch.no_grad():
                for i in range(0, len(train), size):
                    batch = train[i : i + size]
                    margins = pair_margin(
                        logits_for(model, tokenizer, [r["prompt"] for r in batch]), batch, ids
                    )
                    reference.update(zip([r["id"] for r in batch], margins.tolist()))
        model.train()
        losses[stage] = []
        for step in range(steps):
            batch = rng.sample(train, size)
            scores = logits_for(model, tokenizer, [r["prompt"] for r in batch])
            if stage == "sft":
                targets = torch.tensor([ids[LETTERS.index(r["target"])] for r in batch], device=model.device)
                loss = torch.nn.functional.cross_entropy(scores, targets)
            else:
                ref = scores.new_tensor([reference[r["id"]] for r in batch])
                loss = -torch.nn.functional.logsigmoid(0.1 * (pair_margin(scores, batch, ids) - ref)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses[stage].append(float(loss.detach()))
        record(stage)
        model.save_pretrained(out / stage, safe_serialization=True)
    report = dict(
        config=config,
        train_sha256=digest(train),
        test_sha256=digest(test),
        test_n=len(test),
        choices_accuracy=stages,
        losses=losses,
        wall_seconds=time.monotonic() - start,
        torch=torch.__version__,
        python=platform.python_version(),
        machine=platform.machine(),
        cpu_threads=torch.get_num_threads(),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        prediction_sha256={s: hashlib.sha256((out / f"{s}.json").read_bytes()).hexdigest() for s in stages},
        adapters={
            s: hashlib.sha256((out / s / "adapter_model.safetensors").read_bytes()).hexdigest()
            for s in losses
        },
    )
    (out / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/cpu.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(json.loads(args.config.read_text()), args.out, os.getenv("LOCAL_MODEL"))["choices_accuracy"])
