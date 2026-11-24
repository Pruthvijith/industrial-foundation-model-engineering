# Industrial language model post-training

Can a small open-weight model learn to choose the right engineering unit and abstain on an incomplete lab record? This experiment adapts SmolLM2-135M using 460,800 trainable LoRA parameters. The candidate fails the release criteria.

Independent, AI-assisted portfolio research milestone · v0.1.0 · Python 3.12 · CPU reference environment.

## Measured outcome

Base 8/48, SFT 13/48, DPO 14/48, constrained GRPO 11/48. The separate reward head ranked the oracle answer above its distractor on 23/48 cases. The 72,004-parameter student matched its SFT teacher on 87.5% of cases and answered 11/48 correctly. A second supervised and DPO run reproduced both adapter SHA-256 digests. See reports/run.json, repeated_run.json, alignment.json and trl_single.json.

## Architecture and data flow

A seeded generator creates equipment groups, shuffles answer positions and switches wording between training and test. A frozen causal transformer plus trainable query/value adapters produces next-token scores. SFT uses cross-entropy over the full vocabulary. DPO uses the chosen-minus-rejected logit margin relative to the frozen SFT margin. For single-token completions the softmax normaliser cancels exactly. Each stage writes independent predictions and a safetensors adapter.

## Dataset and model usage contract

96 synthetic instruction and preference records across 24 training groups. The holdout contains 48 records across 12 new identifiers. Position is shuffled. Every group sees the same four task families. Effective linguistic diversity is much smaller than the case count.

The supported purpose is offline research and human-reviewed engineering decision support. Aviation design approval, operational aircraft decisions and regulated certification are outside the validated scope.

## Reproduce

Run from this checkout on Linux x86-64. The reference CPU was an AMD EPYC 9V74. The portfolio index includes a downloader for the pinned models and UCI input.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
python -m pip install -c constraints-cpu.txt -r requirements-dev.txt
python -m posttrain.experiment --out work/run
python -m posttrain.alignment work/run/sft work/alignment
python -m posttrain.train_trl --out work/trl --steps 3
```

## Testing and deployment

```bash
python -m pytest --cov=posttrain --cov-report=term-missing
python -m ruff check .
python -m mypy posttrain
```

The default CPU run uses three PyTorch threads. Set LOCAL_MODEL to a verified SmolLM2 snapshot to work offline. For a new GPU host, install its compatible CUDA PyTorch build and bitsandbytes, then use --config configs/qwen.json or configs/mistral.json with posttrain.experiment. Their model configuration and four answer tokens were checked at pinned revisions. Actual CUDA allocations, quantised kernels and convergence remain open. For full-precision-base distributed LoRA use torchrun -m posttrain.train_trl with --profile and --strategy fsdp or zero3. That runner uses its own BF16 settings and does not enable QLoRA. Native CPU DDP was attempted and Gloo interface setup was denied. No multi-GPU result is reported.

## Operations and next validation

If a run exhausts memory, lower batch size and start in a fresh output directory. The supervised runner refuses to overwrite a prior run. TRL persists checkpoints. This milestone does not provide mid-step resume for the custom loop. Before running a scaling experiment, check per-device activation memory with the real sequence length and monitor the full loss trace.

See [decisions and interview guide](docs/review.md) and the measured files in [reports](reports/). Hosted CI and production qualification require their own evidence. For dependencies and known advisories see the portfolio index security report.

## Provenance and licensing

Original code and original fixtures use Apache-2.0. Upstream models, data and third-party executables retain their own licences. External weights and source archives are fetched separately. References follow.

- https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- https://huggingface.co/docs/trl/v0.22.2/en/sft_trainer
- https://huggingface.co/docs/peft/developer_guides/quantization
- https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
- https://huggingface.co/docs/accelerate/usage_guides/fsdp
- https://huggingface.co/docs/accelerate/usage_guides/deepspeed

## Local verification

8 tests passed. Statement coverage of the posttrain package was 93.46%. Ruff and mypy passed. See [verification record](reports/verification.json).
