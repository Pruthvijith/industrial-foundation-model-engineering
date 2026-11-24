# Design decision and review guide

## Decision record 001

Version one starts with a small open checkpoint to make real training and an independent rerun affordable. The one-token interface permits an exact preference margin and deterministic reward, but does not measure free-text reasoning. The TRL completion-only trainer has its own checkpoint smoke test. Four-action GRPO uses eight sampled actions per prompt, a clipped ratio and exact categorical KL. Only 34 of 40 groups contained reward variation.

## Operational response

If a run exhausts memory, lower batch size and start in a fresh output directory. The supervised runner refuses to overwrite a prior run. TRL persists checkpoints. This milestone does not provide mid-step resume for the custom loop. Before running a scaling experiment, check per-device activation memory with the real sequence length and monitor the full loss trace.

## Interview practice

Use these answers after reproducing the experiment. Describe the work as an AI-assisted independent portfolio project.

### Why compare DPO against the SFT policy?

The preference objective rewards changes in relative preference. I cache the chosen-minus-rejected margin from the frozen SFT policy for each training pair. Otherwise the loss can reward a preference the reference already expressed.

### Why did distillation agreement exceed task accuracy?

The target was the teacher distribution over four labels. A student can faithfully reproduce a weak teacher. I would report both measures and work on the teacher before compressing it.

### What would test generalisation?

Reserve independent formula families, terminology and externally authored records. Generate labels independently of the prompt generator. Grouped identifiers alone cannot isolate template leakage.


## Distributed profiles and capacity planning

For a cluster smoke test, stage the same base revision and training code on each node. Confirm CUDA, NCCL, driver compatibility and BF16 support. Accelerate/Trainer launches through torchrun. Set the same rendezvous address and port on every node and the appropriate node rank. The FSDP path wraps decoder-layer classes with full parameter sharding. The ZeRO path delegates parameter, gradient and optimizer sharding to DeepSpeed. Begin with a short checkpoint/reload drill before committing compute to a long job. No distributed scaling number was measured.

The compact CPU experiment uses 460,800 trainable adapter parameters. FP32 adapter storage alone is about 1.76 MiB. For a nominal 7B base, BF16 weights alone are approximately 14 GB in decimal units, before gradients, optimizer state, activations, allocator reserve and any KV cache. Four-bit raw weights nominally require about 3.5 GB before scales and unquantised components. These are planning estimates, not measured VRAM. Validate throughput and peak allocation under the intended sequence length, batch size and optimizer. A correct CPU forward pass cannot validate a CUDA kernel or a sharded checkpoint.

The TRL path enables gradient checkpointing and BF16 under CUDA. The current distributed trainer loads an FP32 base before Trainer initialization, which can require substantial host RAM. The configs are starting points for a hardware rehearsal. Use a reviewed sharded-loading path for a model that will not fit during startup.
