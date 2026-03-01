# Dystrio Sculpt

Structural FFN compiler for decoder-only transformer LLMs. Sculpt removes
redundant neurons from SwiGLU feed-forward blocks using operator-fidelity
analysis, coupling-geometry diversity scoring, and staged repair fine-tuning.
The result is a smaller, faster HuggingFace-compatible model with controlled
quality loss.

## What Sculpt does

1. **Analyze** — Prescan all FFN layers on the uncompressed model to measure
   block-level operator sensitivity and inter-block coupling geometry.
2. **Select** — Rank neuron blocks using a Physarum-inspired diversity penalty
   on top of operator-fidelity scores. Blocks that are both unimportant and
   redundant with their neighbours are pruned first.
3. **Slice** — Physically remove the pruned neurons from `gate_proj`,
   `up_proj`, and `down_proj` weight matrices. The model stays a valid
   HuggingFace checkpoint with a reduced `intermediate_size`.
4. **Repair** — Fine-tune only the compressed MLP parameters with a cosine LR
   schedule, staged across layer groups to prevent catastrophic quality loss.
5. **Validate** — Reload the saved model and run a forward pass to confirm no
   NaN/Inf and correct output shapes.

Sculpt searches over compression ratios automatically. You specify how many
points on the quality-vs-speed Pareto frontier you want, and Sculpt emits that
many fully self-contained model directories.

## Enterprise hardening (V1)

Sculpt includes several production-safety features that make it robust across
model families (Qwen, Mistral, LLaMA):

- **Adaptive repair policy selection** — Before the full frontier search, Sculpt
  runs a short pilot compile on a single layer to auto-select a stable repair
  learning rate and stage size from a policy ladder. No manual tuning required.
- **Best-checkpoint restore** — During every repair loop the best metric
  checkpoint is tracked, and weights are restored to that point at the end. Late
  regression is never shipped.
- **"Never ship worse than compiled" invariant** — If repair fails to improve
  quality beyond the post-compile baseline, pre-repair weights are restored
  automatically. A failed candidate is labelled clearly and excluded from the
  frontier.
- **Staged rollback with policy downshift** — If a stage repair violates the
  regression limit or produces NaN/Inf, the stage is rolled back and retried
  with a more conservative policy (lower LR, smaller stage). If retry also
  fails, the candidate is marked as failed.
- **Deterministic two-tier evaluation** — Cheap eval (small fixed subset, small
  token budget) is used during repair curve checkpoints and early stopping.
  Full-budget final eval runs only on selected Pareto points, saving hours on
  large model searches.

## Quick start

```bash
pip install -e .

dystrio sculpt --model-id Qwen/Qwen2-0.5B --outdir sculpt_out --frontier 4
```

This will:
- Auto-select a stable repair policy via pilot compile.
- Compute a baseline (no compression).
- Search over `keep_frac` in [0.4, 1.0] with adaptive binary refinement.
- Emit 4 evenly-spaced Pareto-optimal models under `sculpt_out/`.

Each frontier directory contains:

```
sculpt_out/
  frontier_0_conservative/
    model/          # save_pretrained output (config.json, safetensors, tokenizer)
    metrics.json    # PPL, throughput, speedup vs baseline
    compile_report.json
    manifest.json   # full reproducibility record (incl. repair policy, pilot report)
  frontier_1_balanced/
    ...
  frontier_2_aggressive/
    ...
  frontier_3_extreme/
    ...
  summary.csv       # incremental summary appended after each artifact
```

## Frontier example

Search for the fastest model that stays within 1.5x baseline perplexity:

```bash
dystrio sculpt \
  --model-id Qwen/Qwen2-0.5B \
  --outdir sculpt_constrained \
  --frontier 3 \
  --max-ppl-multiplier 1.5
```

Target a specific prefill speedup:

```bash
dystrio sculpt \
  --model-id Qwen/Qwen2-0.5B \
  --outdir sculpt_fast \
  --frontier 2 \
  --target-prefill-speedup 1.3
```

Time-bounded search (stop after 2 hours):

```bash
dystrio sculpt \
  --model-id Qwen/Qwen2-0.5B \
  --outdir sculpt_timed \
  --frontier 4 \
  --max-compile-hours 2.0
```

## Deterministic builds

For bitwise-reproducible compilation:

```bash
dystrio sculpt \
  --model-id Qwen/Qwen2-0.5B \
  --outdir sculpt_deterministic \
  --frontier 4 \
  --deterministic
```

Deterministic mode:
- Seeds `random`, `numpy`, `torch`, and CUDA RNGs.
- Disables TF32 matmul and cuDNN non-deterministic algorithms.
- Uses an isolated `np.random.RandomState` inside the structural selector's
  Physarum conductance solver.
- Selects fixed eval subsets via seeded shuffle for stable early-stopping
  and repair curve checkpoints.
- Records all determinism settings in `manifest.json`.

## Supported architectures

Sculpt targets **decoder-only transformers with SwiGLU FFN blocks**:
- Qwen2 / Qwen2.5
- Llama 2 / Llama 3
- Mistral / Mixtral (dense MLP layers)
- Any HuggingFace model with `gate_proj` / `up_proj` / `down_proj` structure

Attention layers, embeddings, layer norms, and residual connections are not
modified. Only the MLP projections are physically sliced.

## CLI reference

```
dystrio sculpt [OPTIONS]

Options:
  --model-id TEXT               HuggingFace model ID (required)
  --outdir TEXT                 Output directory [default: sculpt_out]
  --frontier INTEGER            Frontier points to emit [default: 4]
  --max-ppl-multiplier FLOAT    Max PPL as multiple of baseline
  --target-prefill-speedup FLOAT  Min prefill speedup to keep
  --max-compile-hours FLOAT     Time budget (hours)
  --deterministic               Enable deterministic mode
  --policy TEXT                 Override auto-selected repair policy (advanced)
  --help                        Show this message and exit
```

## License

Apache 2.0
