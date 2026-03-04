# Dystrio Sculpt

Structural FFN compiler for decoder-only transformer LLMs. Sculpt removes
redundant neurons from SwiGLU feed-forward blocks using operator-fidelity
analysis, coupling-geometry diversity scoring, and staged repair fine-tuning.
The result is a smaller, faster HuggingFace-compatible model with controlled
quality loss.

## What Sculpt does

1. **Analyze** — Prescan all FFN layers on the uncompressed model to measure
   block-level operator sensitivity and inter-block coupling geometry.
2. **Score risk** — Compute a structural risk score per layer (and aggregate)
   from sensitivity, coupling concentration, and spectrum rank. This steers
   the search bracket, repair policy, and stage ordering automatically.
3. **Select** — Rank neuron blocks using a Physarum-inspired diversity penalty
   on top of operator-fidelity scores. Blocks that are both unimportant and
   redundant with their neighbours are pruned first.
4. **Slice** — Physically remove the pruned neurons from `gate_proj`,
   `up_proj`, and `down_proj` weight matrices. The model stays a valid
   HuggingFace checkpoint with a uniform reduced `intermediate_size`.
5. **Repair** — Fine-tune only the compressed MLP parameters with a cosine LR
   schedule, staged across layer groups ordered by compressibility (safest
   first) to prevent catastrophic quality loss.
6. **Validate** — Reload the saved model and run a forward pass to confirm no
   NaN/Inf and correct output shapes.

## Default behaviour: fastest safe model

By default Sculpt searches for the **fastest model under a quality ceiling**
(`--max-ppl-multiplier`, default 2.0x baseline PPL). The search algorithm
(Safe Bracket Search) uses structural priors to choose an initial bracket of
`keep_frac` candidates adapted to the model's risk profile:

| Risk level | Starting bracket | Description |
|---|---|---|
| Low (≤ 0.35) | 0.85 → 0.55 | Aggressively explore — model tolerates pruning |
| Medium | 0.88 → 0.62 | Standard sweep |
| High (≥ 0.65) | 0.92 → 0.72 | Conservative — tight coupling or high sensitivity |

Points that exceed the quality ceiling are rejected immediately. The search
then bisects the safe/unsafe boundary to find the fastest keep_frac that stays
within budget.

## Uniform-width artifact guarantee

All emitted models use a **single scalar `intermediate_size`** across all
layers. This means every artifact is a drop-in HuggingFace checkpoint — no
custom model classes, no per-layer width vectors, no special runtime code.
Just `AutoModelForCausalLM.from_pretrained(path)`.

## Enterprise hardening (V1)

- **Structural risk scoring** — Prescan-derived risk score (sensitivity +
  coupling concentration + spectrum rank) drives bracket selection, policy
  ladder start, and repair budget scaling.
- **Adaptive repair policy selection** — Pilot compile auto-selects a stable
  repair LR and stage size. High-risk models start lower on the policy ladder.
- **Risk-scaled repair** — High-risk models get more repair steps and larger
  cheap-eval subsets automatically.
- **Compressibility-ordered staging** — Layers are compressed safest-first
  (sorted by increasing risk) to reduce inter-stage interference.
- **Best-checkpoint restore** — During every repair loop the best metric
  checkpoint is tracked and restored at end. Late regression is never shipped.
- **"Never ship worse than compiled" invariant** — If repair fails to improve,
  pre-repair weights are restored and the candidate is excluded from the
  frontier.
- **Staged rollback with policy downshift** — If a stage repair fails, the
  stage is rolled back and retried with a more conservative policy.
- **Quality ceiling enforcement** — No point is labelled "balanced" if its
  PPL ratio exceeds the ceiling. If no safe point exists, only "conservative"
  is emitted with a clear message.
- **Deterministic two-tier evaluation** — Cheap eval for search; full eval
  only on final selected points.

## Logging and verbosity

By default, Dystrio suppresses noisy output from Hugging Face libraries,
httpx, and datasets (including harmless 404 probes for `additional_chat_templates`
etc.). Only Dystrio's own INFO-level logs are shown.

```bash
# Default: clean, product-like output
dystrio sculpt --model-id Qwen/Qwen2-0.5B

# Quiet: warnings and errors only, progress bars disabled
dystrio -q sculpt --model-id Qwen/Qwen2-0.5B

# Verbose: full debug output including HF/httpx request tracing
dystrio -v sculpt --model-id Qwen/Qwen2-0.5B
```

The `--quiet` / `-q` and `--verbose` / `-v` flags are global (apply to all
commands: `sculpt`, `bench`, `bench-report`, `bench-audit`). They are mutually
exclusive.

## Quick start

```bash
pip install -e .

dystrio sculpt --model-id Qwen/Qwen2-0.5B --outdir sculpt_out --frontier 4
```

This will:
- Compute structural risk score from prescan.
- Auto-select a risk-aware repair policy via pilot compile.
- Search for the fastest safe models under 2.0x PPL ceiling.
- Emit up to 4 named points under `sculpt_out/`.

Each frontier directory contains:

```
sculpt_out/
  frontier_0_conservative/
    model/          # save_pretrained output (config.json, safetensors, tokenizer)
    metrics.json    # PPL, throughput, speedup, risk_score
    compile_report.json
    manifest.json   # full reproducibility record
  frontier_1_balanced/
    ...
  summary.csv       # incremental summary with risk_score column
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

## Custom calibration corpus

By default Sculpt calibrates on `wikitext / wikitext-2-raw-v1 / train`.
You can point it at any Hugging Face dataset:

```bash
dystrio sculpt \
  --model-id Qwen/Qwen2-0.5B \
  --outdir sculpt_c4 \
  --calib-dataset allenai/c4 \
  --calib-config en \
  --calib-split train \
  --calib-text-field text \
  --calib-num-samples 1000 \
  --calib-seed 42
```

| Flag | Default | Description |
|------|---------|-------------|
| `--calib-dataset` | `wikitext` | HF dataset identifier |
| `--calib-config` | `wikitext-2-raw-v1` | Dataset config name |
| `--calib-split` | `train` | Dataset split |
| `--calib-text-field` | `text` | Name of the text column |
| `--calib-num-samples` | all available | Max calibration samples (deterministically sampled) |
| `--calib-seq-len` | model default | Override sequence length for calibration |
| `--calib-seed` | 0 | Seed for calibration sampling |

Eval always uses WikiText-103 validation for comparable PPL measurement
regardless of calibration corpus. The calibration dataset parameters are
recorded in `run_metadata.json` for reproducibility.

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

## Benchmarking sculpted models

After sculpting, evaluate baseline and compiled models across workloads:

```bash
dystrio bench \
  --models org/baseline-model org/sculpted-conservative org/sculpted-balanced \
  --workloads wikitext chat rag code \
  --prompts-dir prompts/ \
  --outdir bench_out \
  --dtype bf16 --device cuda --deterministic --seed 42
```

Generate report from existing results (no model loading):

```bash
dystrio bench-report --results-dir bench_out/results --outdir bench_out/report
```

Audit results for publishability:

```bash
dystrio bench-audit --bench-out bench_out
```

### Metric definitions

| Metric | Level | Description |
|--------|-------|-------------|
| **TTFT incl. prefill** (`ttft_ms`) | Request | Wall time from prompt submission to first token: prefill forward + first decode step. Computed per-prompt with CUDA sync. |
| **First decode step** (`first_decode_step_ms`) | Request | Wall time of the first decode forward call only (post-prefill). Per-prompt. |
| **Prefill wall** (`prefill_ms`) | Request | Wall time of the prefill forward pass only. Per-prompt. |
| Prefill / Decode TPS | Microbench | Throughput from batched iteration benchmarks. Used for throughput comparison, not latency claims. |
| `microbench_prefill_ms_*` | Microbench | Batched iteration latency percentiles. Internal reference — not publishable as request-level claims. |

Warmup prompts (default 5) are excluded from all published percentile metrics.

### Output layout

```
bench_out/
  run_metadata.json
  benchmarks.csv
  results/
    <sanitized_model_id>/
      wikitext/metrics.json
      chat/metrics.json + per_prompt.csv + run_metadata.json
      rag/metrics.json + per_prompt.csv + run_metadata.json
      code/metrics.json + per_prompt.csv + run_metadata.json
  report/
    frontier_rag_ttft_p95_vs_pplratio.png
    frontier_chat_decode_p95_vs_pplratio.png
    p95_latency_by_workload.png
    throughput_by_workload.png
    rag_ttft_cdf.png
    model_card_snippet.md
    audit.json
    audit.txt
```

### Generating prompt packs

```bash
python scripts/make_prompt_packs.py --outdir prompts/
```

## CLI reference

```
Global Options (apply to all commands):
  --quiet / -q                  Minimal output; suppress most logs + progress bars
  --verbose / -v                Debug output; show external library logs + request tracing

dystrio sculpt [OPTIONS]

Options:
  --model-id TEXT               HuggingFace model ID (required)
  --outdir TEXT                 Output directory [default: sculpt_out]
  --frontier INTEGER            Frontier points to emit [default: 4]
  --max-ppl-multiplier FLOAT    Quality ceiling (PPL/baseline) [default: 2.0]
  --target-prefill-speedup FLOAT  Min prefill speedup to keep
  --max-compile-hours FLOAT     Time budget (hours)
  --deterministic               Enable deterministic mode
  --policy TEXT                 Override auto-selected repair policy (advanced)
  --calib-dataset TEXT          HF dataset for calibration [default: wikitext]
  --calib-config TEXT           HF dataset config [default: wikitext-2-raw-v1]
  --calib-split TEXT            HF dataset split [default: train]
  --calib-text-field TEXT       Text column name [default: text]
  --calib-num-samples INTEGER   Max calibration samples
  --calib-seq-len INTEGER       Calibration sequence length
  --calib-seed INTEGER          Calibration sampling seed [default: 0]
  --help                        Show this message and exit

dystrio bench [OPTIONS]

Options:
  --models TEXT                 Model IDs to benchmark (required, repeatable)
  --workloads TEXT              Workloads [default: wikitext chat rag code]
  --prompts-dir TEXT            Directory with JSONL prompt packs
  --outdir TEXT                 Output directory [default: bench_out]
  --dtype TEXT                  bf16|fp16|fp32 [default: bf16]
  --device TEXT                 cuda|cpu [default: cuda]
  --seed INTEGER                Random seed [default: 0]
  --deterministic               Enable deterministic mode
  --baseline-model TEXT         Baseline model for ppl_ratio

dystrio bench-report [OPTIONS]

Options:
  --results-dir TEXT            Path to results/ directory (required)
  --outdir TEXT                 Report output [default: bench_out/report]
  --bench-out TEXT              Root bench dir (for model card env footnote)

dystrio bench-audit [OPTIONS]

Options:
  --bench-out TEXT              Root bench output dir (required)
```

## License

Apache 2.0
