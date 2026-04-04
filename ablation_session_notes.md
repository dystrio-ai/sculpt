# Ablation Study Session Notes — March 6, 2026

## Model: meta-llama/Llama-3.2-3B-Instruct

### Baseline
- ppl_w103 = 18.07
- downstream accuracy = 0.5862 (MMLU=0.52, HS=0.66, ARC=0.43)

### Perplexity Results (ppl_ratio — lower is better, 1.0 = baseline)

| keep_frac | structural | magnitude | sensitivity (BUGGED) | random |
|-----------|-----------|-----------|---------------------|--------|
| 0.90 | **0.985** | 1.021 | 140.3 ❌ | — |
| 0.85 | **1.134** | 1.146 | 178.6 ❌ | — |
| 0.80 | 1.275 | **1.245** | 151.1 ❌ | — |
| 0.75 | **1.492** | 1.523 | 191.5 ❌ | — |

### Key Finding
Magnitude and structural are nearly identical on perplexity. Magnitude slightly beats structural at kf=0.80.

### Sensitivity Bug (FOUND & FIXED LOCALLY, NOT YET ON H100)
`select_blocks_sensitivity()` in `selectors/baselines.py` divided block count by `feature_multiplier=3`, but `block_sensitivity` is already at block granularity (unlike the covariance matrix `D`). This made it think there were 21 blocks instead of 64, effectively pruning to ~30% instead of the target keep_frac. All sensitivity results are invalid.

**Fix:** `n_blocks = block_sensitivity.shape[0]` instead of `n_feat // F`.
- Fixed locally in repo, tests pass (424/424).
- NOT yet patched on H100 installed package.

### Other Bugs Fixed During Session
1. `adapter` not passed through `_select_stage_size` in `policy.py` → NameError
2. `compress_layer()` on dense adapters didn't accept `**kwargs` → crashed on `coupling_matrix` arg
3. `ablation_study.sh` had `set -euo pipefail` → script died on sculpt failures (changed to `set -uo pipefail`)
4. Skip check used `metrics.json` but sculpt produces `run_metadata.json`

### What Needs to Happen Next Session on H100

```bash
# 1. Patch the sensitivity bug on installed package
python3 -c "
p = '/home/shadeform/.local/lib/python3.10/site-packages/dystrio_sculpt/selectors/baselines.py'
t = open(p).read()
t = t.replace(
    '''    n_feat = block_sensitivity.shape[0]
    F = feature_multiplier
    n_blocks = n_feat // F
    if n_blocks == 0:
        n_blocks = max(1, n_feat)
        F = 1
    keep_n = max(1, int(math.ceil(keep_frac * n_blocks)))''',
    '''    n_blocks = block_sensitivity.shape[0]
    keep_n = max(1, int(math.ceil(keep_frac * n_blocks)))''')
open(p,'w').write(t)
print('patched')
"

# 2. Delete invalid sensitivity results
rm -rf ~/BumbleB/ablation_results/Llama-3.2-3B-Instruct_sensitivity_kf*

# 3. Relaunch (skips structural + magnitude, runs sensitivity + random)
nohup bash -c 'export PATH="$HOME/.local/bin:$PATH"; MODEL=meta-llama/Llama-3.2-3B-Instruct SKIP_LMEVAL=1 bash ~/BumbleB/scripts/ablation_study.sh' > ~/BumbleB/ablation.log 2>&1 & echo "PID: $!"
```

### Open Questions
1. Will fixed sensitivity close the gap with structural? (That's the real ablation.)
2. Will downstream tasks (lm-eval) show more separation than perplexity?
3. Is the repair/distillation step equalizing all selectors?
4. Does Physarum's advantage appear more on larger models?

---

## Head-to-Head Benchmark vs Published Methods

### Script: `scripts/head_to_head_bench.sh`

### Model: meta-llama/Llama-2-7b-hf

### Published Targets (DDP ICML'26, Table 2)

| Method | Venue | 20% Wiki2 | 20% Acc | 50% Wiki2 | 50% Acc |
|--------|-------|----------|---------|----------|---------|
| Dense | — | 12.18 | 66.63% | 12.18 | 66.63% |
| LoRAPrune | 2024 | 16.80* | 60.05%* | 30.12* | 49.71%* |
| LoRAP | 2024 | 14.67 | 61.20% | 26.26 | 52.31% |
| SlimLLM | ICML'25 | 15.28 | 61.70% | 27.29 | 52.02% |
| **DDP (SOTA)** | **ICML'26** | **14.39** | **64.82%** | **26.34** | **56.70%** |

*LoRAPrune numbers are from LLaMA-7B (not LLaMA-2-7B)

### Eval Setup (matching DDP paper exactly)
- WikiText-2 perplexity
- 9 zero-shot tasks via lm-eval: ARC-e, ARC-c, OBQA, WinoGrande, PIQA, HellaSwag, MathQA, RTE, BoolQ
- No sample limit (full eval)

### Run Configurations

Sculpt prunes MLP only (not attention). To compare fairly:

| Label | MLP keep_frac | Rationale |
|-------|--------------|-----------|
| matched_20pct | 0.69 | MLP reduction matching 20% total params |
| matched_50pct | 0.22 | MLP reduction matching 50% total params |
| direct_20pct | 0.80 | Direct 20% MLP pruning (less total compression than DDP) |
| direct_50pct | 0.50 | Direct 50% MLP pruning |

LLaMA-2-7B param breakdown: Embed 3.9%, Attention 31.9%, MLP 64.3%

### H100 Launch Commands

```bash
# 1. Patch sensitivity bug (if not yet done)
python3 -c "
p = '/home/shadeform/.local/lib/python3.10/site-packages/dystrio_sculpt/selectors/baselines.py'
t = open(p).read()
t = t.replace(
    '''    n_feat = block_sensitivity.shape[0]
    F = feature_multiplier
    n_blocks = n_feat // F
    if n_blocks == 0:
        n_blocks = max(1, n_feat)
        F = 1
    keep_n = max(1, int(math.ceil(keep_frac * n_blocks)))''',
    '''    n_blocks = block_sensitivity.shape[0]
    keep_n = max(1, int(math.ceil(keep_frac * n_blocks)))''')
open(p,'w').write(t)
print('patched')
"

# 2. Pull latest code (has h2h script)
cd ~/BumbleB && git pull origin main

# 3. Reinstall
pip install --user --force-reinstall .

# 4. Launch head-to-head benchmark
nohup bash -c 'export PATH="$HOME/.local/bin:$PATH"; bash ~/BumbleB/scripts/head_to_head_bench.sh' > ~/BumbleB/h2h.log 2>&1 & echo "PID: $!"

# 5. Monitor
tail -30 ~/BumbleB/h2h.log
grep -aE '\[matched|direct' ~/BumbleB/h2h.log
```

### What We're Proving
If Sculpt at matched_20pct (kf=0.69) achieves Wiki2 ppl_ratio ≤ 1.20 and Mean Acc ≥ 64%,
we match or beat DDP SOTA — with a faster one-shot method vs their gradient-based mask optimization.

### Local Repo State
- Branch: `experimental/moe-expert-prune`
- All fixes committed and pushed to `clusteroptimizerengine/sculpt` (private)
- Remote H100 repo at `~/BumbleB` is behind by the sensitivity bugfix + h2h script
