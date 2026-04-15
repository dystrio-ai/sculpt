#!/usr/bin/env python3
"""Local CPU comparison of structural vs cohesion vs sensitivity selectors.

Generates realistic synthetic prescan data with known block coupling structure
and measures proxy quality metrics across multiple keep_fracs. Runs in seconds.

Metrics:
  - sensitivity_preserved: fraction of total block sensitivity kept (higher = better)
  - coupling_preserved: fraction of total intra-group coupling retained among
    kept blocks (higher = more functional circuits intact)
  - circuit_survival: fraction of planted "circuits" (strongly coupled block
    groups) where ALL members survive pruning (higher = better)
  - combined: geometric mean of the above three

Usage:
    python scripts/compare_selectors_local.py
    python scripts/compare_selectors_local.py --keep-fracs 0.90,0.80,0.70 --n-layers 16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dystrio_sculpt.selectors.structural import (
    select_blocks_structural,
    select_blocks_cohesion,
)
from dystrio_sculpt.selectors.baselines import select_blocks_sensitivity


def make_synthetic_layer(
    n_blocks: int = 32,
    feature_multiplier: int = 3,
    n_circuits: int = 4,
    circuit_size: int = 3,
    coupling_strength: float = 4.0,
    rng: np.random.RandomState | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[int]]]:
    """Build a synthetic layer with planted circuits (coupled block groups).

    Returns (D, block_sensitivity, block_energy, circuits) where circuits
    is a list of block-index groups that are strongly coupled.
    """
    if rng is None:
        rng = np.random.RandomState()

    F = feature_multiplier
    dim = F * n_blocks

    # Base: near-diagonal covariance with mild noise
    D_np = np.eye(dim, dtype=np.float64) * 5.0
    D_np += rng.randn(dim, dim) * 0.05
    D_np = (D_np + D_np.T) / 2

    # Plant circuits: groups of blocks with strong mutual coupling
    all_blocks = list(range(n_blocks))
    rng.shuffle(all_blocks)
    circuits: List[List[int]] = []
    for c in range(n_circuits):
        start = c * circuit_size
        if start + circuit_size > n_blocks:
            break
        group = sorted(all_blocks[start : start + circuit_size])
        circuits.append(group)
        for a in group:
            for b in group:
                if a != b:
                    for fi in range(F):
                        for fj in range(F):
                            D_np[a * F + fi, b * F + fj] += coupling_strength
                            D_np[b * F + fj, a * F + fi] += coupling_strength

    D = torch.from_numpy(D_np)

    # Sensitivity: circuit members have moderate sensitivity, some non-circuit
    # blocks have individually higher sensitivity (this is the trap — a naive
    # selector keeps the individually-important blocks and breaks circuits).
    sens = np.full(n_blocks, 0.5, dtype=np.float64)
    circuit_members = set()
    for group in circuits:
        for b in group:
            circuit_members.add(b)
            sens[b] = 0.4 + rng.rand() * 0.3  # 0.4–0.7

    for b in range(n_blocks):
        if b not in circuit_members:
            sens[b] = 0.5 + rng.rand() * 0.4  # 0.5–0.9 (individually higher)

    block_sensitivity = torch.from_numpy(sens)
    block_energy = torch.ones(n_blocks, dtype=torch.float64)

    return D, block_sensitivity, block_energy, circuits


def measure_selection(
    kept: List[int],
    block_sensitivity: torch.Tensor,
    D: torch.Tensor,
    circuits: List[List[int]],
    feature_multiplier: int = 3,
) -> Dict[str, float]:
    """Compute quality proxy metrics for a selection."""
    sens = block_sensitivity.numpy()
    n_blocks = len(sens)
    F = feature_multiplier
    kept_set = set(kept)

    # 1. Sensitivity preserved
    total_sens = sens.sum()
    kept_sens = sum(sens[b] for b in kept)
    sens_preserved = kept_sens / (total_sens + 1e-30)

    # 2. Coupling preserved: what fraction of total off-diagonal coupling
    #    exists between kept blocks
    D_np = D.numpy()
    total_coupling = 0.0
    kept_coupling = 0.0
    for a in range(n_blocks):
        for b in range(a + 1, n_blocks):
            c = 0.0
            for fi in range(F):
                for fj in range(F):
                    c += abs(D_np[a * F + fi, b * F + fj])
            total_coupling += c
            if a in kept_set and b in kept_set:
                kept_coupling += c
    coupling_preserved = kept_coupling / (total_coupling + 1e-30)

    # 3. Circuit survival: fraction of planted circuits fully preserved
    circuits_survived = 0
    for group in circuits:
        if all(b in kept_set for b in group):
            circuits_survived += 1
    circuit_survival = circuits_survived / max(len(circuits), 1)

    # 4. Combined (geometric mean)
    combined = (sens_preserved * coupling_preserved * max(circuit_survival, 1e-6)) ** (1 / 3)

    return {
        "sensitivity_preserved": round(sens_preserved, 4),
        "coupling_preserved": round(coupling_preserved, 4),
        "circuit_survival": round(circuit_survival, 4),
        "combined": round(combined, 4),
    }


def run_comparison(
    n_layers: int = 32,
    n_blocks: int = 32,
    keep_fracs: List[float] = None,
    seed: int = 42,
) -> None:
    if keep_fracs is None:
        keep_fracs = [0.90, 0.85, 0.80, 0.75, 0.70]

    F = 3
    rng = np.random.RandomState(seed)

    selectors = {
        "cohesion": lambda D, kf, bs, be: select_blocks_cohesion(
            D, kf, block_size=1, block_sensitivity=bs,
            block_energy=be, feature_multiplier=F,
            rng=np.random.RandomState(seed), cohesion_lambda=0.15,
        ),
        "structural": lambda D, kf, bs, be: select_blocks_structural(
            D, kf, block_size=1, block_sensitivity=bs,
            block_energy=be, feature_multiplier=F,
            rng=np.random.RandomState(seed), diversity_lambda=0.2,
        ),
        "sensitivity": lambda D, kf, bs, be: select_blocks_sensitivity(
            bs, kf, block_size=1, block_energy=be, feature_multiplier=F,
        ),
    }

    print("=" * 78)
    print("  Selector Comparison (CPU, synthetic data with planted circuits)")
    print(f"  {n_layers} layers × {n_blocks} blocks, F={F}, seed={seed}")
    print("=" * 78)

    # Aggregate results
    agg: Dict[str, Dict[str, List[float]]] = {
        name: {m: [] for m in ["sensitivity_preserved", "coupling_preserved",
                                "circuit_survival", "combined"]}
        for name in selectors
    }

    for kf in keep_fracs:
        print(f"\n{'─' * 78}")
        print(f"  keep_frac = {kf:.2f}  ({(1-kf)*100:.0f}% pruned)")
        print(f"{'─' * 78}")
        header = f"  {'Selector':<16} {'Sens.Pres':>10} {'Coupl.Pres':>11} {'CircuitSurv':>12} {'Combined':>9}"
        print(header)
        print(f"  {'─'*16} {'─'*10} {'─'*11} {'─'*12} {'─'*9}")

        for name, selector_fn in selectors.items():
            layer_metrics = {m: [] for m in ["sensitivity_preserved", "coupling_preserved",
                                              "circuit_survival", "combined"]}
            for li in range(n_layers):
                layer_rng = np.random.RandomState(seed + li)
                D, bs, be, circuits = make_synthetic_layer(
                    n_blocks=n_blocks, feature_multiplier=F,
                    n_circuits=4, circuit_size=3,
                    coupling_strength=4.0, rng=layer_rng,
                )
                kept, _, _ = selector_fn(D, kf, bs, be)
                metrics = measure_selection(kept, bs, D, circuits, F)
                for m in layer_metrics:
                    layer_metrics[m].append(metrics[m])

            avgs = {m: np.mean(layer_metrics[m]) for m in layer_metrics}
            for m in avgs:
                agg[name][m].append(avgs[m])

            print(f"  {name:<16} {avgs['sensitivity_preserved']:>10.4f} "
                  f"{avgs['coupling_preserved']:>11.4f} "
                  f"{avgs['circuit_survival']:>12.4f} "
                  f"{avgs['combined']:>9.4f}")

    # Summary: average across all keep_fracs
    print(f"\n{'=' * 78}")
    print("  OVERALL AVERAGE (across all keep_fracs)")
    print(f"{'=' * 78}")
    header = f"  {'Selector':<16} {'Sens.Pres':>10} {'Coupl.Pres':>11} {'CircuitSurv':>12} {'Combined':>9}"
    print(header)
    print(f"  {'─'*16} {'─'*10} {'─'*11} {'─'*12} {'─'*9}")

    for name in selectors:
        avgs = {m: np.mean(agg[name][m]) for m in agg[name]}
        print(f"  {name:<16} {avgs['sensitivity_preserved']:>10.4f} "
              f"{avgs['coupling_preserved']:>11.4f} "
              f"{avgs['circuit_survival']:>12.4f} "
              f"{avgs['combined']:>9.4f}")

    # Delta table
    print(f"\n{'=' * 78}")
    print("  COHESION vs STRUCTURAL (per keep_frac delta)")
    print(f"{'=' * 78}")
    print(f"  {'KF':<8} {'Δ Sens':>10} {'Δ Coupling':>11} {'Δ Circuit':>12} {'Δ Combined':>11}")
    print(f"  {'─'*8} {'─'*10} {'─'*11} {'─'*12} {'─'*11}")
    for i, kf in enumerate(keep_fracs):
        deltas = {m: agg["cohesion"][m][i] - agg["structural"][m][i]
                  for m in agg["cohesion"]}
        print(f"  {kf:<8.2f} {deltas['sensitivity_preserved']:>+10.4f} "
              f"{deltas['coupling_preserved']:>+11.4f} "
              f"{deltas['circuit_survival']:>+12.4f} "
              f"{deltas['combined']:>+11.4f}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Local selector comparison (CPU)")
    parser.add_argument("--n-layers", type=int, default=32)
    parser.add_argument("--n-blocks", type=int, default=32)
    parser.add_argument("--keep-fracs", default="0.90,0.85,0.80,0.75,0.70")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    keep_fracs = [float(x) for x in args.keep_fracs.split(",")]

    t0 = time.time()
    run_comparison(
        n_layers=args.n_layers,
        n_blocks=args.n_blocks,
        keep_fracs=keep_fracs,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
