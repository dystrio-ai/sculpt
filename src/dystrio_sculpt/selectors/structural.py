"""Structural block selection: operator-fidelity + coupling-geometry diversity.

Pipeline:
  1. Base score  = operator sensitivity (how much zeroing block changes MLP output)
  2. Modulated   by normalised block energy
  3. Novelty     from cross-layer selection history (penalises blocks that are
                 selected in every layer, encouraging structural diversity)
  4. Diversity   from coupling graph (covariance -> correlation -> Physarum conductance)
                 penalises selecting tightly-coupled neighbours

Cross-layer novelty filtering inspired by ShinkaEvolve (Sakana AI,
arXiv:2509.19349) novelty-based rejection sampling.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .._calibrate import (
    collect_block_geometry_swiglu,
    collect_block_operator_sensitivity_swiglu,
)

DEFAULT_NOVELTY_LAMBDA = 0.15


# ── Graph construction ────────────────────────────────────────────────────────


def build_graph_from_cov(
    D: torch.Tensor, k: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert block covariance to a sparse correlation-based edge list."""
    D_np = D.numpy().astype(np.float64)
    n = D_np.shape[0]
    std = np.sqrt(np.diag(D_np).clip(1e-30))
    W = np.abs(D_np / np.outer(std, std))
    np.fill_diagonal(W, 0.0)

    mask = np.zeros_like(W, dtype=bool)
    for i in range(n):
        topk = np.argpartition(W[i], -min(k, n - 1))[-min(k, n - 1):]
        mask[i, topk] = True
    mask = mask | mask.T
    W = W * mask

    rows, cols = np.where((W > 0) & (np.arange(n)[:, None] < np.arange(n)[None, :]))
    return rows.astype(np.int64), cols.astype(np.int64), W[rows, cols].astype(np.float64)


# ── Physarum conductance ──────────────────────────────────────────────────────


def physarum_conductance(
    u: np.ndarray, v: np.ndarray, w: np.ndarray,
    n_nodes: int, n_iters: int = 200, mu: float = 1.5, eps: float = 1e-8,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Physarum-inspired conductance learning on an undirected weighted graph.

    Uses *rng* for source/sink selection to support deterministic mode.
    """
    n_edges = len(u)
    if n_edges == 0:
        return np.zeros(0, dtype=np.float64)
    if rng is None:
        rng = np.random.RandomState()

    k = w.copy() + eps
    for _ in range(n_iters):
        src = rng.randint(0, n_nodes)
        snk = (src + n_nodes // 2) % n_nodes
        G = k + eps
        L = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        for e in range(n_edges):
            L[u[e], u[e]] += G[e]
            L[v[e], v[e]] += G[e]
            L[u[e], v[e]] -= G[e]
            L[v[e], u[e]] -= G[e]
        L += eps * np.eye(n_nodes)
        rhs = np.zeros(n_nodes, dtype=np.float64)
        rhs[src] = 1.0
        rhs[snk] = -1.0
        try:
            p = np.linalg.solve(L, rhs)
        except np.linalg.LinAlgError:
            continue
        flow = np.abs(G * (p[u] - p[v]))
        k = 0.95 * k + 0.05 * (np.power(flow + eps, mu) + eps)
    return k


# ── Cross-layer novelty tracking ──────────────────────────────────────────────


class CrossLayerNoveltyTracker:
    """Conductance-aware cross-layer novelty tracking.

    Combines ShinkaEvolve-inspired novelty rejection (Sakana AI,
    arXiv:2509.19349) with Physarum conductance coupling to perform
    *structural deduplication* across layers.

    Plain frequency counting treats blocks as independent: "block 5 was
    selected in 3/4 layers, penalize it."  But Physarum conductance
    reveals that blocks 5 and 6 are structurally coupled.  Selecting
    block 6 in a new layer provides redundant structural coverage even
    though block 6's raw count is low.

    The hybrid approach propagates selection counts through the running
    average conductance adjacency, so structurally-coupled neighbours of
    frequently-selected blocks also receive a novelty penalty.
    """

    def __init__(
        self,
        novelty_lambda: float = DEFAULT_NOVELTY_LAMBDA,
        conductance_diffusion: float = 0.3,
    ):
        self.novelty_lambda = novelty_lambda
        self.conductance_diffusion = conductance_diffusion
        self._block_counts: Optional[np.ndarray] = None
        self._adj_accum: Optional[np.ndarray] = None
        self._n_layers: int = 0

    def record(
        self,
        kept_blocks: List[int],
        n_blocks: int,
        block_adj: Optional[np.ndarray] = None,
    ) -> None:
        """Register a layer's block selection and conductance adjacency.

        *block_adj*, when provided, is the row-normalised block-level
        adjacency from the Physarum conductance graph.  It is accumulated
        as a running average across layers so that the novelty multiplier
        reflects the model's average coupling topology.
        """
        if self._block_counts is None:
            self._block_counts = np.zeros(n_blocks, dtype=np.float64)
        for b in kept_blocks:
            if b < len(self._block_counts):
                self._block_counts[b] += 1.0

        if block_adj is not None:
            if self._adj_accum is None:
                self._adj_accum = np.zeros_like(block_adj)
            self._adj_accum += block_adj[:n_blocks, :n_blocks]

        self._n_layers += 1

    def novelty_multiplier(self, n_blocks: int) -> Optional[np.ndarray]:
        """Per-block multiplier in [1.0, 1.0 + novelty_lambda].

        When conductance adjacency is available, raw selection frequency
        is diffused through the coupling graph: if block *i* was selected
        often AND block *j* is structurally coupled to *i*, then *j* also
        receives a (softer) novelty penalty.  The ``conductance_diffusion``
        parameter controls how strongly the coupling propagates (0 = pure
        frequency counting, 1 = full one-hop diffusion).

        Returns None when no history is available yet.
        """
        if self._n_layers == 0 or self._block_counts is None:
            return None

        counts = self._block_counts[:n_blocks]
        frequency = counts / self._n_layers

        if self._adj_accum is not None and self.conductance_diffusion > 0:
            avg_adj = self._adj_accum[:n_blocks, :n_blocks] / self._n_layers
            row_max = avg_adj.max(axis=1, keepdims=True) + 1e-30
            adj_norm = avg_adj / row_max
            diffused = frequency + self.conductance_diffusion * (adj_norm @ frequency)
            diffused = np.clip(diffused / (diffused.max() + 1e-30), 0.0, 1.0)
        else:
            diffused = frequency

        novelty = 1.0 - diffused
        return 1.0 + self.novelty_lambda * novelty

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def block_counts(self) -> Optional[np.ndarray]:
        return self._block_counts.copy() if self._block_counts is not None else None


# ── Selection ─────────────────────────────────────────────────────────────────


def select_blocks_structural(
    D: torch.Tensor,
    keep_frac: float,
    block_size: int,
    topk_edges: int = 20,
    n_physarum_iters: int = 200,
    diversity_lambda: float = 0.2,
    block_energy: torch.Tensor | None = None,
    feature_multiplier: int = 3,
    block_sensitivity: torch.Tensor | None = None,
    rng: np.random.RandomState | None = None,
    cross_layer_novelty: np.ndarray | None = None,
) -> Tuple[List[int], torch.Tensor, Dict[str, object]]:
    """Structural block selection (operator-fidelity + coupling diversity).

    When *cross_layer_novelty* is provided (from a CrossLayerNoveltyTracker),
    base scores are modulated to favour blocks that weren't frequently selected
    in earlier layers.
    """
    n_feat = D.shape[0]
    F = feature_multiplier
    n_blocks = n_feat // F
    if n_blocks == 0:
        n_blocks = max(1, n_feat)
        F = 1
    keep_n = max(1, int(math.ceil(keep_frac * n_blocks)))

    u, v, w = build_graph_from_cov(D, k=topk_edges)
    k_edge = physarum_conductance(u, v, w, n_feat, n_iters=n_physarum_iters, rng=rng)

    # Block-level adjacency for diversity penalty
    adj = np.zeros((n_blocks, n_blocks), dtype=np.float64)
    for e in range(len(u)):
        bu, bv = int(u[e]) // F, int(v[e]) // F
        if bu != bv:
            adj[bu, bv] += k_edge[e]
            adj[bv, bu] += k_edge[e]
    adj_norm = adj / (adj.max(axis=1, keepdims=True) + 1e-30)

    # Base scores
    if block_sensitivity is not None:
        sens = np.asarray(block_sensitivity, dtype=np.float64)[:n_blocks]
        raw_scores = sens.copy()
    else:
        raw_scores = np.zeros(n_blocks, dtype=np.float64)
        for e in range(len(u)):
            bu, bv = int(u[e]) // F, int(v[e]) // F
            raw_scores[bu] += k_edge[e]
            if bv != bu:
                raw_scores[bv] += k_edge[e]

    if block_energy is not None:
        be = np.asarray(block_energy, dtype=np.float64)[:n_blocks]
        be_norm = be / (be.max() + 1e-30)
        raw_scores = raw_scores * (0.5 + 0.5 * be_norm)

    # Cross-layer novelty modulation
    if cross_layer_novelty is not None:
        raw_scores = raw_scores * cross_layer_novelty[:n_blocks]

    # Greedy selection with diversity penalty
    scores = raw_scores.copy()
    selected: List[int] = []
    for _ in range(keep_n):
        best = int(np.argmax(scores))
        selected.append(best)
        scores[best] = -1e30
        if diversity_lambda > 0:
            scores -= diversity_lambda * adj_norm[best] * raw_scores[best]
    selected.sort()

    ffn = n_blocks * block_size
    idx: List[int] = []
    for b in selected:
        lo = b * block_size
        hi = min(ffn, (b + 1) * block_size)
        idx.extend(range(lo, hi))

    edges = (
        torch.tensor(np.stack([u, v, w], axis=1), dtype=torch.float64)
        if len(u) > 0
        else torch.zeros(0, 3, dtype=torch.float64)
    )
    artifacts = {
        "edges": edges,
        "k_edge": torch.from_numpy(k_edge),
        "block_scores": torch.from_numpy(raw_scores),
        "block_adj_norm": adj_norm,
    }
    return selected, torch.tensor(idx, dtype=torch.long), artifacts


# ── Prescan ───────────────────────────────────────────────────────────────────


def prescan_structural_artifacts(
    model,
    tokenizer,
    layers: Sequence[int],
    texts_cal: Sequence[str],
    max_len: int,
    device: str,
    block_size: int = 128,
    max_tokens: int = 30_000,
    adapter=None,
) -> Dict[int, Dict[str, Any]]:
    """Precompute structural tensors on an uncompressed model.

    Returns {layer_idx: {D, block_energy, block_sensitivity, feature_multiplier}}
    with all tensors on CPU.

    When *adapter* is provided, calibration is dispatched through it instead
    of calling the SwiGLU-specific functions directly.
    """
    out: Dict[int, Dict[str, Any]] = {}
    for li in layers:
        if adapter is not None:
            geom = adapter.collect_block_geometry(
                model, tokenizer, li, texts_cal, max_len, device,
                block_size=block_size, max_tokens=max_tokens,
            )
            sens = adapter.collect_block_sensitivity(
                model, tokenizer, li, texts_cal, max_len, device,
                block_size=block_size, max_tokens=max_tokens,
            )
        else:
            geom = collect_block_geometry_swiglu(
                model, tokenizer, li, texts_cal, max_len, device,
                block_size=block_size, max_tokens=max_tokens,
            )
            sens = collect_block_operator_sensitivity_swiglu(
                model, tokenizer, li, texts_cal, max_len, device,
                block_size=block_size, max_tokens=max_tokens,
            )
        out[li] = {
            "D": geom["D"].detach().cpu(),
            "block_energy": (
                geom["block_energy"].detach().cpu()
                if geom.get("block_energy") is not None else None
            ),
            "block_sensitivity": sens["block_sensitivity"].detach().cpu(),
            "feature_multiplier": geom.get("feature_multiplier", 3),
        }
    return out
