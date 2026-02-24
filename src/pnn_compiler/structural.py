"""Structural block selection via operator-fidelity + coupling-geometry diversity.

v3 pipeline:
  1. Base score  = operator sensitivity (how much zeroing block changes MLP output)
  2. Modulated   by normalised block energy
  3. Diversity   from coupling graph (D_cov -> correlation -> Physarum conductance)
                 penalises selecting tightly-coupled neighbours

Falls back to v2 (conductance centrality) when block_sensitivity is not provided.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import numpy as np


# ── Graph construction ────────────────────────────────────────────────────────


def build_graph_from_cov(
    D: torch.Tensor,
    mode: str = "corr",
    sparsify: str = "topk",
    k: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert block covariance to a sparse weighted edge list.

    Returns (u, v, w) arrays — each of length n_edges — representing an
    undirected graph over blocks.
    """
    D_np = D.numpy().astype(np.float64)
    n = D_np.shape[0]

    if mode == "corr":
        std = np.sqrt(np.diag(D_np).clip(1e-30))
        W = D_np / np.outer(std, std)
        np.fill_diagonal(W, 0.0)
        W = np.abs(W)
    elif mode == "cov":
        W = np.abs(D_np.copy())
        np.fill_diagonal(W, 0.0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if sparsify == "topk":
        mask = np.zeros_like(W, dtype=bool)
        for i in range(n):
            row = W[i]
            if k >= n:
                topk_idx = np.arange(n)
            else:
                topk_idx = np.argpartition(row, -k)[-k:]
            mask[i, topk_idx] = True
        mask = mask | mask.T
        W = W * mask
    elif sparsify == "none":
        pass
    else:
        raise ValueError(f"Unknown sparsify: {sparsify}")

    rows, cols = np.where((W > 0) & (np.arange(n)[:, None] < np.arange(n)[None, :]))
    weights = W[rows, cols]

    return rows.astype(np.int64), cols.astype(np.int64), weights.astype(np.float64)


# ── Physarum conductance learning ─────────────────────────────────────────────


def physarum_conductance(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    n_nodes: int,
    n_iters: int = 200,
    mu: float = 1.5,
    eps: float = 1e-8,
) -> np.ndarray:
    """Physarum-inspired conductance learning on an undirected weighted graph.

    Each edge carries a "conductance" k that adapts based on flow:
        Q_e = k_e * |p_u - p_v|  (Ohm-like)
        dk/dt ∝ |Q_e|^mu - k_e   (reinforcement + decay)

    Pressures are solved by injecting +1 at a source and -1 at a sink,
    cycling sources/sinks through all nodes to get a consensus conductance.
    """
    n_edges = len(u)
    if n_edges == 0:
        return np.zeros(0, dtype=np.float64)

    k = w.copy() + eps

    for _ in range(n_iters):
        src = np.random.randint(0, n_nodes)
        snk = (src + n_nodes // 2) % n_nodes

        G = np.zeros(n_edges, dtype=np.float64)
        G[:] = k + eps

        L = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        for e in range(n_edges):
            L[u[e], u[e]] += G[e]
            L[v[e], v[e]] += G[e]
            L[u[e], v[e]] -= G[e]
            L[v[e], u[e]] -= G[e]

        # Regularise for invertibility
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


# ── Block scoring from conductance ────────────────────────────────────────────


def block_scores_from_conductance(
    u: np.ndarray,
    v: np.ndarray,
    k: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Score each block by its total learned conductance (weighted degree centrality)."""
    score = np.zeros(n_nodes, dtype=np.float64)
    for e in range(len(u)):
        score[u[e]] += k[e]
        score[v[e]] += k[e]
    return score


# ── Geometry diagnostics ──────────────────────────────────────────────────────


def geometry_diagnostics(
    D: torch.Tensor,
    k_edge: np.ndarray,
) -> Dict[str, float]:
    """Compute summary statistics for the coupling geometry."""
    D_np = D.numpy().astype(np.float64)
    eigs = np.linalg.eigvalsh(D_np)
    eigs = eigs[eigs > 0][::-1]
    if len(eigs) == 0:
        return {"eff_rank95_D": 0.0, "gini_k": 0.0, "top10_edge_mass": 0.0}

    cumvar = np.cumsum(eigs) / np.sum(eigs)
    eff_rank95 = float(np.searchsorted(cumvar, 0.95) + 1)

    if len(k_edge) == 0:
        return {"eff_rank95_D": eff_rank95, "gini_k": 0.0, "top10_edge_mass": 0.0}

    k_sorted = np.sort(k_edge)
    n = len(k_sorted)
    idx = np.arange(1, n + 1)
    gini = float((2.0 * np.sum(idx * k_sorted) / (n * np.sum(k_sorted) + 1e-30)) - (n + 1) / n)

    k_desc = np.sort(k_edge)[::-1]
    total_k = k_desc.sum() + 1e-30
    top10 = int(max(1, min(10, n)))
    top10_mass = float(k_desc[:top10].sum() / total_k)

    return {
        "eff_rank95_D": eff_rank95,
        "gini_k": round(gini, 4),
        "top10_edge_mass": round(top10_mass, 4),
    }


# ── Diversity-penalized greedy selection ──────────────────────────────────────


def _aggregate_feature_conductance_to_blocks(
    u: np.ndarray,
    v: np.ndarray,
    k_edge: np.ndarray,
    n_blocks: int,
    feat_per_block: int,
) -> np.ndarray:
    """Sum edge conductance incident to any feature node of a block.

    Feature indices for block b are [b*F, b*F+1, ..., b*F+F-1].
    Block score = sum of k_edge over all edges touching any of those nodes.
    """
    score = np.zeros(n_blocks, dtype=np.float64)
    for e in range(len(u)):
        block_u = int(u[e]) // feat_per_block
        block_v = int(v[e]) // feat_per_block
        score[block_u] += k_edge[e]
        if block_v != block_u:
            score[block_v] += k_edge[e]
    return score


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
) -> Tuple[List[int], torch.Tensor, Dict[str, object]]:
    """Structural block selection (v3: operator-fidelity + coupling diversity).

    When block_sensitivity is provided (v3), it is used as the base score
    (how much zeroing the block changes MLP output).  The coupling graph from
    D is used only for the diversity penalty during greedy selection.

    When block_sensitivity is None, falls back to v2 behaviour (conductance
    centrality as base score).

    Returns (kept_blocks, kept_idx, artifacts).
    """
    n_feat = D.shape[0]
    F = feature_multiplier
    n_blocks = n_feat // F
    if n_blocks == 0:
        n_blocks = max(1, n_feat)
        F = 1
    keep_blocks_n = max(1, int(math.ceil(keep_frac * n_blocks)))

    # ── Coupling graph (used for diversity in v3, or scoring in v2 fallback) ──
    u, v, w = build_graph_from_cov(D, mode="corr", sparsify="topk", k=topk_edges)
    k_edge = physarum_conductance(u, v, w, n_feat, n_iters=n_physarum_iters)

    diags = geometry_diagnostics(D, k_edge)

    # ── Block-level adjacency from coupling (always needed for diversity) ─────
    adj_weight = np.zeros((n_blocks, n_blocks), dtype=np.float64)
    for e in range(len(u)):
        bu = int(u[e]) // F
        bv = int(v[e]) // F
        if bu != bv:
            adj_weight[bu, bv] += k_edge[e]
            adj_weight[bv, bu] += k_edge[e]
    # Normalise rows so penalty is relative
    row_max = adj_weight.max(axis=1, keepdims=True) + 1e-30
    adj_norm = adj_weight / row_max

    # ── Base scores ───────────────────────────────────────────────────────────
    if block_sensitivity is not None:
        # v3: operator-fidelity as base score
        sens = (block_sensitivity.numpy().astype(np.float64)
                if isinstance(block_sensitivity, torch.Tensor)
                else np.asarray(block_sensitivity, dtype=np.float64))
        sens = sens[:n_blocks]
        raw_scores = sens.copy()
    else:
        # v2 fallback: conductance centrality
        raw_scores = _aggregate_feature_conductance_to_blocks(
            u, v, k_edge, n_blocks, F,
        )

    # Modulate by normalised block energy
    if block_energy is not None:
        be = (block_energy.numpy().astype(np.float64)
              if isinstance(block_energy, torch.Tensor)
              else np.asarray(block_energy, dtype=np.float64))
        be = be[:n_blocks]
        be_norm = be / (be.max() + 1e-30)
        raw_scores = raw_scores * (0.5 + 0.5 * be_norm)

    # ── Greedy selection with coupling-diversity penalty ──────────────────────
    scores = raw_scores.copy()
    selected: List[int] = []

    for _ in range(keep_blocks_n):
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

    kept_idx = torch.tensor(idx, dtype=torch.long)

    edges = (torch.tensor(np.stack([u, v, w], axis=1), dtype=torch.float64)
             if len(u) > 0 else torch.zeros(0, 3, dtype=torch.float64))
    artifacts = {
        "edges": edges,
        "k_edge": torch.from_numpy(k_edge),
        "block_scores": torch.from_numpy(raw_scores),
        "diagnostics": diags,
    }

    return selected, kept_idx, artifacts
