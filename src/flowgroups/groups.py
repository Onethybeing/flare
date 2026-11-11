"""Candidate-group construction and group scoring.

Three group builders operate on the transactions of one time step:

* :func:`flow_groups` (proposed): seeds are the top ``seed_frac`` transactions by risk; each seed
  expands for at most ``hops`` hops through neighbours whose risk is in the step's top ``q_exp``
  quantile, never through hub transactions (degree > ``hub_degree``). Connected pieces that
  contain a seed become groups; pieces larger than ``max_size`` are split with Louvain.
* :func:`cc_groups` (baseline): connected components of the risk-thresholded sub-graph, with no
  hop limit, hub filter or size cap.
* :func:`ego_groups` (baseline): each seed plus all of its 1-hop neighbours.

Group features (one row per group): ``n, max, mean, top3, sum, motif, exposure``.
``motif`` is the share of members that split (out-degree >= 2), merge (in-degree >= 2) or pass
funds through (in = out = 1) inside the group; ``exposure`` is the mean known-bad-address
indicator of the members (computed from training-period labels only).
"""
from __future__ import annotations

import math

import numpy as np

from .graph import StepGraph

FEATURE_NAMES = ["n", "max", "mean", "top3", "sum", "motif", "exposure"]
_BASE = {"max": 1, "mean": 2, "top3": 3, "sum": 4}


def select_seeds(nodes: np.ndarray, risk: np.ndarray, seed_frac: float) -> np.ndarray:
    """Top ``max(1, round(seed_frac * len(nodes)))`` nodes by risk (descending)."""
    nodes = np.asarray(nodes)
    k = max(1, int(round(seed_frac * len(nodes))))
    return nodes[np.argsort(-risk[nodes], kind="stable")][:k]


def _split_large(G: StepGraph, comp, seedset, max_size):
    import networkx as nx

    H = nx.Graph()
    H.add_nodes_from(comp)
    cs = set(comp)
    for v in comp:
        H.add_edges_from((v, int(u)) for u in G.neighbors(v) if int(u) in cs)
    parts = nx.community.louvain_communities(H, seed=0)
    return [list(p) for p in parts if set(p) & seedset]


def flow_groups(G: StepGraph, nodes, risk, seed_frac=0.01, q_exp=0.1, hops=2, hub_degree=30,
                max_size=40, hub_filter=True, size_cap=True):
    """Flow-constrained candidate groups for one time step (see module docstring)."""
    nodes = np.asarray(nodes)
    seeds = select_seeds(nodes, risk, seed_frac)
    seedset = set(int(s) for s in seeds)
    tau = np.quantile(risk[nodes], 1 - q_exp)
    U = set(seedset)
    for s0 in seeds:
        s0 = int(s0)
        if hub_filter and G.degree[s0] > hub_degree:
            continue
        frontier = [s0]
        for _ in range(hops):
            nxt = []
            for v in frontier:
                for u in G.neighbors(v):
                    u = int(u)
                    if u in U or risk[u] < tau:
                        continue
                    if hub_filter and G.degree[u] > hub_degree:
                        continue
                    U.add(u)
                    nxt.append(u)
            frontier = nxt
    groups = []
    for comp in G.components(U):
        if not set(comp) & seedset:
            continue
        if size_cap and len(comp) > max_size:
            groups += _split_large(G, comp, seedset, max_size)
        else:
            groups.append(comp)
    return [np.array(sorted(g), dtype=np.int64) for g in groups]


def cc_groups(G: StepGraph, nodes, risk, seed_frac=0.01, q_exp=0.1):
    """Connected components of the risk-thresholded sub-graph that contain a seed."""
    nodes = np.asarray(nodes)
    seeds = select_seeds(nodes, risk, seed_frac)
    seedset = set(int(s) for s in seeds)
    tau = np.quantile(risk[nodes], 1 - q_exp)
    U = set(int(v) for v in nodes[risk[nodes] >= tau]) | seedset
    return [np.array(sorted(c), dtype=np.int64) for c in G.components(U) if set(c) & seedset]


def ego_groups(G: StepGraph, nodes, risk, seed_frac=0.01):
    """Each seed together with all of its 1-hop neighbours (groups may overlap)."""
    seeds = select_seeds(np.asarray(nodes), risk, seed_frac)
    return [np.array(sorted({int(s)} | set(int(u) for u in G.neighbors(int(s)))), dtype=np.int64) for s in seeds]


def group_features(G: StepGraph, group, risk, exposure=None) -> np.ndarray:
    """Feature vector ``[n, max, mean, top3, sum, motif, exposure]`` for one group."""
    g = np.asarray(group, dtype=np.int64)
    n = len(g)
    rs = np.sort(risk[g])[::-1]
    if n > 1:
        indeg, outdeg, _ = G.flow_degrees(g)
        motif = ((outdeg >= 2).sum() + (indeg >= 2).sum() + ((indeg == 1) & (outdeg == 1)).sum()) / n
    else:
        motif = 0.0
    exp_ = float(np.mean(exposure[g])) if exposure is not None else 0.0
    return np.array([n, rs[0], rs.mean(), rs[:3].mean(), rs.sum(), motif, exp_], dtype=float)


def group_score(F: np.ndarray, kind: str = "sum", w=(0.0, 0.0, 0.0, "sum")) -> np.ndarray:
    """Score groups from their feature matrix ``F`` (rows from :func:`group_features`).

    ``kind`` in {"max", "mean", "top3", "sum"} returns that column. ``kind="ours"`` returns::

        base * (1 + beta * motif + gamma * exposure) * n ** (-delta)

    with ``w = (beta, gamma, delta, base)``. ``delta = 0`` keeps the sum-like behaviour,
    ``delta = 1`` turns a sum base into a mean-like score.
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    if kind in _BASE:
        return F[:, _BASE[kind]]
    if kind != "ours":
        raise ValueError(f"unknown score kind: {kind}")
    beta, gamma, delta, base = w
    return F[:, _BASE[base]] * (1 + beta * F[:, 5] + gamma * F[:, 6]) * F[:, 0] ** (-delta)


def budget_for(n_step: int, budget_frac: float) -> int:
    """Review capacity for a step: ``ceil(budget_frac * n_step)``, at least 1."""
    return max(1, int(math.ceil(budget_frac * n_step)))
