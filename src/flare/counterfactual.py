"""Counterfactual evidence test for a group score (requires PyTorch).

1. Compute the group score ``S(G)`` with all edge weights equal to 1.
2. Rank the edges in the group's receptive field by the gradient ``dS/dw_e``.
3. Delete (set ``w_e = 0``) the top-``m`` edges and recompute ``S``; the score drop is
   ``Delta_top``. Compare against deleting ``m`` random edges, the ``m`` lowest-ranked edges and
   a simple heuristic (edges touching the highest-risk member).

A useful explanation produces ``Delta_top`` clearly larger than the random baseline.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def topk_mean(p, k: int = 3):
    return torch.topk(p, min(k, p.shape[0])).values.mean()


def edge_saliency(model, x, src, dst, group_idx, score_fn=topk_mean):
    """Return (score, gradient per edge) for the group's score w.r.t. edge weights."""
    w = torch.ones(src.shape[0], dtype=x.dtype, requires_grad=True)
    s = score_fn(torch.sigmoid(model(x, src, dst, w))[group_idx])
    (g,) = torch.autograd.grad(s, w)
    return float(s.detach()), g.detach().cpu().numpy()


def deletion_drop(model, x, src, dst, group_idx, remove, score_fn=topk_mean) -> float:
    """Score after setting the weights of ``remove`` edges to zero (no gradient)."""
    w = torch.ones(src.shape[0], dtype=x.dtype)
    if len(remove):
        w[torch.as_tensor(np.asarray(remove), dtype=torch.long)] = 0
    with torch.no_grad():
        return float(score_fn(torch.sigmoid(model(x, src, dst, w))[group_idx]))


def deletion_test(model, x, src, dst, group_idx, m: int = 3, n_random: int = 10, seed: int = 0,
                  score_fn=topk_mean) -> dict | None:
    """Score drops for gradient-top, random and gradient-bottom deletions of ``m`` edges."""
    s0, grad = edge_saliency(model, x, src, dst, group_idx, score_fn)
    cand = np.where(np.abs(grad) > 0)[0]
    if len(cand) < max(2, m):
        return None
    ranked = cand[np.argsort(-grad[cand])]
    rng = np.random.default_rng(seed)
    rnd = [deletion_drop(model, x, src, dst, group_idx, rng.choice(cand, m, replace=False), score_fn)
           for _ in range(n_random)]
    return dict(score=s0,
                d_top=s0 - deletion_drop(model, x, src, dst, group_idx, ranked[:m], score_fn),
                d_random=s0 - float(np.mean(rnd)),
                d_bottom=s0 - deletion_drop(model, x, src, dst, group_idx, ranked[-m:], score_fn),
                n_candidates=int(len(cand)))
