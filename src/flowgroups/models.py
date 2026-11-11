"""Direction-aware GraphSAGE with edge-weighted mean aggregation (requires PyTorch).

Layer update for node ``v`` (directed variant)::

    h_v' = ReLU( W [ h_v || mean_{u -> v} w_uv h_u || mean_{v -> u} w_vu h_u ] )

Incoming and outgoing money-flow neighbours are aggregated separately, so the model can tell
"where funds came from" from "where funds went". Edge weights ``w`` default to 1; setting a
weight to 0 removes that edge, which the counterfactual test uses, and the weights are
differentiable, which gives edge saliency.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None


if torch is not None:

    class DirSAGE(nn.Module):
        """Two-layer GraphSAGE; ``graph=False`` gives an MLP, ``directed=False`` a symmetric variant."""

        def __init__(self, in_dim: int, hidden: int = 128, graph: bool = True, directed: bool = True,
                     dropout: float = 0.3):
            super().__init__()
            self.graph, self.directed, self.p = graph, directed, dropout
            k = (3 if directed else 2) if graph else 1
            self.l1 = nn.Linear(in_dim * k, hidden)
            self.l2 = nn.Linear(hidden * k, hidden)
            self.out = nn.Linear(hidden, 1)

        def mix(self, h, src, dst, w):
            if not self.graph:
                return h
            n = h.shape[0]

            def agg(a, b):  # mean of messages a -> b
                o = torch.zeros_like(h).index_add_(0, b, h[a] * w[:, None])
                dg = torch.zeros(n, device=h.device, dtype=h.dtype).index_add_(0, b, w)
                return o / (dg[:, None] + 1e-9)

            if self.directed:
                return torch.cat([h, agg(src, dst), agg(dst, src)], 1)
            o = torch.zeros_like(h).index_add_(0, dst, h[src] * w[:, None]).index_add_(0, src, h[dst] * w[:, None])
            dg = torch.zeros(n, device=h.device, dtype=h.dtype).index_add_(0, dst, w).index_add_(0, src, w)
            return torch.cat([h, o / (dg[:, None] + 1e-9)], 1)

        def forward(self, x, src, dst, w=None):
            if w is None:
                w = torch.ones(src.shape[0], device=x.device, dtype=x.dtype)
            h = F.dropout(F.relu(self.l1(self.mix(x, src, dst, w))), self.p, self.training)
            h = F.dropout(F.relu(self.l2(self.mix(h, src, dst, w))), self.p, self.training)
            return self.out(h).squeeze(-1)
else:  # pragma: no cover
    DirSAGE = None
